"""
Alternative adapter architectures for G11 comparison.

Adapter types:
1. PerceiverResampler (existing) — cross/self attention + SwiGLU FFN
2. MLPProjector (LLaVA-style) — adaptive pool + 2-layer MLP
3. QFormerAdapter (BLIP2-style) — transformer decoder with cross-attention

All adapters share the same interface:
    __init__(d_vision, d_model, d_llm, num_queries, num_layers, n_heads, ff_mult)
    forward(vision_features) -> (B, T, d_llm)
    num_params -> int
    num_trainable_params -> int
    num_queries -> int
    d_model -> int
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────
# MLP Projector (LLaVA-style)
# ─────────────────────────────────────────────────────────

class MLPProjector(nn.Module):
    """
    LLaVA-style 2-layer MLP with adaptive pooling for T control.

    Pipeline:
        vision_features (B, T₀, d_vision)
        → adaptive_avg_pool1d → (B, T, d_vision)
        → Linear(d_vision, d_model) → GELU → Linear(d_model, d_llm)
        → (B, T, d_llm)

    N_A is controlled by d_model (hidden dimension).
    T is controlled by num_queries (pooling target).
    """

    def __init__(
        self,
        d_vision: int = 1152,
        d_model: int = 768,
        d_llm: int = 2048,
        num_queries: int = 64,
        num_layers: int = 2,  # stored for interface compat, always 2-layer MLP
        n_heads: int = 8,     # ignored
        ff_mult: int = 4,     # ignored
    ):
        super().__init__()
        self._d_model = d_model
        self._num_queries = num_queries
        self._num_layers = num_layers

        # Adaptive pooling: T₀ → T
        self.pool = nn.AdaptiveAvgPool1d(num_queries)

        # 2-layer MLP with GELU
        self.fc1 = nn.Linear(d_vision, d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model, d_llm)

        self.norm = nn.LayerNorm(d_llm)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @property
    def num_queries(self) -> int:
        return self._num_queries

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, T₀, d_vision)
        Returns:
            (B, T, d_llm)
        """
        # Pool: (B, T₀, d_vision) → (B, d_vision, T₀) → pool → (B, d_vision, T) → (B, T, d_vision)
        x = vision_features.transpose(1, 2)  # (B, d_vision, T₀)
        x = self.pool(x)                      # (B, d_vision, T)
        x = x.transpose(1, 2)                 # (B, T, d_vision)

        # MLP
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────────────────
# Q-Former Adapter (BLIP2-style)
# ─────────────────────────────────────────────────────────

class QFormerLayer(nn.Module):
    """Single Q-Former layer: self-attention → cross-attention → FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Self-attention (bidirectional, unlike Perceiver which is also bidirectional)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_q = nn.Linear(d_model, d_model, bias=False)
        self.self_k = nn.Linear(d_model, d_model, bias=False)
        self.self_v = nn.Linear(d_model, d_model, bias=False)
        self.self_o = nn.Linear(d_model, d_model, bias=False)

        # Cross-attention
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_kv_norm = nn.LayerNorm(d_model)
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_k = nn.Linear(d_model, d_model, bias=False)
        self.cross_v = nn.Linear(d_model, d_model, bias=False)
        self.cross_o = nn.Linear(d_model, d_model, bias=False)

        # FFN (standard MLP, not SwiGLU — key difference from Perceiver)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_fc1 = nn.Linear(d_model, d_ff)
        self.ffn_fc2 = nn.Linear(d_ff, d_model)

    def _attention(self, q, k, v):
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)
        out = F.scaled_dot_product_attention(q, k, v)
        return rearrange(out, "b h t d -> b t (h d)")

    def forward(self, queries: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = self.self_attn_norm(queries)
        attn_out = self._attention(self.self_q(x), self.self_k(x), self.self_v(x))
        queries = queries + self.self_o(attn_out)

        # Cross-attention
        q_normed = self.cross_attn_norm(queries)
        kv_normed = self.cross_kv_norm(vision_features)
        attn_out = self._attention(self.cross_q(q_normed), self.cross_k(kv_normed), self.cross_v(kv_normed))
        queries = queries + self.cross_o(attn_out)

        # FFN (standard GELU, not SwiGLU)
        x = self.ffn_norm(queries)
        x = F.gelu(self.ffn_fc1(x))
        x = self.ffn_fc2(x)
        queries = queries + x

        return queries


class QFormerAdapter(nn.Module):
    """
    BLIP2-style Q-Former adapter.

    Key differences from PerceiverResampler:
    1. Self-attention BEFORE cross-attention (Perceiver: cross first)
    2. Standard GELU FFN instead of SwiGLU
    3. Separate projection layers for self/cross attention

    Same interface as PerceiverResampler for drop-in replacement.
    """

    def __init__(
        self,
        d_vision: int = 1152,
        d_model: int = 768,
        d_llm: int = 2048,
        num_queries: int = 64,
        num_layers: int = 2,
        n_heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self._d_model = d_model
        self._num_queries = num_queries
        self._num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(d_vision, d_model, bias=False)
        self.input_norm = nn.LayerNorm(d_model)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        # Q-Former layers
        d_ff = d_model * ff_mult
        self.layers = nn.ModuleList([
            QFormerLayer(d_model, n_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_llm, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def num_queries(self) -> int:
        return self._num_queries

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, T₀, d_vision)
        Returns:
            (B, T, d_llm)
        """
        B = vision_features.shape[0]

        # Project vision features
        vision_features = self.input_norm(self.input_proj(vision_features))

        # Expand queries
        queries = self.queries.expand(B, -1, -1)

        # Q-Former layers
        for layer in self.layers:
            queries = layer(queries, vision_features)

        # Output projection
        return self.output_proj(self.output_norm(queries))


# ─────────────────────────────────────────────────────────
# Unified builder
# ─────────────────────────────────────────────────────────

ADAPTER_TYPES = {
    "perceiver": "PerceiverResampler",
    "mlp": "MLPProjector",
    "qformer": "QFormerAdapter",
}


def build_adapter_by_type(
    adapter_type: str = "perceiver",
    level: str = "M",
    num_queries: int = 64,
    d_vision: int = 1152,
    d_llm: int = 2048,
    num_layers: int = 2,
):
    """Build adapter of specified type with predefined size level."""
    from .perceiver_resampler import PerceiverResampler, get_adapter_config

    cfg = get_adapter_config(level)

    if adapter_type == "perceiver":
        return PerceiverResampler(
            d_vision=d_vision,
            d_model=cfg["d_model"],
            d_llm=d_llm,
            num_queries=num_queries,
            num_layers=num_layers,
            n_heads=cfg["n_heads"],
        )
    elif adapter_type == "mlp":
        return MLPProjector(
            d_vision=d_vision,
            d_model=cfg["d_model"],
            d_llm=d_llm,
            num_queries=num_queries,
            num_layers=num_layers,
            n_heads=cfg["n_heads"],
        )
    elif adapter_type == "qformer":
        return QFormerAdapter(
            d_vision=d_vision,
            d_model=cfg["d_model"],
            d_llm=d_llm,
            num_queries=num_queries,
            num_layers=num_layers,
            n_heads=cfg["n_heads"],
        )
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}. Choose from {list(ADAPTER_TYPES.keys())}")


if __name__ == "__main__":
    """Print parameter counts for all adapter types and configurations."""
    from perceiver_resampler import get_adapter_config

    print("=" * 80)
    print("Adapter Parameter Counts Comparison")
    print("=" * 80)

    for adapter_type in ["perceiver", "mlp", "qformer"]:
        print(f"\n--- {adapter_type} ---")
        print(f"{'Level':<6} {'d_model':<10} {'T=64 N_A':<15} {'T=128 N_A':<15}")
        print("-" * 50)

        for level in ["XS", "S", "M", "L", "XL"]:
            cfg = get_adapter_config(level)
            a64 = build_adapter_by_type(adapter_type, level=level, num_queries=64)
            a128 = build_adapter_by_type(adapter_type, level=level, num_queries=128)
            print(f"{level:<6} {cfg['d_model']:<10} {a64.num_params:>12,}   {a128.num_params:>12,}")
