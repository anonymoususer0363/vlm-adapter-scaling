"""
Perceiver Resampler for VLM Adapter Scaling Law experiments.

Key design:
- T (num queries) and N_A (adapter params) are independently controllable
- T is controlled by num_queries
- N_A is controlled by d_model (depth is fixed at 2)
- Cross-attention: queries attend to vision encoder output
- Self-attention: queries attend to each other
- FFN: SwiGLU with 4x expansion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv is None:
            kv = q
        B, T_q, _ = q.shape
        _, T_kv, _ = kv.shape

        queries = rearrange(self.q_proj(q), "b t (h d) -> b h t d", h=self.n_heads)
        keys = rearrange(self.k_proj(kv), "b t (h d) -> b h t d", h=self.n_heads)
        values = rearrange(self.v_proj(kv), "b t (h d) -> b h t d", h=self.n_heads)

        # Scaled dot-product attention (uses flash attention when available)
        attn_out = F.scaled_dot_product_attention(queries, keys, values)
        attn_out = rearrange(attn_out, "b h t d -> b t (h d)")
        return self.o_proj(attn_out)


class PerceiverResamplerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        # Cross-attention: queries attend to vision features
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn_norm_q = nn.LayerNorm(d_model)
        self.cross_attn_norm_kv = nn.LayerNorm(d_model)

        # Self-attention: queries attend to each other
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(d_model)

        # FFN
        self.ffn = SwiGLU(d_model, d_ff)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        # Cross-attention
        q_normed = self.cross_attn_norm_q(queries)
        kv_normed = self.cross_attn_norm_kv(vision_features)
        queries = queries + self.cross_attn(q_normed, kv_normed)

        # Self-attention
        queries = queries + self.self_attn(self.self_attn_norm(queries))

        # FFN
        queries = queries + self.ffn(self.ffn_norm(queries))

        return queries


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler: maps vision encoder output to a fixed number of tokens.

    Args:
        d_vision: dimension of vision encoder output (e.g., 1152 for SigLIP-SO400M)
        d_model: internal dimension of the resampler (controls N_A)
        d_llm: dimension of LLM input embeddings
        num_queries: number of output visual tokens T
        num_layers: number of resampler layers (fixed at 2)
        n_heads: number of attention heads
        ff_mult: FFN expansion ratio
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
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_layers = num_layers

        # Input projection: vision features -> d_model
        self.input_proj = nn.Linear(d_vision, d_model, bias=False)
        self.input_norm = nn.LayerNorm(d_model)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        # Resampler layers
        d_ff = d_model * ff_mult
        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(d_model, n_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Output projection: d_model -> d_llm
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
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, T_0, d_vision) from vision encoder

        Returns:
            (B, T, d_llm) visual tokens for LLM
        """
        B = vision_features.shape[0]

        # Project vision features to d_model
        vision_features = self.input_norm(self.input_proj(vision_features))

        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)

        # Pass through resampler layers
        for layer in self.layers:
            queries = layer(queries, vision_features)

        # Output projection
        output = self.output_proj(self.output_norm(queries))
        return output


def get_adapter_config(level: str) -> dict:
    """Get predefined adapter configurations by size level."""
    configs = {
        "XS": {"d_model": 256, "n_heads": 4},
        "S":  {"d_model": 512, "n_heads": 8},
        "M":  {"d_model": 768, "n_heads": 8},
        "L":  {"d_model": 1024, "n_heads": 8},
        "XL": {"d_model": 1536, "n_heads": 16},
    }
    if level not in configs:
        raise ValueError(f"Unknown adapter level: {level}. Choose from {list(configs.keys())}")
    return configs[level]


def build_adapter(
    level: str = "M",
    num_queries: int = 64,
    d_vision: int = 1152,
    d_llm: int = 2048,
    num_layers: int = 2,
) -> PerceiverResampler:
    """Build a Perceiver Resampler with predefined size level."""
    cfg = get_adapter_config(level)
    return PerceiverResampler(
        d_vision=d_vision,
        d_model=cfg["d_model"],
        d_llm=d_llm,
        num_queries=num_queries,
        num_layers=num_layers,
        n_heads=cfg["n_heads"],
    )


if __name__ == "__main__":
    # Print parameter counts for all configurations
    print("=" * 60)
    print("Perceiver Resampler Parameter Counts")
    print("=" * 60)
    print(f"{'Level':<6} {'d_model':<10} {'T=64 N_A':<15} {'T=128 N_A':<15} {'delta%':<10}")
    print("-" * 60)

    for level in ["XS", "S", "M", "L", "XL"]:
        adapter_64 = build_adapter(level=level, num_queries=64)
        adapter_128 = build_adapter(level=level, num_queries=128)
        n64 = adapter_64.num_params
        n128 = adapter_128.num_params
        delta = (n128 - n64) / n64 * 100
        cfg = get_adapter_config(level)
        print(f"{level:<6} {cfg['d_model']:<10} {n64:>12,}   {n128:>12,}   {delta:>6.2f}%")

    print("\n-> T를 64->128로 바꿔도 N_A 변화 < 1% → T와 N_A 독립 조절 가능")
