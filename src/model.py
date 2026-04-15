"""
VLM model for scaling law experiments.

Architecture: SigLIP (frozen) + Perceiver Resampler (trainable) + Qwen2.5 (frozen)

The model:
1. Encodes image through frozen SigLIP → (B, T₀, d_vision)
2. Resamples through adapter → (B, T, d_llm)
3. Concatenates [visual_tokens, text_tokens] and feeds to frozen LLM
4. Computes NLL loss on text tokens only
"""

import os

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipVisionModel,
    SiglipImageProcessor,
)

from .perceiver_resampler import PerceiverResampler, build_adapter
from .adapters import build_adapter_by_type, ADAPTER_TYPES


# Qwen2.5 model name → hidden dimension mapping
QWEN25_CONFIGS = {
    "Qwen/Qwen2.5-0.5B": {"d_llm": 896},
    "Qwen/Qwen2.5-1.5B": {"d_llm": 1536},
    "Qwen/Qwen2.5-3B":   {"d_llm": 2048},
    "Qwen/Qwen2.5-7B":   {"d_llm": 3584},
    "Qwen/Qwen2.5-14B":  {"d_llm": 5120},
    "Qwen/Qwen2.5-32B":  {"d_llm": 5120},  # same hidden dim as 14B, more layers
}

SIGLIP_CONFIG = {
    "google/siglip-so400m-patch14-384": {"d_vision": 1152, "image_size": 384},
    "google/siglip-so400m-patch14-224": {"d_vision": 1152, "image_size": 224},
}


class VLMForScaling(nn.Module):
    """
    VLM model: frozen VE + trainable adapter + frozen LLM.
    Only the adapter parameters are trained.
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2.5-3B",
        vision_name: str = "google/siglip-so400m-patch14-384",
        adapter_level: str = "M",
        num_queries: int = 64,
        adapter_num_layers: int = 2,
        adapter_type: str = "perceiver",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.llm_name = llm_name
        self.vision_name = vision_name
        self.adapter_type = adapter_type
        self.torch_dtype = torch_dtype

        # Get dimensions
        d_llm = QWEN25_CONFIGS[llm_name]["d_llm"]
        d_vision = SIGLIP_CONFIG[vision_name]["d_vision"]

        # Build adapter (the only trainable component)
        self.adapter = build_adapter_by_type(
            adapter_type=adapter_type,
            level=adapter_level,
            num_queries=num_queries,
            d_vision=d_vision,
            d_llm=d_llm,
            num_layers=adapter_num_layers,
        )

        # Placeholders — loaded lazily to save memory during config
        self.vision_encoder = None
        self.llm = None
        self.tokenizer = None
        self.image_processor = None

    @staticmethod
    def _estimate_model_gb(model_name: str) -> float:
        """Estimate bf16 weight size in GB."""
        size_map = {
            "0.5B": 1.0, "1.5B": 3.0, "3B": 6.0,
            "7B": 14.0, "14B": 28.0, "32B": 64.0,
        }
        for key, gb in size_map.items():
            if key in model_name:
                return gb
        return 10.0  # fallback

    @staticmethod
    def _get_vram_mb() -> list[int]:
        """Get VRAM in MB for each visible GPU."""
        vrams = []
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            vrams.append(total // (1024 * 1024))
        return vrams

    def load_backbones(self, device: torch.device | str = "cuda"):
        """Load frozen backbones. Automatically uses device_map for large models."""
        num_gpus = torch.cuda.device_count()
        vrams = self._get_vram_mb() if num_gpus > 0 else []
        min_vram_gb = min(vrams) / 1024 if vrams else 0
        llm_size_gb = self._estimate_model_gb(self.llm_name)

        # Overhead: VE (~1GB) + adapter (<1GB) + activations (~3-5GB)
        overhead_gb = 6.0
        need_multi_gpu = (llm_size_gb + overhead_gb) > min_vram_gb and num_gpus > 1

        # Gradient checkpointing to reduce activation memory for large-T experiments
        use_grad_ckpt = os.environ.get("GRAD_CHECKPOINT", "0") == "1"

        print(f"  GPU count: {num_gpus}, VRAM/GPU: {min_vram_gb:.0f}GB, "
              f"LLM: ~{llm_size_gb:.0f}GB, multi-GPU: {need_multi_gpu}"
              f"{', grad_ckpt: True' if use_grad_ckpt else ''}")

        # Vision encoder — always on first GPU
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            self.vision_name,
            torch_dtype=self.torch_dtype,
        ).to(device)
        self.vision_encoder.eval()
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # LLM — device_map="auto" if too large for single GPU
        if need_multi_gpu:
            print(f"  Loading LLM with device_map='auto' across {num_gpus} GPUs")
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                torch_dtype=self.torch_dtype,
                attn_implementation="sdpa",
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                torch_dtype=self.torch_dtype,
                attn_implementation="sdpa",
            ).to(device)

        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        # Enable gradient checkpointing to save activation memory
        if use_grad_ckpt:
            self.llm.gradient_checkpointing_enable()
            print("  LLM gradient checkpointing enabled")

        self._multi_gpu = need_multi_gpu

        # Tokenizer and image processor
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_name)

    @property
    def device(self) -> torch.device:
        return next(self.adapter.parameters()).device

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through frozen SigLIP → (B, T₀, d_vision)."""
        with torch.no_grad():
            outputs = self.vision_encoder(
                pixel_values=pixel_values.to(self.torch_dtype),
            )
            # Use last hidden state (all patch tokens, no CLS)
            vision_features = outputs.last_hidden_state
        return vision_features

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Forward pass. Returns dict with 'loss' and 'logits'.

        1. Encode image → vision features
        2. Resample → visual tokens
        3. Get LLM text embeddings
        4. Concatenate [visual_tokens, text_embeddings]
        5. Forward through LLM
        6. Compute NLL on text tokens only
        """
        B = pixel_values.shape[0]
        device = self.device

        # 1. Vision encoding (frozen, no grad)
        vision_features = self.encode_vision(pixel_values.to(device))

        # 2. Adapter (trainable)
        visual_tokens = self.adapter(vision_features.to(self.torch_dtype))
        # visual_tokens: (B, T, d_llm)

        T = visual_tokens.shape[1]

        # 3. Get text embeddings from LLM
        # When using device_map, LLM embedding layer may be on a different GPU
        embed_device = next(self.llm.get_input_embeddings().parameters()).device
        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            text_embeddings = self.llm.get_input_embeddings()(input_ids)
            # text_embeddings: (B, S_text, d_llm)

        # 4. Concatenate: [visual_tokens, text_embeddings]
        # Move visual_tokens to same device as text_embeddings
        combined_embeddings = torch.cat(
            [visual_tokens.to(device=text_embeddings.device, dtype=text_embeddings.dtype),
             text_embeddings], dim=1
        )

        # Build attention mask and labels on embed device
        emb_device = combined_embeddings.device
        attention_mask = attention_mask.to(emb_device)
        visual_mask = torch.ones(B, T, dtype=attention_mask.dtype, device=emb_device)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        input_ids = input_ids.to(emb_device)
        visual_labels = torch.full((B, T), -100, dtype=input_ids.dtype, device=emb_device)
        text_labels = input_ids.clone()
        text_labels[attention_mask == 0] = -100
        combined_labels = torch.cat([visual_labels, text_labels], dim=1)

        # 5. Forward through LLM (frozen, but grad flows through embeddings from adapter)
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            labels=combined_labels,
            use_cache=False,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def apply_lora(self, r: int = 64, alpha: int = 128, target_modules: list[str] = None):
        """Apply LoRA to LLM for G6 experiments (unfrozen LLM)."""
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
        self._has_lora = True

    @property
    def has_lora(self) -> bool:
        return getattr(self, "_has_lora", False)

    def trainable_parameters(self):
        """Return all trainable parameters (adapter + LoRA if applicable)."""
        params = list(self.adapter.parameters())
        if self.has_lora and self.llm is not None:
            params += [p for p in self.llm.parameters() if p.requires_grad]
        return params

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate text conditioned on image (+ optional text prompt).

        Args:
            pixel_values: (B, C, H, W) preprocessed images
            prompt_ids: (B, S) tokenized prompt (e.g., "Question: ... Short answer:")
            prompt_mask: (B, S) attention mask for prompt
            max_new_tokens: max tokens to generate
            **generate_kwargs: passed to LLM.generate()

        Returns:
            generated_ids: (B, max_new_tokens) generated token ids
        """
        device = self.device

        # 1. Vision encoding → adapter
        vision_features = self.encode_vision(pixel_values.to(device))
        visual_tokens = self.adapter(vision_features.to(self.torch_dtype))
        B, T, _ = visual_tokens.shape

        # 2. Build inputs_embeds
        embed_device = next(self.llm.get_input_embeddings().parameters()).device
        visual_embeds = visual_tokens.to(device=embed_device, dtype=self.torch_dtype)

        if prompt_ids is not None:
            prompt_ids = prompt_ids.to(embed_device)
            text_embeds = self.llm.get_input_embeddings()(prompt_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            S_text = prompt_ids.shape[1]
            # attention mask
            vis_mask = torch.ones(B, T, dtype=torch.long, device=embed_device)
            if prompt_mask is not None:
                prompt_mask = prompt_mask.to(embed_device)
            else:
                prompt_mask = torch.ones(B, S_text, dtype=torch.long, device=embed_device)
            attention_mask = torch.cat([vis_mask, prompt_mask], dim=1)
        else:
            inputs_embeds = visual_embeds
            attention_mask = torch.ones(B, T, dtype=torch.long, device=embed_device)

        # 3. Generate
        pad_token_id = generate_kwargs.pop("pad_token_id", None)
        if pad_token_id is None and self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            **generate_kwargs,
        )
        return outputs

    def get_vision_metadata(self) -> dict:
        """Compute vision-related metadata for logging (T₀, ρ, etc.)."""
        siglip_cfg = SIGLIP_CONFIG[self.vision_name]
        image_size = siglip_cfg["image_size"]
        patch_size = 14  # SigLIP uses patch_size=14
        T0 = (image_size // patch_size) ** 2
        T = self.adapter.num_queries
        rho = T / T0
        return {
            "image_size": image_size,
            "patch_size": patch_size,
            "T0": T0,
            "T": T,
            "rho": round(rho, 6),
            "N_A": self.adapter.num_params,
            "d_model": self.adapter.d_model,
            "adapter_num_layers": self.adapter.num_layers,
        }

    def get_param_summary(self) -> dict:
        """Return parameter count summary."""
        adapter_params = self.adapter.num_params
        adapter_trainable = self.adapter.num_trainable_params

        ve_params = sum(p.numel() for p in self.vision_encoder.parameters()) if self.vision_encoder else 0
        llm_params = sum(p.numel() for p in self.llm.parameters()) if self.llm else 0
        lora_trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad) if self.has_lora else 0

        total_trainable = adapter_trainable + lora_trainable
        total_params = adapter_params + ve_params + llm_params

        return {
            "adapter_total": adapter_params,
            "adapter_trainable": adapter_trainable,
            "lora_trainable": lora_trainable,
            "vision_encoder": ve_params,
            "llm": llm_params,
            "total": total_params,
            "total_trainable": total_trainable,
            "trainable_ratio": total_trainable / total_params if total_params > 0 else 0,
        }
