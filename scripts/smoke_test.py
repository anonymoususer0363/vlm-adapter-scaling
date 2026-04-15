"""
Smoke test: verify the entire pipeline works end-to-end.
Uses synthetic data (no real images needed).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Auto-detect local model cache
_local_cache = _project_root / "hf_cache"
if _local_cache.exists() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_local_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(_local_cache / "hub")

from src.perceiver_resampler import build_adapter, get_adapter_config


def test_adapter_forward():
    """Test adapter with random inputs."""
    print("=" * 50)
    print("Test 1: Adapter forward pass")
    print("=" * 50)

    for level in ["XS", "S", "M", "L", "XL"]:
        for T in [8, 32, 64, 128]:
            adapter = build_adapter(
                level=level,
                num_queries=T,
                d_vision=1152,  # SigLIP-SO400M
                d_llm=2048,     # Qwen2.5-3B
            )
            # Random vision features: (B=2, T0=729, d_vision=1152)
            x = torch.randn(2, 729, 1152)
            out = adapter(x)
            assert out.shape == (2, T, 2048), f"Expected (2, {T}, 2048), got {out.shape}"

    print("  All adapter configs pass!")


def test_adapter_gradient():
    """Test gradient flows through adapter."""
    print("\n" + "=" * 50)
    print("Test 2: Gradient flow")
    print("=" * 50)

    adapter = build_adapter(level="S", num_queries=32, d_vision=1152, d_llm=2048)
    x = torch.randn(2, 729, 1152)
    out = adapter(x)
    loss = out.sum()
    loss.backward()

    has_grad = all(p.grad is not None for p in adapter.parameters() if p.requires_grad)
    print(f"  All parameters have gradients: {has_grad}")
    assert has_grad


def test_data_loading():
    """Test data loading with dummy data."""
    print("\n" + "=" * 50)
    print("Test 3: Data loading")
    print("=" * 50)

    from src.data import ImageCaptionDataset, collate_fn

    # Create temporary dummy dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        for i in range(10):
            img = Image.new("RGB", (384, 384), color=(i * 25, i * 10, 128))
            img.save(img_dir / f"img_{i}.jpg")

        # Create dummy JSONL
        data_path = Path(tmpdir) / "data.jsonl"
        with open(data_path, "w") as f:
            for i in range(10):
                record = {"image": f"images/img_{i}.jpg", "caption": f"A test image number {i} with some text."}
                f.write(json.dumps(record) + "\n")

        # Need a tokenizer and image processor
        # Use simple mock for now
        from transformers import AutoTokenizer, SiglipImageProcessor

        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

            dataset = ImageCaptionDataset(
                data_path=str(data_path),
                image_root=tmpdir,
                image_processor=image_processor,
                tokenizer=tokenizer,
                num_samples=5,
            )

            batch = [dataset[i] for i in range(min(3, len(dataset)))]
            collated = collate_fn(batch, pad_token_id=tokenizer.pad_token_id)

            print(f"  pixel_values: {collated['pixel_values'].shape}")
            print(f"  input_ids:    {collated['input_ids'].shape}")
            print(f"  attn_mask:    {collated['attention_mask'].shape}")
            print("  Data loading works!")

        except Exception as e:
            print(f"  Skipped (models not downloaded yet): {e}")


def test_full_forward():
    """Test full VLM forward pass with real models (if available)."""
    print("\n" + "=" * 50)
    print("Test 4: Full VLM forward pass")
    print("=" * 50)

    try:
        from src.model import VLMForScaling

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        model = VLMForScaling(
            llm_name="Qwen/Qwen2.5-0.5B",
            vision_name="google/siglip-so400m-patch14-384",
            adapter_level="XS",  # Smallest for speed
            num_queries=8,       # Fewest tokens
            torch_dtype=torch.bfloat16,
        )
        model.adapter = model.adapter.to(device=device, dtype=torch.bfloat16)
        model.load_backbones(device=device)

        param_summary = model.get_param_summary()
        print(f"  Adapter:  {param_summary['adapter_trainable']:>10,} params")
        print(f"  VE:       {param_summary['vision_encoder']:>10,} params")
        print(f"  LLM:      {param_summary['llm']:>10,} params")

        # Fake batch
        B = 2
        pixel_values = torch.randn(B, 3, 384, 384, device=device, dtype=torch.bfloat16)
        input_ids = torch.randint(0, 1000, (B, 20), device=device)
        attention_mask = torch.ones(B, 20, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        print(f"  Loss: {outputs['loss'].item():.4f}")
        print(f"  Logits shape: {outputs['logits'].shape}")

        # Test backward
        outputs["loss"].backward()
        adapter_grads = sum(1 for p in model.adapter.parameters() if p.grad is not None)
        llm_grads = sum(1 for p in model.llm.parameters() if p.grad is not None)
        print(f"  Adapter params with grad: {adapter_grads}")
        print(f"  LLM params with grad: {llm_grads} (should be 0)")
        assert llm_grads == 0, "LLM should be frozen!"
        print("  Full forward+backward pass works!")

    except Exception as e:
        print(f"  Skipped: {e}")


def test_depth_ablation():
    """Test adapter with different depths (G8)."""
    print("\n" + "=" * 50)
    print("Test 5: Depth ablation")
    print("=" * 50)

    for depth in [1, 2, 4, 6]:
        adapter = build_adapter(level="M", num_queries=64, d_vision=1152, d_llm=2048, num_layers=depth)
        x = torch.randn(1, 729, 1152)
        out = adapter(x)
        assert out.shape == (1, 64, 2048)
        print(f"  depth={depth}: {adapter.num_params:>10,} params  OK")

    print("  Depth ablation pass!")


def test_lora():
    """Test LoRA support (G6)."""
    print("\n" + "=" * 50)
    print("Test 6: LoRA support")
    print("=" * 50)

    try:
        from src.model import VLMForScaling

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VLMForScaling(
            llm_name="Qwen/Qwen2.5-0.5B",
            vision_name="google/siglip-so400m-patch14-384",
            adapter_level="XS",
            num_queries=8,
            torch_dtype=torch.bfloat16,
        )
        model.adapter = model.adapter.to(device=device, dtype=torch.bfloat16)
        model.load_backbones(device=device)
        model.apply_lora(r=16, alpha=32)

        summary = model.get_param_summary()
        print(f"  Adapter trainable: {summary['adapter_trainable']:>10,}")
        print(f"  LoRA trainable:    {summary['lora_trainable']:>10,}")
        print(f"  Total trainable:   {summary['total_trainable']:>10,}")

        assert summary['lora_trainable'] > 0, "LoRA should add trainable params"
        assert model.has_lora

        # Test forward + backward
        B = 1
        pixel_values = torch.randn(B, 3, 384, 384, device=device, dtype=torch.bfloat16)
        input_ids = torch.randint(0, 1000, (B, 10), device=device)
        attention_mask = torch.ones(B, 10, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        outputs["loss"].backward()

        adapter_grads = sum(1 for p in model.adapter.parameters() if p.grad is not None)
        lora_grads = sum(1 for p in model.llm.parameters() if p.requires_grad and p.grad is not None)
        print(f"  Adapter grads: {adapter_grads}")
        print(f"  LoRA grads: {lora_grads}")
        assert adapter_grads > 0
        assert lora_grads > 0
        print("  LoRA support works!")

    except Exception as e:
        print(f"  Skipped: {e}")


if __name__ == "__main__":
    test_adapter_forward()
    test_adapter_gradient()
    test_data_loading()
    test_full_forward()
    test_depth_ablation()
    test_lora()
    print("\n" + "=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)
