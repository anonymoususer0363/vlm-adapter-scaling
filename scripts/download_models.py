"""
Pre-download and cache all required models.
Run this before transferring to B200.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipVisionModel,
    SiglipImageProcessor,
)
import torch
import sys


def download_siglip():
    print("=" * 50)
    print("Downloading SigLIP-SO400M...")
    print("=" * 50)
    model_name = "google/siglip-so400m-patch14-384"
    SiglipImageProcessor.from_pretrained(model_name)
    SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print(f"SigLIP cached.")


def download_qwen(size: str):
    model_name = f"Qwen/Qwen2.5-{size}"
    print(f"\nDownloading {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    # Only download config + weights, don't load to GPU
    AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"{model_name} cached.")


def download_siglip_224():
    print("=" * 50)
    print("Downloading SigLIP-SO400M (224px)...")
    print("=" * 50)
    model_name = "google/siglip-so400m-patch14-224"
    SiglipImageProcessor.from_pretrained(model_name)
    SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print(f"SigLIP-224 cached.")


def main():
    sizes = sys.argv[1:] if len(sys.argv) > 1 else ["0.5B"]

    download_siglip()
    download_siglip_224()

    for size in sizes:
        print("=" * 50)
        print(f"Downloading Qwen2.5-{size}...")
        print("=" * 50)
        download_qwen(size)

    print("\nAll models downloaded and cached!")


if __name__ == "__main__":
    main()
