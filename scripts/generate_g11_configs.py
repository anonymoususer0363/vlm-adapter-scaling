"""
Generate G11 configs: adapter type comparison.

Tests: MLP Projector + Q-Former vs Perceiver Resampler baseline.
Fixed: N_L=3B, N_A=M, D=552K (1 epoch)
Variable: T ∈ {16, 32, 64, 128} × adapter_type ∈ {mlp, qformer}

Perceiver baselines already exist in G2 (3B, T=16/32/64/128, M).

Usage:
    python scripts/generate_g11_configs.py
"""

import os
import yaml
from pathlib import Path


def generate_g11_configs():
    output_dir = Path("configs/g11")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = os.environ.get("VLM_DATA_DIR", "data")
    checkpoint_dir = os.environ.get("VLM_CHECKPOINT_DIR", "checkpoints")

    T_values = [16, 32, 64, 128]
    adapter_types = ["mlp", "qformer"]

    configs = []
    for adapter_type in adapter_types:
        for T in T_values:
            run_name = f"g11_3B_T{T}_M_{adapter_type}_d552k_s42"
            config = {
                "llm_name": "Qwen/Qwen2.5-3B",
                "vision_name": "google/siglip-so400m-patch14-384",
                "adapter_level": "M",
                "num_queries": T,
                "adapter_num_layers": 2,
                "adapter_type": adapter_type,
                "train_data": f"{data_dir}/processed/train.jsonl",
                "val_data": f"{data_dir}/processed/val.jsonl",
                "image_root": f"{data_dir}/llava_pretrain",
                "num_epochs": 1,
                "batch_size": 32,
                "lr": 1e-4,
                "seed": 42,
                "run_name": run_name,
                "output_dir": checkpoint_dir,
                "use_wandb": False,
            }

            config_path = output_dir / f"{run_name}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            configs.append(config_path)

    print(f"Generated {len(configs)} G11 configs in {output_dir}/")
    print(f"\nAdapter types: {adapter_types}")
    print(f"T values: {T_values}")
    print(f"Total: {len(configs)} runs (~12h each @ 3B)")

    print(f"\nBaselines (already in G2):")
    for T in T_values:
        print(f"  g2_3B_T{T}_M_d552k_s42 (perceiver)")

    print(f"\nNew experiments:")
    for c in configs:
        print(f"  {c.stem}")


if __name__ == "__main__":
    generate_g11_configs()
