#!/usr/bin/env python3
"""Generate G9v2 (seed runs at LR=3e-4) + G31 (d2m expansion) configs."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT / "configs"

COMMON = dict(
    train_data="data/processed/train.jsonl",
    val_data="data/processed/val.jsonl",
    image_root="data/llava_pretrain",
    output_dir="checkpoints",
    lr=0.0003,
)

LLM_SETTINGS = {
    "3B": dict(llm_name="Qwen/Qwen2.5-3B", batch_size=32, grad_accum_steps=1),
    "7B": dict(llm_name="Qwen/Qwen2.5-7B", batch_size=16, grad_accum_steps=2),
}


def write_yaml(path, fields):
    with open(path, "w") as f:
        for k, v in fields.items():
            f.write(f"{k}: {v}\n")
    print(f"  {path.relative_to(PROJECT)}")


def generate_g9v2():
    """Seed runs: {3B,7B} × {T32,T64} × M × seeds {123,456}"""
    out = CONFIGS_DIR / "g9v2"
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for llm_size, llm_cfg in LLM_SETTINGS.items():
        for T in [32, 64]:
            for seed in [123, 456]:
                name = f"g9v2_{llm_size}_T{T}_M_d552k_s{seed}_lr3e4"
                cfg = {
                    **COMMON,
                    **llm_cfg,
                    "adapter_level": "M",
                    "num_queries": T,
                    "seed": seed,
                    "run_name": name,
                    "num_epochs": 1,
                    "eval_interval_steps": 500,
                    "save_interval_steps": 2000,
                }
                write_yaml(out / f"{name}.yaml", cfg)
                count += 1
    print(f"G9v2: {count} configs")
    return count


def generate_g31():
    """D=2M expansion: {3B,7B} × T={8,16,32,128} × M × d2m"""
    out = CONFIGS_DIR / "g31"
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for llm_size, llm_cfg in LLM_SETTINGS.items():
        for T in [8, 16, 32, 128]:
            name = f"g31_{llm_size}_T{T}_M_d2m_s42_lr3e4"
            cfg = {
                **COMMON,
                **llm_cfg,
                "adapter_level": "M",
                "num_queries": T,
                "seed": 42,
                "run_name": name,
                "num_epochs": 4,
                "eval_interval_steps": 2000,
                "save_interval_steps": 10000,
            }
            write_yaml(out / f"{name}.yaml", cfg)
            count += 1
    print(f"G31: {count} configs")
    return count


if __name__ == "__main__":
    print("Generating G9v2 + G31 configs...")
    n1 = generate_g9v2()
    n2 = generate_g31()
    print(f"\nTotal: {n1 + n2} configs")
