#!/usr/bin/env python3
"""Generate G33 configs: 32B sweep at LR=2e-4 (optimal LR for 32B).
Mirrors G26-G29 structure: N_A sweep + T sweep + D sweep.
G32 T64_M_d552k already done (loss=3.328) → skip.
"""
import os
from pathlib import Path

CONFIGS_DIR = Path("configs/g33")
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

BASE = {
    "llm_name": "Qwen/Qwen2.5-32B",
    "seed": 42,
    "num_epochs": 1,
    "eval_interval_steps": 500,
    "save_interval_steps": 2000,
    "train_data": "data/processed/train.jsonl",
    "val_data": "data/processed/val.jsonl",
    "image_root": "data/llava_pretrain",
    "output_dir": "checkpoints",
    "lr": 0.0002,
    # 32B on 5090: 4 GPUs, batch=8, grad_accum=4 → effective batch=128
    "batch_size": 8,
    "grad_accum_steps": 4,
}

CONFIGS = []

# 1) N_A sweep at T=64, d552k — check hook shape
# M already done (G32: 3.328), add XS, S, L, XL
for level in ["XS", "S", "L", "XL"]:
    name = f"g33_32B_T64_{level}_d552k_s42_lr2e4"
    cfg = {**BASE, "adapter_level": level, "num_queries": 64, "run_name": name}
    CONFIGS.append((name, cfg))

# 2) T sweep at M adapter, d552k — check T marginal
# T=64 already done (G32), add T=32, T=128
for t in [32, 128]:
    name = f"g33_32B_T{t}_M_d552k_s42_lr2e4"
    cfg = {**BASE, "adapter_level": "M", "num_queries": t, "run_name": name}
    CONFIGS.append((name, cfg))

# 3) D sweep at T=64 M — check D power law
# d552k already done (G32), add d50k, d200k
for d_val, d_label in [(50000, "d50k"), (200000, "d200k")]:
    name = f"g33_32B_T64_M_{d_label}_s42_lr2e4"
    cfg = {**BASE, "adapter_level": "M", "num_queries": 64,
           "num_samples": d_val, "run_name": name}
    CONFIGS.append((name, cfg))

# Write configs
for name, cfg in CONFIGS:
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")

print(f"Generated {len(CONFIGS)} configs in {CONFIGS_DIR}/:")
for name, cfg in CONFIGS:
    t = cfg["num_queries"]
    level = cfg["adapter_level"]
    d = cfg.get("num_samples", "552k")
    if isinstance(d, int):
        d = f"{d//1000}k"
    print(f"  {name}  (T={t}, {level}, D={d})")

print(f"\n+ G32 T64_M_d552k (already done: 3.328)")
print(f"= {len(CONFIGS)+1} total 32B configs at LR=2e-4")
print(f"\nEstimate: ~6h each (d552k), d50k/d200k faster")
print(f"Sequential on 4 GPU: ~{6*6 + 1 + 3}h ≈ 40h")
print(f"2-parallel on 8 GPU: ~{(6*6 + 1 + 3)//2}h ≈ 20h")
