#!/usr/bin/env python3
"""Generate G34 configs: Q-Former proper-LR capacity sweep.

Purpose: Verify that LR confound and hook shape generalize to Q-Former architecture.
- 3B × {XS,S,M,L,XL} × {1e-4, 3e-4} = 10 runs
- 14B × {XS,S,M,L,XL} × {3e-4} = 5 runs
- 14B × {M} × {1e-4} = 1 run (baseline)
Total: 16 runs, all T=64, D=552K, adapter_type=qformer
"""

import os
import yaml

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "g34")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LLM_MAP = {
    "3B": "Qwen/Qwen2.5-3B",
    "14B": "Qwen/Qwen2.5-14B-AWQ" if False else "Qwen/Qwen2.5-14B",
}

LEVELS = ["XS", "S", "M", "L", "XL"]

# Design matrix
configs = []

# 3B: full capacity sweep at both LRs
for level in LEVELS:
    for lr in [1e-4, 3e-4]:
        lr_str = "lr1e4" if lr == 1e-4 else "lr3e4"
        configs.append({
            "llm": "3B",
            "level": level,
            "lr": lr,
            "run_name": f"g34_3B_T64_{level}_qformer_d552k_s42_{lr_str}",
        })

# 14B: full capacity sweep at LR=3e-4
for level in LEVELS:
    configs.append({
        "llm": "14B",
        "level": level,
        "lr": 3e-4,
        "run_name": f"g34_14B_T64_{level}_qformer_d552k_s42_lr3e4",
    })

# 14B: M at LR=1e-4 (baseline)
configs.append({
    "llm": "14B",
    "level": "M",
    "lr": 1e-4,
    "run_name": "g34_14B_T64_M_qformer_d552k_s42_lr1e4",
})

for cfg in configs:
    data = {
        "llm_name": LLM_MAP[cfg["llm"]],
        "vision_name": "google/siglip-so400m-patch14-384",
        "adapter_level": cfg["level"],
        "num_queries": 64,
        "adapter_num_layers": 2,
        "adapter_type": "qformer",
        "train_data": "data/processed/train.jsonl",
        "val_data": "data/processed/val.jsonl",
        "image_root": "data/llava_pretrain",
        "num_epochs": 1,
        "batch_size": 32 if cfg["llm"] == "3B" else 16,
        "lr": cfg["lr"],
        "seed": 42,
        "run_name": cfg["run_name"],
        "output_dir": "checkpoints",
        "use_wandb": False,
    }

    path = os.path.join(OUTPUT_DIR, f"{cfg['run_name']}.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print(f"Generated {len(configs)} configs in {OUTPUT_DIR}")
for cfg in configs:
    print(f"  {cfg['run_name']}")
