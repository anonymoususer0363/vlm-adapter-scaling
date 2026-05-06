#!/usr/bin/env python3
"""Generate G37 (LR x N_A) 2D-sweep configs for 3B and 14B at D=552K, T=64.

Existing coverage (reuse, not regenerated):
  3B  : G1 (LR=1e-4 x 5), G28 (LR=3e-4 x 5), G25c (LR=5e-5 x M)
  14B : G1 (LR=1e-4 x 5), G26 (LR=3e-4 x 5), G25b (LR=5e-5 x M)

This script generates only the missing cells (18 yaml).
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "configs" / "g37"
OUT.mkdir(parents=True, exist_ok=True)

WIDTHS = ["XS", "S", "M", "L", "XL"]

# (scale, lr_tag, lr_value, widths_to_run, batch, accum)
SPECS = [
    # 3B: 5e-5 (M already exists in G25c) -> XS,S,L,XL
    ("3B",  "5e5", 0.00005, ["XS", "S", "L", "XL"],     32, 1),
    # 3B: 2e-4 entirely new -> all 5
    ("3B",  "2e4", 0.0002,  ["XS", "S", "M", "L", "XL"], 32, 1),
    # 14B: 5e-5 (M already exists in G25b) -> XS,S,L,XL ; need batch=8 accum=4
    ("14B", "5e5", 0.00005, ["XS", "S", "L", "XL"],      8, 4),
    # 14B: 2e-4 entirely new -> all 5
    ("14B", "2e4", 0.0002,  ["XS", "S", "M", "L", "XL"],  8, 4),
]

LLM_MAP = {"3B": "Qwen/Qwen2.5-3B", "14B": "Qwen/Qwen2.5-14B"}


def write_yaml(path: Path, content: str) -> None:
    path.write_text(content)


count = 0
for scale, lr_tag, lr_val, widths, batch, accum in SPECS:
    for w in widths:
        run_name = f"g37_{scale}_T64_{w}_d552k_s42_lr{lr_tag}"
        yaml = (
            f"llm_name: {LLM_MAP[scale]}\n"
            f"adapter_level: {w}\n"
            f"num_queries: 64\n"
            f"seed: 42\n"
            f"run_name: {run_name}\n"
            f"num_epochs: 1\n"
            f"batch_size: {batch}\n"
            f"grad_accum_steps: {accum}\n"
            f"lr: {lr_val}\n"
            f"eval_interval_steps: 500\n"
            f"save_interval_steps: 2000\n"
            f"train_data: data/processed/train.jsonl\n"
            f"val_data: data/processed/val.jsonl\n"
            f"image_root: data/llava_pretrain\n"
            f"output_dir: checkpoints\n"
        )
        write_yaml(OUT / f"{run_name}.yaml", yaml)
        count += 1

print(f"Wrote {count} configs to {OUT}")
