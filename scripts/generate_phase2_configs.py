"""
Generate experiment configs for Phase 2 (G12-G16).

Usage:
    python scripts/generate_phase2_configs.py
    python scripts/generate_phase2_configs.py --groups g16        # G16 only
    python scripts/generate_phase2_configs.py --groups g12 g13    # specific groups
    python scripts/generate_phase2_configs.py --dry-run           # print without writing

Groups:
  G12: L(N_L, D) extended D range — 5 LLMs × 6 D = 30 runs (15 new, 15 overlap G0v2)
  G13: L(T) at multiple D — 3B × 7 T × 3 D = 21 runs
  G14: L(N_A) at multiple D — 3B × 5 N_A × 3 D = 15 runs
  G15: T × N_A at large D — 3B × 4T × 4N_A × 2D = 32 runs (16 overlap G3)
  G16: LR sensitivity for 0.5B — 0.5B × 4 LR × 3 T × 2 seeds = 24 runs
"""

import argparse
import os
from itertools import product
from pathlib import Path

import yaml


# ─────────────────── Grid definitions ───────────────────
LLM_MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B",
    "1.5B": "Qwen/Qwen2.5-1.5B",
    "3B":   "Qwen/Qwen2.5-3B",
    "7B":   "Qwen/Qwen2.5-7B",
    "14B":  "Qwen/Qwen2.5-14B",
    "32B":  "Qwen/Qwen2.5-32B",
}

FITTING_LLMS = ["0.5B", "1.5B", "3B", "7B", "14B"]
ADAPTER_LEVELS = ["XS", "S", "M", "L", "XL"]
T_VALUES = [4, 8, 16, 32, 64, 128, 256]

# Data paths
LLAVA_TRAIN = "data/processed/train.jsonl"
LLAVA_VAL = "data/processed/val.jsonl"
LLAVA_IMAGE_ROOT = "data/llava_pretrain"
COMBINED_TRAIN = "data/processed/train_combined.jsonl"  # LLaVA + ShareGPT4V
COMBINED_IMAGE_ROOT = "data"  # parent dir (images in llava_pretrain/ + sharegpt4v/)
# CC3M full combined dataset (LLaVA + CC3M, ~3M+ unique pairs)
FULL_TRAIN = "data/processed/train_full.jsonl"
FULL_IMAGE_ROOT = "data"  # parent dir (images in llava_pretrain/ + cc3m/)


def _d_short(d: int) -> str:
    """Format D value for run name."""
    if d >= 1_000_000:
        m = d / 1_000_000
        if m == int(m):
            return f"d{int(m)}m"
        return f"d{m:.1f}m"
    return f"d{d // 1000}k"


def _eval_save_intervals(num_samples: int | None, num_epochs: int = 1) -> tuple[int, int]:
    """Compute eval/save intervals based on training length."""
    if num_samples is None:
        total_steps = 552544 * num_epochs // 32
    else:
        total_steps = num_samples // 32

    if total_steps < 2000:
        return 50, 300
    elif total_steps < 10000:
        return 200, 1000
    elif total_steps < 30000:
        return 500, 2000
    else:
        return 1000, 5000


def make_config(
    group: str,
    llm_key: str,
    adapter_level: str,
    num_queries: int,
    d_value: int,
    seed: int = 42,
    lr: float = 1e-4,
    **overrides,
) -> dict:
    """Create a single experiment config.

    Args:
        d_value: Target D (number of seen image-text pairs).
            - D <= 552544: subsample from LLaVA-Pretrain
            - D > 552544: use combined dataset (LLaVA + ShareGPT4V)
    """
    llm_name = LLM_MODELS[llm_key]
    d_tag = _d_short(d_value)

    # Determine data source and config
    if d_value <= 552544:
        train_data = LLAVA_TRAIN
        val_data = LLAVA_VAL
        image_root = LLAVA_IMAGE_ROOT
        if d_value < 552544:
            num_samples = d_value
        else:
            num_samples = None  # full dataset
        num_epochs = 1
    elif d_value <= 671_000:
        # Use combined LLaVA+COCO (671K verified pairs)
        train_data = COMBINED_TRAIN
        val_data = LLAVA_VAL
        image_root = COMBINED_IMAGE_ROOT
        num_samples = d_value
        num_epochs = 1
    else:
        # Need full CC3M combined dataset for D > 671K
        train_data = FULL_TRAIN
        val_data = LLAVA_VAL
        image_root = FULL_IMAGE_ROOT
        num_samples = d_value
        num_epochs = 1

    eval_interval, save_interval = _eval_save_intervals(num_samples, num_epochs)

    # LR tag for non-default LR
    lr_tag = ""
    if lr != 1e-4:
        lr_exp = f"{lr:.0e}".replace("e-0", "e-").replace("+", "")
        lr_tag = f"_lr{lr_exp}"

    run_name = f"{group}_{llm_key}_T{num_queries}_{adapter_level}_{d_tag}{lr_tag}_s{seed}"

    config = {
        "llm_name": llm_name,
        "adapter_level": adapter_level,
        "num_queries": num_queries,
        "seed": seed,
        "run_name": run_name,
        "num_epochs": num_epochs,
        "batch_size": 32,
        "grad_accum_steps": 1,
        "lr": lr,
        "eval_interval_steps": eval_interval,
        "save_interval_steps": save_interval,
        "train_data": train_data,
        "val_data": val_data,
        "image_root": image_root,
        "output_dir": "checkpoints",
    }

    if num_samples is not None:
        config["num_samples"] = num_samples

    # Adjust batch/grad_accum for large models
    if llm_key in ("7B", "14B"):
        config["batch_size"] = 16
        config["grad_accum_steps"] = 2
    elif llm_key == "32B":
        config["batch_size"] = 8
        config["grad_accum_steps"] = 4

    config.update(overrides)
    return config


# ─────────────────── Group generators ───────────────────

def generate_g12() -> list[dict]:
    """G12: Base Scaling L(N_L, D) — Extended D with unique data.
    4 LLMs × 3 D = 12 new configs (excl 0.5B, D ≤ 558K already in G0v2).
    D = {1M, 2M, 3M} using CC3M combined dataset (all unique, no multi-epoch).
    """
    configs = []
    d_values = [1_000_000, 2_000_000, 3_000_000]
    llms = ["1.5B", "3B", "7B", "14B"]  # exclude 0.5B (structurally unstable)

    for llm_key, d in product(llms, d_values):
        cfg = make_config("g12", llm_key, "M", 64, d)
        configs.append(cfg)

    return configs


def generate_g13() -> list[dict]:
    """G13: T Marginal at Multiple D — T sensitivity recovery.
    3B × 7T × 3D = 21 configs.
    D=558K overlaps with G2 (skip).
    D=200K already done (7/7 complete).
    New: D=2M using CC3M (truly unique data, T sensitivity test at large D).
    """
    configs = []
    d_values = [200_000, 552_544, 2_000_000]

    for d, t in product(d_values, T_VALUES):
        # D=558K already done in G2, D=200K already done
        if d <= 552_544:
            continue
        cfg = make_config("g13", "3B", "M", t, d)
        configs.append(cfg)

    return configs


def generate_g14() -> list[dict]:
    """G14: N_A Marginal at Multiple D.
    3B × 5 N_A × 3D = 15 configs.
    D=558K overlaps with G1 (skip).
    New: D={200K, 1.5M}.
    """
    configs = []
    d_values = [200_000, 552_544, 1_500_000]

    for d, level in product(d_values, ADAPTER_LEVELS):
        if d == 552_544:
            continue
        cfg = make_config("g14", "3B", level, 64, d)
        configs.append(cfg)

    return configs


def generate_g15() -> list[dict]:
    """G15: T × N_A Interaction at Large D.
    3B × 4T × 4N_A × 2D = 32 configs.
    D=558K overlaps with G3 (skip).
    New: D=1.5M (16 configs).
    """
    configs = []
    t_values = [16, 32, 64, 128]
    na_values = ["S", "M", "L", "XL"]

    for t, level in product(t_values, na_values):
        # Only D=1.5M — D=558K already in G3
        cfg = make_config("g15", "3B", level, t, 1_500_000)
        configs.append(cfg)

    return configs


def generate_g16() -> list[dict]:
    """G16: LR Sensitivity for 0.5B.
    0.5B × 4 LR × 3 T × 2 seeds = 24 configs.
    Uses existing LLaVA-Pretrain data (D=552K).
    """
    configs = []
    lr_values = [1e-5, 3e-5, 5e-5, 1e-4]
    t_values = [32, 64, 128]
    seeds = [42, 123]

    for lr, t, seed in product(lr_values, t_values, seeds):
        cfg = make_config("g16", "0.5B", "M", t, 552_544, seed=seed, lr=lr)
        configs.append(cfg)

    return configs


# ─────────────────── Main ───────────────────

GENERATORS = {
    "g12": generate_g12,
    "g13": generate_g13,
    "g14": generate_g14,
    "g15": generate_g15,
    "g16": generate_g16,
}


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 2 configs (G12-G16)")
    parser.add_argument("--groups", nargs="*", default=list(GENERATORS.keys()),
                        help="Which groups to generate (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config summary without writing files")
    args = parser.parse_args()

    total = 0
    needs_combined = 0

    for group_name in args.groups:
        if group_name not in GENERATORS:
            print(f"Unknown group: {group_name}")
            continue

        configs = GENERATORS[group_name]()
        config_dir = Path("configs") / group_name
        config_dir.mkdir(parents=True, exist_ok=True)

        for cfg in configs:
            if args.dry_run:
                needs_sharegpt = "combined" in cfg.get("train_data", "")
                marker = " [needs ShareGPT4V]" if needs_sharegpt else ""
                print(f"  {cfg['run_name']}{marker}")
                if needs_sharegpt:
                    needs_combined += 1
            else:
                fname = f"{cfg['run_name']}.yaml"
                with open(config_dir / fname, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        total += len(configs)
        print(f"{group_name}: {len(configs)} configs" + (" (dry-run)" if args.dry_run else " generated"))

    print(f"\nTotal: {total} new configs")
    if needs_combined:
        print(f"  {needs_combined} configs need ShareGPT4V (combined dataset)")
        print(f"  {total - needs_combined} configs can run immediately")

    # Summary of what can run now vs later
    if not args.dry_run:
        print("\n--- Immediately runnable ---")
        for group_name in args.groups:
            config_dir = Path("configs") / group_name
            for f in sorted(config_dir.glob("*.yaml")):
                with open(f) as fh:
                    cfg = yaml.safe_load(fh)
                if "combined" not in cfg.get("train_data", ""):
                    print(f"  {f.name}")


if __name__ == "__main__":
    main()
