"""
Generate configs for D-fix experiments (G0v2, G5v2) + outlier reruns.

D control strategy (NO data.py changes needed):
  - D < 552K: subsample via num_samples (existing data.py feature)
  - D = 552K: full dataset, num_epochs=1
  - D > 552K: multi-epoch via num_epochs (trainer handles this)

G0v2: D scaling — 5 LLMs × 5 D = 25 experiments
G5v2: D independence — 20 experiments (15 unique after G0v2 overlap)
Reruns: G2 14B T=16 outlier verification
"""

import os
from pathlib import Path

import yaml


# ─────────────────── Constants ───────────────────

LLM_MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B",
    "1.5B": "Qwen/Qwen2.5-1.5B",
    "3B":   "Qwen/Qwen2.5-3B",
    "7B":   "Qwen/Qwen2.5-7B",
    "14B":  "Qwen/Qwen2.5-14B",
}

DATASET_SIZE = 552_544  # train.jsonl line count

# D targets → (num_samples, num_epochs, approx_seen_pairs)
# D < DATASET_SIZE: subsample + 1 epoch
# D >= DATASET_SIZE: full dataset + N epochs
D_CONFIGS = {
    "d50k":  {"num_samples": 50_000,  "num_epochs": 1,  "seen": 50_000},
    "d200k": {"num_samples": 200_000, "num_epochs": 1,  "seen": 200_000},
    "d552k": {"num_samples": None,    "num_epochs": 1,  "seen": 552_544},
    "d2m":   {"num_samples": None,    "num_epochs": 4,  "seen": 2_210_176},
    "d5m":   {"num_samples": None,    "num_epochs": 10, "seen": 5_525_440},
}

# steps_per_epoch = dataset_size / (batch_size * grad_accum)
# ~30 evals and ~6 saves per experiment
INTERVAL_MAP = {
    "d50k":  {"eval": 50,   "save": 300},
    "d200k": {"eval": 200,  "save": 1000},
    "d552k": {"eval": 500,  "save": 2000},
    "d2m":   {"eval": 2000, "save": 10000},
    "d5m":   {"eval": 5000, "save": 25000},
}


def make_config(
    group: str,
    llm_key: str,
    adapter_level: str,
    num_queries: int,
    d_key: str,
    seed: int = 42,
    **overrides,
) -> dict:
    llm_name = LLM_MODELS[llm_key]
    d_cfg = D_CONFIGS[d_key]
    intervals = INTERVAL_MAP[d_key]
    run_name = f"{group}_{llm_key}_T{num_queries}_{adapter_level}_{d_key}_s{seed}"

    config = {
        "llm_name": llm_name,
        "adapter_level": adapter_level,
        "num_queries": num_queries,
        "seed": seed,
        "run_name": run_name,
        "num_epochs": d_cfg["num_epochs"],
        "batch_size": 32,
        "grad_accum_steps": 1,
        "lr": 1e-4,
        "eval_interval_steps": intervals["eval"],
        "save_interval_steps": intervals["save"],
        "train_data": "data/processed/train.jsonl",
        "val_data": "data/processed/val.jsonl",
        "image_root": "data/llava_pretrain",
        "output_dir": "checkpoints",
    }

    # Only set num_samples for subsampling (D < dataset size)
    if d_cfg["num_samples"] is not None:
        config["num_samples"] = d_cfg["num_samples"]

    # Adjust batch/grad_accum for large models
    if llm_key in ("7B", "14B"):
        config["batch_size"] = 16
        config["grad_accum_steps"] = 2
    elif llm_key == "32B":
        config["batch_size"] = 8
        config["grad_accum_steps"] = 4

    config.update(overrides)
    return config


# ─────────────────── Generators ───────────────────

def generate_g0v2() -> list[dict]:
    """G0v2: D scaling. 5 LLMs × 5 D = 25."""
    configs = []
    for llm_key in ["0.5B", "1.5B", "3B", "7B", "14B"]:
        for d_key in D_CONFIGS:
            cfg = make_config("g0v2", llm_key, "M", 64, d_key)
            configs.append(cfg)
    return configs


def generate_g5v2() -> list[dict]:
    """G5v2: D independence. 20 total."""
    configs = []
    d_t_sweep = ["d50k", "d552k", "d5m"]
    d_na_sweep = ["d50k", "d5m"]

    # (a) T sweep: 3D × 4T @ 3B, M
    for d_key in d_t_sweep:
        for t in [16, 32, 64, 128]:
            cfg = make_config("g5v2", "3B", "M", t, d_key)
            configs.append(cfg)

    # (b) N_A sweep: 2D × 4N_A @ 3B, T=64
    for d_key in d_na_sweep:
        for level in ["S", "M", "L", "XL"]:
            cfg = make_config("g5v2", "3B", level, 64, d_key)
            configs.append(cfg)

    return configs


def generate_reruns() -> list[dict]:
    """Rerun outliers."""
    configs = []
    # G2 14B T=16 outlier: different seed
    cfg = make_config("rerun", "14B", "M", 16, "d552k", seed=123)
    cfg["run_name"] = "rerun_g2_14B_T16_M_d552k_s123"
    configs.append(cfg)
    return configs


# ─────────────────── Main ───────────────────

def main():
    groups = {
        "g0v2": generate_g0v2,
        "g5v2": generate_g5v2,
        "rerun": generate_reruns,
    }

    total = 0
    for group_name, gen_fn in groups.items():
        configs = gen_fn()
        config_dir = Path("configs") / group_name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Clean old configs
        for old in config_dir.glob("*.yaml"):
            old.unlink()

        for cfg in configs:
            fname = f"{cfg['run_name']}.yaml"
            with open(config_dir / fname, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        total += len(configs)
        print(f"{group_name}: {len(configs)} configs")

    print(f"\nTotal: {total} configs")

    # Print time estimates
    print("\n=== Training Time Estimates ===")
    speed = {"0.5B": 3, "1.5B": 2.5, "3B": 2, "7B": 1, "14B": 0.5}
    base_steps = DATASET_SIZE // 32  # 17267 steps per epoch (bs=32)
    for d_key, d_cfg in D_CONFIGS.items():
        if d_cfg["num_samples"]:
            total_steps = d_cfg["num_samples"] // 32
        else:
            total_steps = base_steps * d_cfg["num_epochs"]
        times = [f"{llm}:{total_steps/spd/3600:.1f}h" for llm, spd in speed.items()]
        print(f"  {d_key}: {total_steps:>7,} steps ({d_cfg['seen']:>9,} pairs) | {' | '.join(times)}")


if __name__ == "__main__":
    main()
