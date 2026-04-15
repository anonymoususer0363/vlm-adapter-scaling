"""
Generate experiment configs for all groups (G0-G9).

Usage:
    python scripts/generate_configs.py --data_path /path/to/data.json --image_root /path/to/images

This creates YAML configs in configs/ directory, organized by group.

Groups:
  G0: L(N_L, D) base scaling — 5 LLMs × 4 D = 20 runs
  G1: L(N_A) marginal — 5 N_A × 5 LLMs = 25 runs
  G2: L(T) marginal — 7 T × 5 LLMs = 35 runs
  G3: T × N_A interaction — 4T × 4N_A @ 3B = 16 runs
  G4: ρ = T/T₀ invariance — 2 res × 5 T @ 3B = 10 runs
  G5: D independence — 3D × 4T + 2D × 4N_A @ 3B = 20 runs
  G6: LLM unfrozen (LoRA) — 3 LLMs × 4 T = 12 runs
  G7: Extrapolation (32B + extreme configs) — 10 runs
  G8: Depth ablation — depth {1,4,6} × {3B,7B} + extras = 10 runs
  G9: Seed repetition — 2 extra seeds × 10 key configs = 20 runs
  Total: ~178 runs (~163 unique after dedup)
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

# LLMs used in the *fitting* range (excludes 32B which is extrapolation only)
FITTING_LLMS = ["0.5B", "1.5B", "3B", "7B", "14B"]

ADAPTER_LEVELS = ["XS", "S", "M", "L", "XL"]
T_VALUES = [4, 8, 16, 32, 64, 128, 256]
D_VALUES = [5_000_000, 20_000_000, 50_000_000, 100_000_000]


def make_config(
    group: str,
    llm_key: str,
    adapter_level: str,
    num_queries: int,
    num_samples: int,
    seed: int = 42,
    extra: dict = None,
    **overrides,
) -> dict:
    """Create a single experiment config."""
    llm_name = LLM_MODELS[llm_key]
    d_short = f"d{num_samples // 1_000_000}m"
    run_name = f"{group}_{llm_key}_T{num_queries}_{adapter_level}_{d_short}_s{seed}"

    config = {
        "llm_name": llm_name,
        "adapter_level": adapter_level,
        "num_queries": num_queries,
        "num_samples": num_samples,
        "seed": seed,
        "run_name": run_name,
        "num_epochs": 1,
        "batch_size": 32,
        "grad_accum_steps": 1,
        "lr": 1e-4,
        "eval_interval_steps": 500,
        "save_interval_steps": 2000,
    }

    # Adjust batch/grad_accum for large models
    if llm_key in ("7B", "14B"):
        config["batch_size"] = 16
        config["grad_accum_steps"] = 2
    elif llm_key == "32B":
        config["batch_size"] = 8
        config["grad_accum_steps"] = 4

    if extra:
        config.update(extra)
    config.update(overrides)

    return config


# ─────────────────── Group generators ───────────────────

def generate_g0(base_args: dict) -> list[dict]:
    """G0: L(N_L, D) baseline. 5 LLMs × 4 D = 20 runs.
    Fixed: T=64, N_A=M.
    """
    configs = []
    for llm_key, d in product(FITTING_LLMS, D_VALUES):
        cfg = make_config("g0", llm_key, "M", 64, d, extra=base_args)
        configs.append(cfg)
    return configs


def generate_g1(base_args: dict) -> list[dict]:
    """G1: L(N_A) marginal. 5 adapter levels × 5 LLMs = 25 runs.
    Fixed: T=64, D=50M. (5 overlap with G0)
    """
    configs = []
    for llm_key, level in product(FITTING_LLMS, ADAPTER_LEVELS):
        cfg = make_config("g1", llm_key, level, 64, 50_000_000, extra=base_args)
        configs.append(cfg)
    return configs


def generate_g2(base_args: dict) -> list[dict]:
    """G2: L(T) marginal. 7 T values × 5 LLMs = 35 runs.
    Fixed: N_A=M, D=50M. (5 overlap with G0)
    """
    configs = []
    for llm_key, t in product(FITTING_LLMS, T_VALUES):
        cfg = make_config("g2", llm_key, "M", t, 50_000_000, extra=base_args)
        configs.append(cfg)
    return configs


def generate_g3(base_args: dict) -> list[dict]:
    """G3: T × N_A interaction. 4T × 4N_A @ N_L=3B, D=50M = 16 runs."""
    configs = []
    for t, level in product([8, 32, 64, 128], ["S", "M", "L", "XL"]):
        cfg = make_config("g3", "3B", level, t, 50_000_000, extra=base_args)
        configs.append(cfg)
    return configs


def generate_g4(base_args: dict) -> list[dict]:
    """G4: ρ = T/T₀ invariance. 2 resolutions × 5 T values @ 3B, N_A=M.
    T₀(224) = 256, T₀(384) = 729.
    """
    configs = []
    for res, t in product([224, 384], [8, 16, 32, 64, 128]):
        extra = {**base_args}
        if res == 224:
            extra["vision_name"] = "google/siglip-so400m-patch14-224"
        cfg = make_config("g4", "3B", "M", t, 50_000_000, extra=extra)
        cfg["run_name"] = f"g4_3B_T{t}_M_d50m_res{res}_s42"
        configs.append(cfg)
    return configs


def generate_g5(base_args: dict) -> list[dict]:
    """G5: D independence. Verify T_opt and N_A_opt don't shift with D.
    (a) T sweep at different D: 3D × 4T = 12 runs
    (b) N_A sweep at different D: 2D × 4N_A = 8 runs
    Total: 20 runs @ N_L=3B.
    """
    configs = []
    # (a) T sweep at different D
    for d, t in product([5_000_000, 20_000_000, 100_000_000], [16, 32, 64, 128]):
        cfg = make_config("g5", "3B", "M", t, d, extra=base_args)
        configs.append(cfg)

    # (b) N_A sweep at D={5M, 100M}
    for d, level in product([5_000_000, 100_000_000], ["S", "M", "L", "XL"]):
        cfg = make_config("g5", "3B", level, 64, d, extra=base_args)
        configs.append(cfg)

    return configs


def generate_g6(base_args: dict) -> list[dict]:
    """G6: LLM unfrozen (LoRA). 3 LLMs × 4 T = 12 runs.
    Tests whether golden rules shift when LLM is also trained.
    """
    configs = []
    for llm_key, t in product(["0.5B", "1.5B", "3B"], [16, 32, 64, 128]):
        cfg = make_config("g6", llm_key, "M", t, 50_000_000, extra=base_args)
        cfg["use_lora"] = True
        cfg["lora_r"] = 64
        configs.append(cfg)
    return configs


def generate_g7(base_args: dict) -> list[dict]:
    """G7: Extrapolation validation. Test predictions at unseen scales.
    (a) 32B LLM (beyond fitting range): T={32,64,128} = 3 runs
    (b) 14B + extreme adapters: {XL, L} × T={64,128} = 4 runs
    (c) Extreme D: 3B × M × T=64 × D=200M = 1 run
    (d) Extreme combo: 7B × XL × T=128 = 1 run
    (e) 32B + large adapter: 32B × L × T=64 = 1 run
    Total: 10 runs.
    """
    configs = []

    # (a) 32B across T values
    for t in [32, 64, 128]:
        cfg = make_config("g7", "32B", "M", t, 50_000_000, extra=base_args)
        configs.append(cfg)

    # (b) 14B with large adapters and high T
    for level, t in [("XL", 64), ("XL", 128), ("L", 64), ("L", 128)]:
        cfg = make_config("g7", "14B", level, t, 50_000_000, extra=base_args)
        configs.append(cfg)

    # (c) Extreme D
    cfg = make_config("g7", "3B", "M", 64, 200_000_000, extra=base_args)
    configs.append(cfg)

    # (d) Extreme combo: large LLM + large adapter + many tokens
    cfg = make_config("g7", "7B", "XL", 128, 50_000_000, extra=base_args)
    configs.append(cfg)

    # (e) 32B + large adapter
    cfg = make_config("g7", "32B", "L", 64, 50_000_000, extra=base_args)
    configs.append(cfg)

    return configs


def generate_g8(base_args: dict) -> list[dict]:
    """G8: Adapter depth ablation. Verify depth=2 is a reasonable default.
    (a) Depth {1, 4, 6} × {3B, 7B} × T=64 × M = 6 runs
    (b) Depth {1, 4} × 3B × T=32 × M = 2 runs (check T interaction)
    (c) Depth {4} × 3B × {S, L} = 2 runs (check N_A interaction)
    Total: 10 runs. (depth=2 baselines already in G0/G1/G2)
    """
    configs = []

    # (a) Depth sweep at two LLM scales
    for depth, llm_key in product([1, 4, 6], ["3B", "7B"]):
        cfg = make_config("g8", llm_key, "M", 64, 50_000_000, extra=base_args)
        cfg["adapter_num_layers"] = depth
        cfg["run_name"] = f"g8_{llm_key}_T64_M_d50m_depth{depth}_s42"
        configs.append(cfg)

    # (b) Depth × T interaction
    for depth in [1, 4]:
        cfg = make_config("g8", "3B", "M", 32, 50_000_000, extra=base_args)
        cfg["adapter_num_layers"] = depth
        cfg["run_name"] = f"g8_3B_T32_M_d50m_depth{depth}_s42"
        configs.append(cfg)

    # (c) Depth × N_A interaction
    for level in ["S", "L"]:
        cfg = make_config("g8", "3B", level, 64, 50_000_000, extra=base_args)
        cfg["adapter_num_layers"] = 4
        cfg["run_name"] = f"g8_3B_T64_{level}_d50m_depth4_s42"
        configs.append(cfg)

    return configs


def generate_g9(base_args: dict) -> list[dict]:
    """G9: Seed repetition for confidence intervals.
    Run 2 extra seeds (123, 456) for 10 key configs from G1/G2/G3.
    These are selected at the "interesting" regions:
    - T_opt region across LLM scales
    - N_A transition region
    Total: 20 runs.
    """
    configs = []
    extra_seeds = [123, 456]

    # Key configs for T marginal (from G2): T={32,64} × {0.5B, 3B, 7B, 14B}
    for seed in extra_seeds:
        for llm_key, t in product(["0.5B", "3B", "7B", "14B"], [32, 64]):
            cfg = make_config("g9", llm_key, "M", t, 50_000_000, seed=seed, extra=base_args)
            configs.append(cfg)

    # Key configs for N_A marginal (from G1): N_A={M, L} × {3B}
    for seed in extra_seeds:
        cfg = make_config("g9", "3B", "L", 64, 50_000_000, seed=seed, extra=base_args)
        configs.append(cfg)

    return configs


# ─────────────────── Main ───────────────────

def main():
    # Defaults from environment variables (set by setup_env.sh) or fallback
    data_dir = os.environ.get("VLM_DATA_DIR", "data")
    ckpt_dir = os.environ.get("VLM_CHECKPOINT_DIR", "checkpoints")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default=f"{data_dir}/processed/train.jsonl")
    parser.add_argument("--val_data", type=str, default=f"{data_dir}/processed/val.jsonl")
    parser.add_argument("--image_root", type=str, default=f"{data_dir}/llava_pretrain")
    parser.add_argument("--output_dir", type=str, default=ckpt_dir)
    args = parser.parse_args()

    base_args = {
        "train_data": args.train_data,
        "val_data": args.val_data,
        "image_root": args.image_root,
        "output_dir": args.output_dir,
    }

    generators = {
        "g0": generate_g0,
        "g1": generate_g1,
        "g2": generate_g2,
        "g3": generate_g3,
        "g4": generate_g4,
        "g5": generate_g5,
        "g6": generate_g6,
        "g7": generate_g7,
        "g8": generate_g8,
        "g9": generate_g9,
    }

    total = 0
    for group_name, gen_fn in generators.items():
        configs = gen_fn(base_args)
        config_dir = Path("configs") / group_name
        config_dir.mkdir(parents=True, exist_ok=True)

        for cfg in configs:
            fname = f"{cfg['run_name']}.yaml"
            with open(config_dir / fname, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        total += len(configs)
        print(f"{group_name}: {len(configs)} configs generated")

    print(f"\nTotal: {total} configs")

    # Estimate unique runs (dedup G0/G1/G2 shared configs)
    overlap = len(FITTING_LLMS)  # M/T64/D50M shared across G0, G1, G2
    print(f"Estimated unique runs (after dedup): ~{total - overlap * 2}")

    # Generate run scripts
    _generate_run_scripts(generators.keys())


def _generate_run_scripts(groups):
    """Generate shell scripts to run each group."""
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)

    # Priority order: small models first, then large
    priority_note = {
        "g0": "# Priority: HIGH (base law fitting)",
        "g1": "# Priority: HIGH (novelty: N_A marginal)",
        "g2": "# Priority: HIGH (T marginal)",
        "g3": "# Priority: HIGH (T×N_A interaction)",
        "g4": "# Priority: MEDIUM (ρ invariance)",
        "g5": "# Priority: MEDIUM (D independence)",
        "g6": "# Priority: LOW (LoRA comparison)",
        "g7": "# Priority: HIGH (extrapolation validation)",
        "g8": "# Priority: MEDIUM (depth ablation)",
        "g9": "# Priority: MEDIUM (statistical significance)",
    }

    for group in groups:
        config_dir = Path("configs") / group
        if not config_dir.exists():
            continue

        configs = sorted(config_dir.glob("*.yaml"))
        note = priority_note.get(group, "")
        lines = [
            "#!/bin/bash",
            f"# Run all {group} experiments",
            note,
            "set -e",
            "",
        ]

        for cfg_path in configs:
            lines.append(f"echo 'Running {cfg_path.name}...'")
            lines.append(f"python train.py --config {cfg_path} --use_wandb")
            lines.append("")

        script_path = scripts_dir / f"run_{group}.sh"
        with open(script_path, "w") as f:
            f.write("\n".join(lines))
        os.chmod(script_path, 0o755)

    # Master script with recommended execution order
    master_lines = [
        "#!/bin/bash",
        "# Run all experiment groups",
        "# Recommended order: G0→G1→G2→G3→G7→G4→G5→G8→G9→G6",
        "# Run small-model configs in parallel, large models sequentially",
        "set -e",
        "",
    ]
    for group in ["g0", "g1", "g2", "g3", "g7", "g4", "g5", "g8", "g9", "g6"]:
        master_lines.append(f"echo '=== {group.upper()} ==='")
        master_lines.append(f"bash scripts/run_{group}.sh")
        master_lines.append("")

    with open(scripts_dir / "run_all.sh", "w") as f:
        f.write("\n".join(master_lines))
    os.chmod(scripts_dir / "run_all.sh", 0o755)

    print("Run scripts generated in scripts/")


if __name__ == "__main__":
    main()
