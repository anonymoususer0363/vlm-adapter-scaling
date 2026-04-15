#!/usr/bin/env python3
"""Generate G3v2, G4v2, G30 configs for LR=3e-4 validation experiments.

G3v2: T×N_A interaction re-run (3B, 16 configs) — validates Claim 5 (additive)
G4v2: ρ invariance re-run (3B, 10 configs) — validates ρ robustness
G30:  32B extrapolation (6 configs) — validates new law at 32B

All use LR=3e-4, D=552K (1 epoch), no num_samples.
"""
import yaml, os

COMMON = {
    "seed": 42,
    "num_epochs": 1,
    "eval_interval_steps": 500,
    "save_interval_steps": 2000,
    "train_data": "data/processed/train.jsonl",
    "val_data": "data/processed/val.jsonl",
    "image_root": "data/llava_pretrain",
    "output_dir": "checkpoints",
    "lr": 0.0003,
}


def write_config(out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cfg['run_name']}.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def generate_g3v2():
    """G3v2: 3B, T={8,32,64,128} × N_A={S,M,L,XL}, D=552K, LR=3e-4"""
    out_dir = "configs/g3v2"
    T_values = [8, 32, 64, 128]
    NA_levels = ["S", "M", "L", "XL"]
    configs = []

    for t in T_values:
        for na in NA_levels:
            name = f"g3v2_3B_T{t}_{na}_d552k_s42_lr3e4"
            cfg = {
                "llm_name": "Qwen/Qwen2.5-3B",
                "adapter_level": na,
                "num_queries": t,
                "run_name": name,
                "batch_size": 32,
                "grad_accum_steps": 1,
                **COMMON,
            }
            path = write_config(out_dir, cfg)
            configs.append(path)

    print(f"G3v2: {len(configs)} configs in {out_dir}/")
    return configs


def generate_g4v2():
    """G4v2: 3B, M, T={8,16,32,64,128} × res={224,384}, D=552K, LR=3e-4"""
    out_dir = "configs/g4v2"
    T_values = [8, 16, 32, 64, 128]
    configs = []

    for t in T_values:
        for res in [224, 384]:
            name = f"g4v2_3B_T{t}_M_d552k_res{res}_s42_lr3e4"
            cfg = {
                "llm_name": "Qwen/Qwen2.5-3B",
                "adapter_level": "M",
                "num_queries": t,
                "run_name": name,
                "batch_size": 32,
                "grad_accum_steps": 1,
                **COMMON,
            }
            if res == 224:
                cfg["vision_name"] = "google/siglip-so400m-patch14-224"
            # res=384 uses default (no vision_name needed)
            path = write_config(out_dir, cfg)
            configs.append(path)

    print(f"G4v2: {len(configs)} configs in {out_dir}/")
    return configs


def generate_g30():
    """G30: 32B extrapolation, 6 configs, D=552K, LR=3e-4"""
    out_dir = "configs/g30"
    specs = [
        # (T, N_A) — original G7 32B replicas
        (32, "M"),
        (64, "M"),
        (64, "L"),
        (128, "M"),
        # N_A hook verification at 32B
        (64, "XS"),
        (64, "XL"),
    ]
    configs = []

    for t, na in specs:
        name = f"g30_32B_T{t}_{na}_d552k_s42_lr3e4"
        cfg = {
            "llm_name": "Qwen/Qwen2.5-32B",
            "adapter_level": na,
            "num_queries": t,
            "run_name": name,
            "batch_size": 8,
            "grad_accum_steps": 4,
            **COMMON,
        }
        path = write_config(out_dir, cfg)
        configs.append(path)

    print(f"G30: {len(configs)} configs in {out_dir}/")
    return configs


if __name__ == "__main__":
    g3v2 = generate_g3v2()
    g4v2 = generate_g4v2()
    g30 = generate_g30()
    print(f"\nTOTAL: {len(g3v2) + len(g4v2) + len(g30)} configs generated")
