#!/usr/bin/env python3
"""Generate G26 configs: re-run all 14B experiments with LR=3e-4.

Copies every 14B config from fit groups (g0v2, g1, g2) and changes:
  - lr: 0.0001 -> 0.0003
  - batch_size: 16 -> 8, grad_accum_steps: 2 -> 4  (OOM fix, same effective batch)
  - run_name: g26_... prefix
  - output_dir: checkpoints
"""
import yaml, os, glob

SRC_GROUPS = ["g0v2", "g1", "g2"]
OUT_DIR = "configs/g26"

configs = []
for grp in SRC_GROUPS:
    pattern = f"configs/{grp}/*14B*.yaml"
    configs.extend(sorted(glob.glob(pattern)))

print(f"Found {len(configs)} 14B configs to convert\n")

for src_path in configs:
    with open(src_path) as f:
        cfg = yaml.safe_load(f)

    old_name = cfg["run_name"]
    # Build new run_name: g26_<rest of original name>_lr3e4
    # Strip original group prefix
    parts = old_name.split("_", 1)  # e.g. "g0v2", "14B_T64_M_d200k_s42"
    rest = parts[1] if len(parts) > 1 else old_name
    new_name = f"g26_{rest}_lr3e4"

    cfg["lr"] = 0.0003
    cfg["batch_size"] = 8
    cfg["grad_accum_steps"] = 4
    cfg["run_name"] = new_name
    cfg["output_dir"] = "checkpoints"

    out_path = os.path.join(OUT_DIR, f"{new_name}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  {os.path.basename(src_path):45s} -> {new_name}.yaml")

print(f"\nGenerated {len(configs)} configs in {OUT_DIR}/")
