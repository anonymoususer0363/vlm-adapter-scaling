#!/usr/bin/env python3
"""Generate G27/G28/G29 configs: re-run 7B/3B/1.5B experiments with LR=3e-4.

Same logic as G26 (14B re-run):
  - Copies configs from fit groups (g0v2, g1, g2) for each LLM size
  - lr: 0.0001 -> 0.0003
  - Removes num_samples from g1/g2 (was 50M no-op, now would cause 90+ epochs)
  - Preserves g0v2 num_samples (d50k/d200k) and num_epochs (d2m=4ep, d5m=10ep)
  - Preserves original batch_size/grad_accum (no OOM issue for <=7B)

Groups:
  G27: 7B  (batch_size=16, grad_accum=2)
  G28: 3B  (batch_size=32, grad_accum=1)
  G29: 1.5B (batch_size=32, grad_accum=1)
"""
import yaml, os, glob

SRC_GROUPS = ["g0v2", "g1", "g2"]

LLM_CONFIGS = {
    "7B":   {"group": "g27", "filter": "7B"},
    "3B":   {"group": "g28", "filter": "3B"},
    "1.5B": {"group": "g29", "filter": "1.5B"},
}

total = 0
for llm_size, info in LLM_CONFIGS.items():
    grp_name = info["group"]
    filt = info["filter"]
    out_dir = f"configs/{grp_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Collect source configs
    src_configs = []
    for grp in SRC_GROUPS:
        pattern = f"configs/{grp}/*{filt}*.yaml"
        src_configs.extend(sorted(glob.glob(pattern)))

    seen_names = set()
    count = 0

    print(f"\n=== {grp_name} ({llm_size}) ===")

    for src_path in src_configs:
        with open(src_path) as f:
            cfg = yaml.safe_load(f)

        old_name = cfg["run_name"]
        src_grp = os.path.basename(os.path.dirname(src_path))  # e.g. "g0v2"

        # Build new run_name: g27_<rest>_lr3e4
        parts = old_name.split("_", 1)  # e.g. "g0v2", "7B_T64_M_d200k_s42"
        rest = parts[1] if len(parts) > 1 else old_name
        new_name = f"{grp_name}_{rest}_lr3e4"

        # Skip duplicates (g1_*_T64_M and g2_*_T64_M generate same name)
        if new_name in seen_names:
            print(f"  [SKIP DUP] {new_name}")
            continue
        seen_names.add(new_name)

        # Change LR
        cfg["lr"] = 0.0003
        cfg["run_name"] = new_name
        cfg["output_dir"] = "checkpoints"

        # Remove num_samples for g1/g2 (50M no-op fix)
        if src_grp in ("g1", "g2"):
            ns = cfg.get("num_samples")
            if ns is not None and ns >= 1_000_000:
                del cfg["num_samples"]
                print(f"  {os.path.basename(src_path):50s} -> {new_name}.yaml  [removed num_samples={ns}]")
            else:
                print(f"  {os.path.basename(src_path):50s} -> {new_name}.yaml")
        else:
            print(f"  {os.path.basename(src_path):50s} -> {new_name}.yaml")

        out_path = os.path.join(out_dir, f"{new_name}.yaml")
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        count += 1

    print(f"  -> {count} configs in {out_dir}/")
    total += count

print(f"\n=== TOTAL: {total} configs generated ===")
