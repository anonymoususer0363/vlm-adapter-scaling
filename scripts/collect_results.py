"""
Collect all experiment results into a single CSV for analysis.

Usage:
    python scripts/collect_results.py --results_dir checkpoints/ --output results.csv

Reads result.json and config.json from each run directory and
combines them into one table.
"""

import argparse
import json
import csv
import os
from pathlib import Path


def collect(results_dir: str, output: str):
    results_dir = Path(results_dir)
    rows = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        result_path = run_dir / "result.json"
        config_path = run_dir / "config.json"

        if not result_path.exists():
            print(f"  SKIP {run_dir.name}: no result.json (possibly incomplete)")
            continue

        with open(result_path) as f:
            result = json.load(f)

        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        # Extract key fields
        row = {
            "run_name": result.get("run_name", run_dir.name),
            "final_val_loss": result.get("final_val_loss"),
            "best_val_loss": result.get("best_val_loss"),
            "total_steps": result.get("total_steps"),
            "adapter_params": result.get("adapter_params"),
            # From config
            "llm_name": config.get("llm_name", ""),
            "adapter_level": config.get("adapter_level", ""),
            "num_queries": config.get("num_queries"),
            "num_samples": config.get("num_samples"),
            "seed": config.get("seed", 42),
            "adapter_num_layers": config.get("adapter_num_layers", 2),
            "use_lora": config.get("use_lora", False),
            "vision_name": config.get("vision_name", ""),
            # Derived
            "llm_size": "",
            "group": "",
        }

        # Extract LLM size from name
        llm = row["llm_name"]
        for size in ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]:
            if size in llm:
                row["llm_size"] = size
                break

        # Extract group from run_name
        rn = row["run_name"]
        if rn.startswith("g"):
            row["group"] = rn.split("_")[0]

        rows.append(row)

    if not rows:
        print("No results found.")
        return

    # Write CSV
    output_path = Path(output)
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCollected {len(rows)} results → {output_path}")

    # Summary
    groups = {}
    for r in rows:
        g = r["group"] or "unknown"
        groups[g] = groups.get(g, 0) + 1

    print("\nPer-group counts:")
    for g in sorted(groups):
        print(f"  {g}: {groups[g]}")

    # Check for missing/failed
    total_expected = 174  # from generate_configs.py
    if len(rows) < total_expected:
        print(f"\nWARNING: {total_expected - len(rows)} runs missing or incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default=os.environ.get("VLM_CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    collect(args.results_dir, args.output)
