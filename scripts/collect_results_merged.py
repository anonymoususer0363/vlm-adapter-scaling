"""
Collect all experiment results from NAS + local checkpoints into a single CSV.

Merges results from two directories (NAS priority), marks G0/G5 as INVALID (D bug),
and produces a comprehensive table for analysis.

Usage:
    python scripts/collect_results_merged.py \
        --nas_dir $VLM_NAS_CHECKPOINT_DIR \
        --local_dir ./checkpoints \
        --output ./results_all.csv
"""

import argparse
import json
import csv
import os
from pathlib import Path
from collections import defaultdict


def load_run(run_dir: Path):
    """Load result.json and config.json from a run directory."""
    result_path = run_dir / "result.json"
    config_path = run_dir / "config.json"

    if not result_path.exists():
        return None, f"no result.json"

    try:
        with open(result_path) as f:
            result = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        return None, f"corrupt result.json: {e}"

    config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, Exception):
            pass

    # Extract key fields
    row = {
        "run_name": result.get("run_name", run_dir.name),
        "best_val_loss": result.get("best_val_loss"),
        "final_val_loss": result.get("final_val_loss"),
        "total_steps": result.get("total_steps"),
        "seen_pairs": result.get("seen_pairs"),
        "adapter_params": result.get("adapter_params"),
        "best_step": result.get("best_step"),
        "training_time_hours": result.get("training_time_hours"),
        # From config
        "llm_name": config.get("llm_name", ""),
        "adapter_level": config.get("adapter_level", ""),
        "num_queries": config.get("num_queries"),
        "num_samples": config.get("num_samples"),
        "num_epochs": config.get("num_epochs"),
        "seed": config.get("seed", 42),
        "adapter_num_layers": config.get("adapter_num_layers", 2),
        "adapter_d_model": config.get("adapter_d_model"),
        "use_lora": config.get("use_lora", False),
        "vision_name": config.get("vision_name", ""),
        "image_size": config.get("image_size"),
        "learning_rate": config.get("learning_rate"),
        "batch_size": config.get("batch_size"),
        # Vision metadata (may be in result or config)
        "T0": result.get("T0") or config.get("T0"),
        "rho": result.get("rho") or config.get("rho"),
        "N_A": result.get("N_A") or result.get("adapter_params"),
    }

    # Extract LLM size from name
    llm = row["llm_name"]
    row["llm_size"] = ""
    for size in ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]:
        if size in llm:
            row["llm_size"] = size
            break

    # Extract group from run_name
    rn = row["run_name"]
    row["group"] = ""
    if rn.startswith("rerun_"):
        row["group"] = "rerun"
    elif rn.startswith("g"):
        # Handle g0v2, g5v2 etc
        parts = rn.split("_")
        row["group"] = parts[0]

    # Mark D-bug invalidity
    g = row["group"]
    if g in ("g0", "g5"):
        row["valid"] = "INVALID_D_BUG"
    else:
        row["valid"] = "valid"

    # Detect 5090 duplicates
    row["source"] = ""

    return row, None


def collect_merged(nas_dir: str, local_dir: str, output: str):
    nas_path = Path(nas_dir) if nas_dir else None
    local_path = Path(local_dir) if local_dir else None

    # Gather all unique run names and their directories
    # NAS takes priority; also check local for anything NAS doesn't have
    run_dirs = {}  # run_name -> (path, source)

    if nas_path and nas_path.exists():
        for d in sorted(nas_path.iterdir()):
            if d.is_dir():
                run_dirs[d.name] = (d, "NAS")

    if local_path and local_path.exists():
        for d in sorted(local_path.iterdir()):
            if d.is_dir():
                if d.name not in run_dirs:
                    run_dirs[d.name] = (d, "local")
                # If both exist, NAS is authoritative (already set)

    print(f"Total unique run directories: {len(run_dirs)}")

    rows = []
    missing = []
    duplicates_5090 = []

    # Track runs we've seen (for dedup of _5090 suffixed copies)
    base_names_seen = set()

    for run_name in sorted(run_dirs.keys()):
        run_path, source = run_dirs[run_name]

        # Handle _5090 suffix duplicates: these are copies from 5090 server
        # Check if there's a non-_5090 version
        if run_name.endswith("_5090"):
            base_name = run_name.replace("_5090", "")
            if base_name in run_dirs:
                duplicates_5090.append(run_name)
                # Skip - the base version is authoritative
                continue

        row, err = load_run(run_path)
        if row is None:
            missing.append((run_name, source, err))
            continue

        row["source"] = source
        rows.append(row)

    # Write CSV
    if not rows:
        print("No results found.")
        return

    output_path = Path(output)
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCollected {len(rows)} results -> {output_path}")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Per-group counts
    groups = defaultdict(list)
    for r in rows:
        g = r["group"] or "unknown"
        groups[g].append(r)

    print(f"\nTotal experiments with results: {len(rows)}")
    print(f"Skipped _5090 duplicates: {len(duplicates_5090)}")
    print(f"Missing result.json: {len(missing)}")

    print("\nPer-group counts and best_val_loss range:")
    print(f"{'Group':<10} {'Count':>5}  {'Valid':>8}  {'Min Loss':>10}  {'Max Loss':>10}  {'Mean Loss':>10}")
    print("-" * 65)

    for g in sorted(groups.keys()):
        runs = groups[g]
        count = len(runs)
        valid = runs[0]["valid"] if runs else ""
        losses = [r["best_val_loss"] for r in runs if r["best_val_loss"] is not None]
        if losses:
            min_l = min(losses)
            max_l = max(losses)
            mean_l = sum(losses) / len(losses)
            print(f"{g:<10} {count:>5}  {valid:>8}  {min_l:>10.4f}  {max_l:>10.4f}  {mean_l:>10.4f}")
        else:
            print(f"{g:<10} {count:>5}  {valid:>8}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")

    # Missing experiments
    if missing:
        print(f"\nMissing result.json ({len(missing)} directories):")
        for name, src, err in missing:
            print(f"  {name} ({src}): {err}")

    # Skipped 5090 duplicates
    if duplicates_5090:
        print(f"\nSkipped _5090 duplicates ({len(duplicates_5090)}):")
        for name in duplicates_5090:
            print(f"  {name}")

    # Invalid experiments
    invalid_rows = [r for r in rows if r["valid"] != "valid"]
    if invalid_rows:
        print(f"\nINVALID experiments (D bug): {len(invalid_rows)}")
        for r in invalid_rows:
            print(f"  {r['run_name']} ({r['group']}): best_val_loss={r['best_val_loss']}")

    # Valid-only summary
    valid_rows = [r for r in rows if r["valid"] == "valid"]
    print(f"\n{'='*70}")
    print(f"Valid experiments: {len(valid_rows)} / {len(rows)}")

    # Highlight key results
    print(f"\nKey results (valid only, sorted by best_val_loss):")
    valid_with_loss = [(r["best_val_loss"], r["run_name"], r["group"])
                       for r in valid_rows if r["best_val_loss"] is not None]
    valid_with_loss.sort()

    print("\n  Top 10 (lowest loss):")
    for loss, name, group in valid_with_loss[:10]:
        print(f"    {loss:.4f}  {name}  ({group})")

    print("\n  Bottom 10 (highest loss):")
    for loss, name, group in valid_with_loss[-10:]:
        print(f"    {loss:.4f}  {name}  ({group})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _root = Path(__file__).resolve().parent.parent
    parser.add_argument("--nas_dir", type=str,
                        default=os.environ.get("VLM_NAS_CHECKPOINT_DIR", ""))
    parser.add_argument("--local_dir", type=str,
                        default=os.environ.get("VLM_CHECKPOINT_DIR",
                                               str(_root / "checkpoints")))
    parser.add_argument("--output", type=str,
                        default=str(_root / "results_all.csv"))
    args = parser.parse_args()

    collect_merged(args.nas_dir, args.local_dir, args.output)
