"""
Combined result collector: reads from NAS + local, deduplicates, outputs CSV.
"""
import json
import csv
import os
from pathlib import Path


def read_run(run_dir):
    """Read result.json and config.json from a run directory."""
    result_path = run_dir / "result.json"
    config_path = run_dir / "config.json"

    if not result_path.exists():
        return None, run_dir.name

    with open(result_path) as f:
        result = json.load(f)

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Extract key fields
    row = {
        "run_name": result.get("run_name", run_dir.name),
        "source": "",  # will be set by caller
        "final_val_loss": result.get("final_val_loss"),
        "best_val_loss": result.get("best_val_loss"),
        "total_steps": result.get("total_steps"),
        "seen_pairs": result.get("seen_pairs"),
        "adapter_params": result.get("adapter_params"),
        # Vision metadata from result.json
        "vision_T0": result.get("vision_T0"),
        "vision_T": result.get("vision_T"),
        "vision_rho": result.get("vision_rho"),
        "vision_N_A": result.get("vision_N_A"),
        "vision_d_model": result.get("vision_d_model"),
        "vision_adapter_num_layers": result.get("vision_adapter_num_layers"),
        "vision_image_size": result.get("vision_image_size"),
        # From config
        "llm_name": config.get("llm_name", ""),
        "adapter_level": config.get("adapter_level", ""),
        "num_queries": config.get("num_queries"),
        "num_samples": config.get("num_samples"),
        "num_epochs": config.get("num_epochs", 1),
        "seed": config.get("seed", 42),
        "adapter_num_layers": config.get("adapter_num_layers", 2),
        "adapter_type": config.get("adapter_type", "perceiver"),
        "use_lora": config.get("use_lora", False),
        "vision_name": config.get("vision_name", ""),
        "batch_size": config.get("batch_size"),
        "lr": config.get("lr"),
        # Additional config fields
        "llm_params": config.get("llm"),
        "vision_params": config.get("vision_encoder"),
        "total_params": config.get("total"),
        "total_trainable": config.get("total_trainable"),
        # Derived
        "llm_size": "",
        "group": "",
        "image_size": "",
    }

    # Parse image_size from vision_name or result
    if result.get("vision_image_size"):
        row["image_size"] = result["vision_image_size"]
    elif "224" in config.get("vision_name", ""):
        row["image_size"] = 224
    else:
        row["image_size"] = 384  # default SigLIP

    # Extract LLM size from name
    llm = row["llm_name"]
    for size in ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]:
        if size in llm:
            row["llm_size"] = size
            break

    # Extract group from run_name
    rn = row["run_name"]
    if rn.startswith("rerun"):
        row["group"] = "rerun"
    elif rn.startswith("g"):
        # Handle g0v2, g5v2, etc.
        parts = rn.split("_")
        row["group"] = parts[0]

    return row, None


def collect_all():
    project_root = Path(__file__).resolve().parent.parent
    nas_dir = Path(os.environ.get("VLM_NAS_CHECKPOINT_DIR", ""))
    local_dir = Path(os.environ.get("VLM_CHECKPOINT_DIR", project_root / "checkpoints"))
    output_path = project_root / "analysis" / "results_full.csv"

    # Collect from both sources
    all_rows = {}  # run_name -> row (NAS takes priority for dedup)
    incomplete = []
    duplicates_skipped = []

    # First collect from NAS
    if nas_dir.exists():
        for run_dir in sorted(nas_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            row, skip_name = read_run(run_dir)
            if row is None:
                incomplete.append(("NAS", skip_name))
                continue
            row["source"] = "NAS"
            all_rows[row["run_name"]] = row

    # Then collect from local (only add if not already from NAS)
    if local_dir.exists():
        for run_dir in sorted(local_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            row, skip_name = read_run(run_dir)
            if row is None:
                incomplete.append(("local", skip_name))
                continue
            if row["run_name"] in all_rows:
                duplicates_skipped.append(row["run_name"])
                continue
            row["source"] = "local"
            all_rows[row["run_name"]] = row

    rows = [all_rows[k] for k in sorted(all_rows.keys())]

    if not rows:
        print("No results found.")
        return

    # Write full CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write analysis-friendly CSV (minimal columns for scaling_fit.py / plot_figures.py)
    analysis_cols = ["run_name", "group", "best_val_loss", "final_val_loss", "total_steps",
                     "seen_pairs",
                     "llm_name", "num_queries", "adapter_level", "adapter_type", "adapter_params",
                     "num_samples", "num_epochs", "image_size"]
    analysis_path = output_path.parent / "results_all.csv"
    with open(analysis_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=analysis_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Analysis CSV: {analysis_path} ({len(rows)} rows)")

    print(f"\n=== Collection Summary ===")
    print(f"Total experiments with results: {len(rows)}")
    print(f"Output: {output_path}")

    # Group breakdown
    groups = {}
    for r in rows:
        g = r["group"] or "unknown"
        groups[g] = groups.get(g, 0) + 1

    print(f"\n--- Per-group counts ---")
    for g in sorted(groups):
        print(f"  {g}: {groups[g]}")

    # Source breakdown
    sources = {}
    for r in rows:
        s = r["source"]
        sources[s] = sources.get(s, 0) + 1
    print(f"\n--- By source ---")
    for s in sorted(sources):
        print(f"  {s}: {sources[s]}")

    # Duplicates
    if duplicates_skipped:
        print(f"\n--- Duplicates (local skipped, NAS used): {len(duplicates_skipped)} ---")

    # Incomplete
    if incomplete:
        print(f"\n--- Incomplete (no result.json): {len(incomplete)} ---")
        for src, name in incomplete:
            print(f"  [{src}] {name}")

    # Sample rows
    print(f"\n--- CSV Columns ({len(fieldnames)}) ---")
    print(", ".join(fieldnames))

    print(f"\n--- Sample rows (first 5) ---")
    for r in rows[:5]:
        print(f"  {r['run_name']}: best_val_loss={r['best_val_loss']}, "
              f"llm_size={r['llm_size']}, T={r['num_queries']}, "
              f"adapter={r['adapter_level']}, group={r['group']}")

    print(f"\n--- Sample rows (last 5) ---")
    for r in rows[-5:]:
        print(f"  {r['run_name']}: best_val_loss={r['best_val_loss']}, "
              f"llm_size={r['llm_size']}, T={r['num_queries']}, "
              f"adapter={r['adapter_level']}, group={r['group']}")


if __name__ == "__main__":
    collect_all()
