"""
Select ~15 representative checkpoints for G10 downstream evaluation.

Covers the variable space systematically:
- N_L scaling (0.5B, 3B, 7B, 14B from G0v2)
- T effect (best T per N_L from G2)
- N_A scaling (S, M, XL at 3B from G1)
- T×N_A interaction (best + worst from G3)
- Compression ratio (2 configs from G4)
- Extrapolation (32B from G7)

Usage:
    python scripts/select_g10_configs.py --results_csv analysis/results_all.csv
    python scripts/select_g10_configs.py --results_csv analysis/results_all.csv --checkpoint_dir checkpoints
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def select_configs(csv_path: str, checkpoint_dir: str = "checkpoints") -> list[dict]:
    """Select ~15 representative configs for G10 eval."""
    df = pd.read_csv(csv_path)
    ckpt_dir = Path(checkpoint_dir)

    selected = []

    def add(row, reason):
        ckpt_path = ckpt_dir / row["run_name"]
        selected.append({
            "run_name": row["run_name"],
            "checkpoint": str(ckpt_path),
            "group": row["group"],
            "llm_name": row["llm_name"],
            "num_queries": row["num_queries"],
            "best_val_loss": row["best_val_loss"],
            "reason": reason,
        })

    # 1. G0v2: N_L scaling (best D per LLM, exclude 32B)
    g0v2 = df[df["group"] == "g0v2"]
    for llm in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"]:
        subset = g0v2[g0v2["llm_name"] == llm]
        if not subset.empty:
            best = subset.loc[subset["best_val_loss"].idxmin()]
            add(best, f"G0v2 N_L scaling: {llm.split('-')[-1]}")

    # 2. G2: Best T per N_L (varied T, fixed N_A=M)
    g2 = df[df["group"] == "g2"]
    for llm in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]:
        subset = g2[g2["llm_name"] == llm]
        if not subset.empty:
            best = subset.loc[subset["best_val_loss"].idxmin()]
            # Avoid duplicating if same config as G0v2 selection
            if best["run_name"] not in [s["run_name"] for s in selected]:
                add(best, f"G2 T effect: {llm.split('-')[-1]} T={int(best['num_queries'])}")

    # 3. G1: N_A scaling at 3B (S, M, XL)
    g1 = df[df["group"] == "g1"]
    g1_3b = g1[g1["llm_name"] == "Qwen/Qwen2.5-3B"]
    for level in ["S", "XL"]:  # M already in G0v2/G2
        subset = g1_3b[g1_3b["adapter_level"] == level]
        if not subset.empty:
            row = subset.iloc[0]
            if row["run_name"] not in [s["run_name"] for s in selected]:
                add(row, f"G1 N_A scaling: 3B {level}")

    # 4. G3: T×N_A interaction (best + worst)
    g3 = df[df["group"] == "g3"]
    if not g3.empty:
        best = g3.loc[g3["best_val_loss"].idxmin()]
        worst = g3.loc[g3["best_val_loss"].idxmax()]
        if best["run_name"] not in [s["run_name"] for s in selected]:
            add(best, f"G3 interaction best: T={int(best['num_queries'])} {best['adapter_level']}")
        if worst["run_name"] not in [s["run_name"] for s in selected]:
            add(worst, f"G3 interaction worst: T={int(worst['num_queries'])} {worst['adapter_level']}")

    # 5. G4: Compression ratio (2 configs, same T different resolution)
    g4 = df[df["group"] == "g4"]
    if not g4.empty:
        # Pick T=32 configs (different resolutions)
        g4_t32 = g4[g4["num_queries"] == 32]
        for _, row in g4_t32.iterrows():
            if row["run_name"] not in [s["run_name"] for s in selected]:
                add(row, f"G4 rho: T=32 img={row.get('image_size', '?')}")
                break  # Just 1-2 from G4

    # 6. G7: Extrapolation (32B if available)
    g7 = df[df["group"] == "g7"]
    g7_32b = g7[g7["llm_name"].str.contains("32B", na=False)]
    if not g7_32b.empty:
        best = g7_32b.loc[g7_32b["best_val_loss"].idxmin()]
        add(best, "G7 extrapolation: 32B")

    # Also include 7B T128 XL if available (best overall)
    g7_7b = g7[g7["llm_name"].str.contains("7B", na=False)]
    if not g7_7b.empty:
        best = g7_7b.loc[g7_7b["best_val_loss"].idxmin()]
        if best["run_name"] not in [s["run_name"] for s in selected]:
            add(best, "G7 extreme: 7B best config")

    return selected


def main():
    parser = argparse.ArgumentParser(description="Select G10 eval configs")
    parser.add_argument("--results_csv", type=str, default="analysis/results_all.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="configs/g10_eval_list.txt")
    args = parser.parse_args()

    selected = select_configs(args.results_csv, args.checkpoint_dir)

    print(f"Selected {len(selected)} configs for G10 evaluation:\n")
    print(f"{'#':<3} {'Run Name':<45} {'Loss':>8} {'Reason'}")
    print("-" * 90)
    for i, s in enumerate(selected, 1):
        print(f"{i:<3} {s['run_name']:<45} {s['best_val_loss']:>8.4f} {s['reason']}")

    # Write batch config file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# G10 evaluation checkpoints (auto-generated)\n")
        for s in selected:
            f.write(f"{s['checkpoint']}\n")
    print(f"\nBatch config written: {output_path}")

    # Also save as JSON for analysis
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(selected, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))
    print(f"Config details: {json_path}")


if __name__ == "__main__":
    main()
