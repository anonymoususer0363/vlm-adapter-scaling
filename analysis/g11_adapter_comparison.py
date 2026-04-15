"""
G11 adapter type comparison analysis.

Compares Perceiver Resampler, MLP Projector, and Q-Former
at matched (N_L=3B, adapter_level=M, D=552K) across T∈{16,32,64,128}.

Produces:
- Loss vs T curves for each adapter type
- Parameter efficiency comparison (loss per N_A)
- BIC model selection if enough data

Usage:
    python analysis/g11_adapter_comparison.py --csv analysis/results_all.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_g11_data(csv_path: str) -> pd.DataFrame:
    """Load G11 + G2 baseline data from results CSV."""
    df = pd.read_csv(csv_path)

    # G11 runs: g11_3B_T{T}_M_{type}_d552k_s42
    g11 = df[df["run_name"].str.startswith("g11_")].copy()

    # G2 Perceiver baselines: g2_3B_T{T}_M_d50m_s42
    # (D bug means all G2 also trained on 552K)
    g2_baselines = df[
        df["run_name"].str.match(r"g2_3B_T(16|32|64|128)_M_")
    ].copy()

    if len(g2_baselines) > 0:
        g2_baselines = g2_baselines.copy()
        g2_baselines["adapter_type"] = "perceiver"

    # Parse adapter_type from G11 run names
    if len(g11) > 0:
        g11["adapter_type"] = g11["run_name"].apply(
            lambda x: "mlp" if "_mlp_" in x else ("qformer" if "_qformer_" in x else "perceiver")
        )

    # Combine
    combined = pd.concat([g2_baselines, g11], ignore_index=True)
    return combined


def analyze_adapter_comparison(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Compare adapter types across T values."""
    results = {
        "adapter_types": {},
        "t_comparison": {},
        "parameter_efficiency": {},
    }

    if verbose:
        print("\n" + "=" * 60)
        print("G11 ADAPTER TYPE COMPARISON")
        print("=" * 60)

    # Get adapter_type column (may need to be parsed)
    if "adapter_type" not in df.columns:
        df["adapter_type"] = df["run_name"].apply(
            lambda x: "mlp" if "_mlp_" in x else ("qformer" if "_qformer_" in x else "perceiver")
        )

    # Get N_A column
    na_col = "adapter_params" if "adapter_params" in df.columns else "adapter_total"

    # Per-adapter summary
    for atype in sorted(df["adapter_type"].unique()):
        subset = df[df["adapter_type"] == atype].sort_values("num_queries")
        if verbose:
            print(f"\n--- {atype} ---")

        n_a = subset[na_col].iloc[0] if na_col in subset.columns else 0
        t_vals = sorted(subset["num_queries"].unique())

        results["adapter_types"][atype] = {
            "N_A": int(n_a),
            "T_values": [int(t) for t in t_vals],
            "losses": {},
        }

        if verbose:
            print(f"  N_A = {n_a:,}")
            print(f"  {'T':>6} {'Loss':>10} {'Run':>40}")
            print(f"  {'-'*58}")

        for _, row in subset.iterrows():
            T = int(row["num_queries"])
            loss = row["best_val_loss"]
            results["adapter_types"][atype]["losses"][str(T)] = float(loss)
            if verbose:
                print(f"  {T:>6} {loss:>10.4f} {row['run_name']:>40}")

    # Head-to-head comparison at each T
    adapter_types = sorted(df["adapter_type"].unique())
    if verbose:
        print(f"\n--- Head-to-Head at Each T ---")
        print(f"  {'T':>6}", end="")
        for atype in adapter_types:
            print(f" {atype:>12}", end="")
        print(f" {'Best':>12}")
        print(f"  {'-'*66}")

    for T in sorted(df["num_queries"].unique()):
        t_data = df[df["num_queries"] == T]
        losses = {}
        for atype in adapter_types:
            subset = t_data[t_data["adapter_type"] == atype]
            if len(subset) > 0:
                losses[atype] = float(subset["best_val_loss"].iloc[0])

        if losses:
            best = min(losses, key=losses.get)
            results["t_comparison"][str(int(T))] = {"losses": losses, "best": best}

            if verbose:
                print(f"  {int(T):>6}", end="")
                for atype in adapter_types:
                    val = losses.get(atype, float("nan"))
                    marker = " *" if atype == best else "  "
                    print(f" {val:>10.4f}{marker}", end="")
                print(f" {best:>12}")

    # Parameter efficiency: loss / log10(N_A)
    if verbose:
        print(f"\n--- Parameter Efficiency ---")
        print(f"  {'Type':<12} {'N_A':>12} {'Mean Loss':>10} {'Loss/logN_A':>12}")
        print(f"  {'-'*50}")

    for atype in adapter_types:
        subset = df[df["adapter_type"] == atype]
        if len(subset) > 0:
            n_a = subset[na_col].iloc[0]
            mean_loss = subset["best_val_loss"].mean()
            efficiency = mean_loss / np.log10(max(n_a, 1))
            results["parameter_efficiency"][atype] = {
                "N_A": int(n_a),
                "mean_loss": float(mean_loss),
                "efficiency": float(efficiency),
            }
            if verbose:
                print(f"  {atype:<12} {n_a:>12,} {mean_loss:>10.4f} {efficiency:>12.4f}")

    # Winner summary
    if results["t_comparison"]:
        wins = {}
        for t_data in results["t_comparison"].values():
            best = t_data["best"]
            wins[best] = wins.get(best, 0) + 1
        results["winner_summary"] = wins
        if verbose:
            print(f"\n--- Winner Summary ---")
            for atype, count in sorted(wins.items(), key=lambda x: -x[1]):
                print(f"  {atype}: wins {count}/{len(results['t_comparison'])} T values")

    return results


def main():
    parser = argparse.ArgumentParser(description="G11 Adapter Comparison Analysis")
    parser.add_argument("--csv", type=str, default="analysis/results_all.csv",
                        help="Path to combined results CSV")
    parser.add_argument("--output_dir", type=str, default="analysis/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_g11_data(args.csv)
    if len(df) == 0:
        print("No G11 or G2 baseline data found. Run G11 experiments first.")
        return

    print(f"Loaded {len(df)} experiments for adapter comparison")

    # Run analysis
    results = analyze_adapter_comparison(df)

    # Save results
    output_path = output_dir / "g11_adapter_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
