"""
G10 downstream correlation analysis.

Computes Pearson & Spearman correlation between training validation loss
and downstream task metrics from both generation and PPL evaluation.

Usage:
    python analysis/g10_correlation.py --eval_dirs eval_results
    python analysis/g10_correlation.py --eval_dirs eval_results eval_results_ppl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_eval_results(eval_dirs: list[str]) -> pd.DataFrame:
    """Load and merge eval result JSONs into a DataFrame keyed by checkpoint."""
    rows_by_ckpt = {}

    for eval_dir in eval_dirs:
        eval_path = Path(eval_dir)
        if not eval_path.exists():
            print(f"Warning: eval dir not found, skipping: {eval_dir}")
            continue

        for f in sorted(eval_path.glob("eval_*.json")):
            if f.name.startswith("eval_summary"):
                continue

            with open(f) as fh:
                data = json.load(fh)

            ckpt = data["checkpoint"]
            row = rows_by_ckpt.get(
                ckpt,
                {
                    "checkpoint": ckpt,
                    "llm_name": data["llm_name"],
                    "num_queries": data["num_queries"],
                    "adapter_params": data["adapter_params"],
                    "train_val_loss": data["train_val_loss"],
                },
            )

            if "vqav2" in data:
                row["vqav2_acc"] = data["vqav2"].get("vqa_accuracy")
                row["vqav2_nll"] = data["vqav2"].get("mean_answer_nll")
            if "textvqa" in data:
                row["textvqa_acc"] = data["textvqa"].get("vqa_accuracy")
                row["textvqa_nll"] = data["textvqa"].get("mean_answer_nll")
            if "coco_caption" in data:
                row["cider"] = data["coco_caption"].get("CIDEr", data["coco_caption"].get("CIDEr_approx"))
                row["caption_nll"] = data["coco_caption"].get("mean_caption_nll")

            rows_by_ckpt[ckpt] = row

    return pd.DataFrame([rows_by_ckpt[k] for k in sorted(rows_by_ckpt)])


def compute_correlation(x, y, name_x="X", name_y="Y"):
    """Compute Pearson and Spearman correlation with p-values."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    return {
        "x": name_x,
        "y": name_y,
        "n": len(x),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 6),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 6),
    }


def analyze_correlations(df: pd.DataFrame) -> dict:
    """Run all correlation analyses."""
    results = {}

    loss = df["train_val_loss"].values

    # Generate metrics (expect negative: lower loss -> higher accuracy/CIDEr)
    if "vqav2_acc" in df.columns:
        corr = compute_correlation(loss, df["vqav2_acc"].values, "val_loss", "VQAv2 Acc")
        if corr:
            results["vqav2"] = corr
            print(
                f"VQAv2:       Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f}), "
                f"Spearman rho={corr['spearman_r']:.3f} (p={corr['spearman_p']:.4f})"
            )

    if "textvqa_acc" in df.columns:
        corr = compute_correlation(loss, df["textvqa_acc"].values, "val_loss", "TextVQA Acc")
        if corr:
            results["textvqa"] = corr
            print(
                f"TextVQA:     Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f}), "
                f"Spearman rho={corr['spearman_r']:.3f} (p={corr['spearman_p']:.4f})"
            )

    if "cider" in df.columns:
        corr = compute_correlation(loss, df["cider"].values, "val_loss", "CIDEr")
        if corr:
            results["cider"] = corr
            print(
                f"CIDEr:       Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f}), "
                f"Spearman rho={corr['spearman_r']:.3f} (p={corr['spearman_p']:.4f})"
            )

    # PPL metrics (expect positive: lower val loss -> lower downstream NLL)
    if "vqav2_nll" in df.columns:
        corr = compute_correlation(loss, df["vqav2_nll"].values, "val_loss", "VQAv2 NLL")
        if corr:
            results["vqav2_nll"] = corr
            print(
                f"VQAv2 NLL:   Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f}), "
                f"Spearman rho={corr['spearman_r']:.3f} (p={corr['spearman_p']:.4f})"
            )

    if "textvqa_nll" in df.columns:
        corr = compute_correlation(loss, df["textvqa_nll"].values, "val_loss", "TextVQA NLL")
        if corr:
            results["textvqa_nll"] = corr
            print(
                f"TextVQA NLL: Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f}), "
                f"Spearman rho={corr['spearman_r']:.3f} (p={corr['spearman_p']:.4f})"
            )

    if "caption_nll" in df.columns:
        corr = compute_correlation(loss, df["caption_nll"].values, "val_loss", "Caption NLL")
        if corr:
            results["caption_nll"] = corr
            print(
                f"Caption NLL: Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f}), "
                f"Spearman rho={corr['spearman_r']:.3f} (p={corr['spearman_p']:.4f})"
            )

    all_r = [abs(v["pearson_r"]) for v in results.values()]
    if all_r:
        avg_r = sum(all_r) / len(all_r)
        results["summary"] = {
            "avg_abs_pearson_r": round(avg_r, 4),
            "threshold_met": avg_r > 0.85,
        }
        print(
            f"\nAvg |r| = {avg_r:.3f}  "
            f"{'threshold met (>0.85)' if avg_r > 0.85 else 'below threshold'}"
        )

    return results


def plot_correlation(df: pd.DataFrame, save_dir: str):
    """Generate Figure 11: NLL vs downstream scatter plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    metrics = []
    if "vqav2_acc" in df.columns:
        metrics.append(("vqav2_acc", "VQAv2 Accuracy (%)"))
    if "textvqa_acc" in df.columns:
        metrics.append(("textvqa_acc", "TextVQA Accuracy (%)"))
    if "cider" in df.columns:
        metrics.append(("cider", "CIDEr"))
    if "vqav2_nll" in df.columns:
        metrics.append(("vqav2_nll", "VQAv2 NLL"))
    if "textvqa_nll" in df.columns:
        metrics.append(("textvqa_nll", "TextVQA NLL"))
    if "caption_nll" in df.columns:
        metrics.append(("caption_nll", "Caption NLL"))

    if not metrics:
        return

    n_panels = len(metrics)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    loss = pd.to_numeric(df["train_val_loss"], errors="coerce").to_numpy(dtype=float)

    for ax, (col, label) in zip(axes, metrics):
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = ~(np.isnan(loss) | np.isnan(y))
        x_clean, y_clean = loss[mask], y[mask]

        ax.scatter(x_clean, y_clean, s=60, alpha=0.8, edgecolors="black", linewidth=0.5)

        if len(x_clean) >= 3:
            z = np.polyfit(x_clean, y_clean, 1)
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)

            r, p = stats.pearsonr(x_clean, y_clean)
            ax.text(
                0.05,
                0.95,
                f"r = {r:.3f}\np = {p:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel("Training Validation Loss (NLL)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    plt.tight_layout()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "fig11_downstream_correlation.pdf")
    plt.savefig(save_path / "fig11_downstream_correlation.png", dpi=150)
    plt.close()
    print(f"\nFigure 11 saved: {save_path / 'fig11_downstream_correlation.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="G10 Correlation Analysis")
    parser.add_argument(
        "--eval_dirs",
        nargs="+",
        default=["eval_results"],
        help="One or more eval result dirs, e.g. eval_results eval_results_ppl",
    )
    parser.add_argument("--save_dir", type=str, default="analysis/figures")
    parser.add_argument("--output", type=str, default="analysis/g10_correlation_results.json")
    args = parser.parse_args()

    print("Loading evaluation results...")
    df = load_eval_results(args.eval_dirs)
    print(f"  {len(df)} checkpoints loaded\n")

    if df.empty:
        print("No evaluation results found!")
        return

    display_cols = [
        "checkpoint",
        "train_val_loss",
        "vqav2_acc",
        "textvqa_acc",
        "cider",
        "vqav2_nll",
        "textvqa_nll",
        "caption_nll",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))
    print()

    results = analyze_correlations(df)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _deep_convert(d):
        if isinstance(d, dict):
            return {k: _deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_deep_convert(v) for v in d]
        return _convert(d)

    with open(args.output, "w") as f:
        json.dump(_deep_convert(results), f, indent=2)
    print(f"\nResults saved: {args.output}")

    plot_correlation(df, args.save_dir)


if __name__ == "__main__":
    main()
