"""
Generate paper figures for VLM adapter scaling law paper.

Figures:
1. log L vs log N_L, log D (basic power law)
2. L vs N_A per N_L (hook/monotone)
3. L vs T per N_L (hook + T_opt)
4. T_opt stability across N_L (golden rule)
5. L heatmap: T × N_A (interaction)
6. ρ_opt analysis
7. r_A_opt vs N_L
8. Predicted vs observed loss (joint law)
9. Extrapolation validation
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = sns.color_palette("Set2", 8)
LLM_COLORS = {
    "0.5B": COLORS[0],
    "1.5B": COLORS[1],
    "3B": COLORS[2],
    "7B": COLORS[3],
    "14B": COLORS[4],
    "32B": COLORS[5],
}
LLM_PARAMS = {"0.5B": 0.5e9, "1.5B": 1.5e9, "3B": 3e9, "7B": 7e9, "14B": 14e9, "32B": 32e9}
NA_MARKERS = {"XS": "v", "S": "s", "M": "o", "L": "D", "XL": "^"}


def _parse_llm_params(df: pd.DataFrame) -> pd.DataFrame:
    """Add llm_params column if missing."""
    if "llm_params" not in df.columns:
        df = df.copy()
        def parse(name):
            s = name.split("-")[-1]
            if s.endswith("B"):
                return float(s[:-1]) * 1e9
            elif s.endswith("M"):
                return float(s[:-1]) * 1e6
            return float(s)
        df["llm_params"] = df["llm_name"].apply(parse)
    return df


def fig1_base_power_law(df: pd.DataFrame, save_dir: str):
    """Figure 1: log L vs log N_L and log D."""
    df = _parse_llm_params(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Loss vs N_L
    ax = axes[0]
    d_col = "num_samples" if "num_samples" in df.columns else "seen_pairs"
    for d_val in sorted(df[d_col].dropna().unique()):
        subset = df[df[d_col] == d_val].sort_values("llm_params")
        d_label = f"D={int(d_val/1e3)}K" if d_val < 1e6 else f"D={d_val/1e6:.0f}M"
        ax.plot(subset["llm_params"], subset["best_val_loss"], "o-", label=d_label)
    ax.set_xscale("log")
    ax.set_xlabel("LLM Parameters (N_L)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(a) Loss vs. N_L")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Loss vs D
    ax = axes[1]
    for n_l in sorted(df["llm_name"].unique()):
        subset = df[df["llm_name"] == n_l].sort_values(d_col)
        label = n_l.split("-")[-1]
        ax.plot(subset[d_col], subset["best_val_loss"], "o-", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Training Data Size (D)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(b) Loss vs. D")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig1_base_power_law.pdf")
    plt.savefig(f"{save_dir}/fig1_base_power_law.png")
    plt.close()


def fig2_na_marginal(df: pd.DataFrame, save_dir: str):
    """Figure 2: L vs N_A for each N_L."""
    fig, ax = plt.subplots(figsize=(8, 6))

    na_col = "adapter_params" if "adapter_params" in df.columns else "adapter_total"

    for i, n_l in enumerate(sorted(df["llm_name"].unique())):
        subset = df[df["llm_name"] == n_l].sort_values(na_col)
        label = n_l.split("-")[-1]
        ax.plot(subset[na_col], subset["best_val_loss"],
                "o-", color=COLORS[i], label=label, markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Adapter Parameters (N_A)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Marginal Effect of Adapter Capacity N_A")
    ax.legend(title="LLM Size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig2_na_marginal.pdf")
    plt.savefig(f"{save_dir}/fig2_na_marginal.png")
    plt.close()


def fig3_t_marginal(df: pd.DataFrame, save_dir: str):
    """Figure 3: L vs T for each N_L (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, n_l in enumerate(sorted(df["llm_name"].unique())):
        subset = df[df["llm_name"] == n_l].sort_values("num_queries")
        label = n_l.split("-")[-1]
        ax.plot(subset["num_queries"], subset["best_val_loss"],
                "o-", color=COLORS[i], label=label, markersize=8)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Visual Token Count (T)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Marginal Effect of Visual Token Count T")
    ax.legend(title="LLM Size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig3_t_marginal.pdf")
    plt.savefig(f"{save_dir}/fig3_t_marginal.png")
    plt.close()


def fig5_interaction_heatmap(df: pd.DataFrame, save_dir: str):
    """Figure 5: L heatmap for T × N_A grid."""
    pivot = df.pivot_table(
        values="best_val_loss",
        index="num_queries",
        columns="adapter_level",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd_r", ax=ax)
    ax.set_xlabel("Adapter Level (N_A)")
    ax.set_ylabel("Visual Token Count (T)")
    ax.set_title("Validation Loss: T x N_A Interaction (N_L=3B)")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig5_interaction_heatmap.pdf")
    plt.savefig(f"{save_dir}/fig5_interaction_heatmap.png")
    plt.close()


def fig8_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str):
    """Figure 8: Predicted vs observed loss for joint scaling law."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.6, s=30, c="steelblue")

    lims = [min(y_true.min(), y_pred.min()) * 0.98,
            max(y_true.max(), y_pred.max()) * 1.02]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)

    mae = np.mean(np.abs(y_true - y_pred))
    ax.text(0.05, 0.95, f"MAE = {mae:.4f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Observed Loss")
    ax.set_ylabel("Predicted Loss")
    ax.set_title("Joint Scaling Law: Predicted vs Observed")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig8_pred_vs_true.pdf")
    plt.savefig(f"{save_dir}/fig8_pred_vs_true.png")
    plt.close()


def fig4_t_opt_vs_nl(t_opt_per_nl: dict, save_dir: str):
    """Figure 4: T_opt vs N_L scatter (golden rule)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    nl_vals, t_vals = [], []
    for llm_name, t_opt in t_opt_per_nl.items():
        if t_opt is not None and t_opt > 0 and t_opt < 1e6:
            label = llm_name.split("-")[-1]
            nl = LLM_PARAMS.get(label, None)
            if nl:
                nl_vals.append(nl)
                t_vals.append(t_opt)
                ax.scatter(nl, t_opt, s=100, color=LLM_COLORS.get(label, "gray"),
                           zorder=5, edgecolors="black", linewidth=0.5)
                ax.annotate(label, (nl, t_opt), textcoords="offset points",
                            xytext=(8, 5), fontsize=10)

    if len(nl_vals) >= 2:
        nl_arr = np.array(nl_vals)
        t_arr = np.array(t_vals)
        # Fit T_opt = a * N_L^b
        from scipy.optimize import curve_fit
        def power(x, a, b):
            return a * np.power(x, b)
        try:
            popt, _ = curve_fit(power, nl_arr, t_arr, p0=[1, 0.1], maxfev=5000)
            x_smooth = np.geomspace(min(nl_arr) * 0.5, max(nl_arr) * 2, 100)
            ax.plot(x_smooth, power(x_smooth, *popt), "k--", alpha=0.5,
                    label=f"T_opt = {popt[0]:.2g} * N_L^{popt[1]:.3f}")
            ax.legend()
        except Exception:
            pass

    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("LLM Parameters (N_L)")
    ax.set_ylabel("Optimal Token Count (T_opt)")
    ax.set_title("Golden Rule: T_opt vs N_L")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig4_t_opt_vs_nl.pdf")
    plt.savefig(f"{save_dir}/fig4_t_opt_vs_nl.png")
    plt.close()


def fig6_rho_invariance(df: pd.DataFrame, save_dir: str):
    """Figure 6: ρ=T/T₀ invariance across resolutions (G4)."""
    T0_MAP = {384: 729, 224: 256}

    df = df.copy()
    df["image_size"] = df["image_size"].astype(int)
    df["T0"] = df["image_size"].map(T0_MAP).fillna(729).astype(float)
    df["T"] = df["num_queries"].astype(float)
    df["rho"] = df["T"] / df["T0"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) L vs T, colored by resolution
    ax = axes[0]
    for res in sorted(df["image_size"].unique()):
        subset = df[df["image_size"] == res].sort_values("T")
        t0 = T0_MAP[res]
        ax.plot(subset["T"], subset["best_val_loss"], "o-", markersize=8,
                label=f"res={res} (T₀={t0})")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Visual Token Count (T)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(a) Loss vs T at Different Resolutions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) L vs ρ — do curves collapse?
    ax = axes[1]
    for res in sorted(df["image_size"].unique()):
        subset = df[df["image_size"] == res].sort_values("rho")
        t0 = T0_MAP[res]
        ax.plot(subset["rho"], subset["best_val_loss"], "o-", markersize=8,
                label=f"res={res} (T₀={t0})")
    ax.set_xscale("log")
    ax.set_xlabel("Compression Ratio ρ = T/T₀")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(b) Loss vs ρ — Invariance Test")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig6_rho_invariance.pdf")
    plt.savefig(f"{save_dir}/fig6_rho_invariance.png")
    plt.close()


def fig7_ra_opt_vs_nl(na_results: dict, save_dir: str):
    """Figure 7: r_A = N_A/N_L optimal ratio vs N_L."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot all (N_A/N_L, Loss) points colored by LLM size
    for llm_name, res in na_results.items():
        label = llm_name.split("-")[-1]
        nl = LLM_PARAMS.get(label, None)
        if not nl:
            continue
        # Get best fit info
        best = res.get("best_fit")
        ra_opt_info = res.get("ra_opt")
        if ra_opt_info and ra_opt_info.get("r_A_opt"):
            ax.scatter(nl, ra_opt_info["r_A_opt"], s=120, color=LLM_COLORS.get(label, "gray"),
                       zorder=5, edgecolors="black", linewidth=0.5)
            ax.annotate(f"{label}\nr_A={ra_opt_info['r_A_opt']:.4f}",
                        (nl, ra_opt_info["r_A_opt"]),
                        textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("LLM Parameters (N_L)")
    ax.set_ylabel("Optimal Adapter Ratio (r_A = N_A/N_L)")
    ax.set_title("Golden Rule: Optimal Adapter Ratio vs LLM Size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig7_ra_opt_vs_nl.pdf")
    plt.savefig(f"{save_dir}/fig7_ra_opt_vs_nl.png")
    plt.close()


def fig9_extrapolation(df_fit: pd.DataFrame, df_test: pd.DataFrame,
                       extrap_result: dict, save_dir: str):
    """Figure 9: Extrapolation validation (fit on small, predict large)."""
    if not extrap_result or "y_true" not in extrap_result:
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    y_true = extrap_result["y_true"]
    y_pred = extrap_result["y_pred"]

    ax.scatter(y_true, y_pred, s=80, c="crimson", edgecolors="black",
               linewidth=0.5, zorder=5, label="32B predictions")

    lims = [min(y_true.min(), y_pred.min()) * 0.98,
            max(y_true.max(), y_pred.max()) * 1.02]
    ax.plot(lims, lims, "k--", alpha=0.5)

    mape = extrap_result["mape"]
    ax.text(0.05, 0.95, f"MAPE = {mape:.2f}%\nn = {len(y_true)}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Observed Loss (32B)")
    ax.set_ylabel("Predicted Loss (from 0.5B-14B fit)")
    ax.set_title("Extrapolation Validation: 32B")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig9_extrapolation.pdf")
    plt.savefig(f"{save_dir}/fig9_extrapolation.png")
    plt.close()


def fig15_depth_ablation(df: pd.DataFrame, save_dir: str):
    """Figure 15 (Appendix B): Depth ablation results (G8)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, n_l in enumerate(sorted(df["llm_name"].unique())):
        subset = df[df["llm_name"] == n_l].sort_values("num_layers")
        label = n_l.split("-")[-1]
        if "num_layers" in subset.columns:
            ax.plot(subset["num_layers"], subset["best_val_loss"],
                    "o-", color=COLORS[i], label=label, markersize=8)

    ax.set_xlabel("Adapter Depth (number of layers)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Effect of Adapter Depth (G8)")
    ax.legend(title="LLM Size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig15_depth_ablation.pdf")
    plt.savefig(f"{save_dir}/fig15_depth_ablation.png")
    plt.close()


def fig16_seed_spread(seed_results: dict, save_dir: str):
    """Figure 16 (Appendix C): Seed spread / error bars (G9)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    configs = sorted(seed_results.keys())
    x_pos = range(len(configs))
    means = [seed_results[c]["mean"] for c in configs]
    stds = [seed_results[c]["std"] for c in configs]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=4, alpha=0.7,
                  color=[COLORS[i % len(COLORS)] for i in range(len(configs))],
                  edgecolor="black", linewidth=0.5)

    # Mark high-spread configs
    for i, c in enumerate(configs):
        spread = seed_results[c]["spread"]
        if spread > 0.1:
            ax.annotate(f"spread={spread:.2f}", (i, means[i] + stds[i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=8, color="red", ha="center")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.replace("g9_", "").replace("_d50m", "") for c in configs],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Validation Loss")
    ax.set_title("Seed Repeatability (G9): Mean ± Std")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig16_seed_spread.pdf")
    plt.savefig(f"{save_dir}/fig16_seed_spread.png")
    plt.close()


def _filter_df_for_fit(df: pd.DataFrame, joint_info: dict) -> pd.DataFrame:
    """Reconstruct the filtered DataFrame that matches joint fit y_true/y_pred."""
    group_counts = joint_info.get("group_counts", {})
    exclude_sizes = joint_info.get("exclude_llm_sizes", [])
    div_threshold = joint_info.get("divergence_threshold", 1.5)

    # Filter by groups
    mask = pd.Series(False, index=df.index)
    for g in group_counts:
        mask |= df["run_name"].str.startswith(g + "_")
    filtered = df[mask].copy()

    # Exclude LLM sizes
    for size in exclude_sizes:
        filtered = filtered[~filtered["llm_name"].str.contains(size.replace("B", "B"))]

    # Exclude divergent (gap = final - best > threshold)
    if div_threshold and "best_val_loss" in filtered.columns and "final_val_loss" in filtered.columns:
        gap = filtered["final_val_loss"] - filtered["best_val_loss"]
        filtered = filtered[gap <= div_threshold]

    # Keep original CSV order to match fit y_true/y_pred ordering
    filtered = filtered.reset_index(drop=True)
    return filtered


def fig_pred_vs_true_colored(fit_result: dict, df: pd.DataFrame, save_dir: str,
                              filename: str = "fig8_pred_vs_true",
                              joint_info: dict = None):
    """Figure 8: Predicted vs Observed with per-LLM coloring."""
    y_true = np.array(fit_result["y_true"])
    y_pred = np.array(fit_result["y_pred"])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Try to reconstruct filtered df for coloring
    colored = False
    if joint_info is not None:
        filtered_df = _filter_df_for_fit(df, joint_info)
        if len(filtered_df) == len(y_true):
            llm_labels = [n.split("-")[-1] for n in filtered_df["llm_name"]]
            for llm_key in ["1.5B", "3B", "7B", "14B", "0.5B", "32B"]:
                mask = np.array([l == llm_key for l in llm_labels])
                if mask.sum() == 0:
                    continue
                ax.scatter(y_true[mask], y_pred[mask], alpha=0.7, s=50,
                           color=LLM_COLORS.get(llm_key, "gray"),
                           edgecolors="white", linewidth=0.3, label=llm_key, zorder=3)
            colored = True

    if not colored:
        ax.scatter(y_true, y_pred, alpha=0.6, s=40, c="steelblue",
                   edgecolors="white", linewidth=0.3)

    lims = [min(y_true.min(), y_pred.min()) - 0.05,
            max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)

    r2 = fit_result.get("r_squared", 0)
    mape = fit_result.get("mape", 0)
    k = fit_result.get("k", "?")
    ax.text(0.05, 0.95, f"R² = {r2:.3f}\nMAPE = {mape:.2f}%\nk = {k}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Observed Loss")
    ax.set_ylabel("Predicted Loss")
    ax.set_title("Joint Scaling Law: Predicted vs Observed")
    if colored:
        ax.legend(title="LLM", loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.pdf")
    plt.savefig(f"{save_dir}/{filename}.png")
    plt.close()


def fig_per_llm_residuals(fit_result: dict, save_dir: str,
                           filename: str = "fig_per_llm_residuals"):
    """Per-LLM fixed effects: residual decomposition."""
    y_true = np.array(fit_result["y_true"])
    y_pred = np.array(fit_result["y_pred"])
    residuals = y_true - y_pred

    # Per-LLM epsilon from fit
    per_llm_eps = fit_result.get("per_llm_eps", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Residual histogram by LLM
    ax = axes[0]
    # We need LLM labels from the fit — approximate from y_true ordering
    # Better: use per_llm_eps directly
    if per_llm_eps:
        llms = sorted(per_llm_eps.keys(), key=lambda x: LLM_PARAMS.get(x.split("-")[-1], 0))
        labels = [l.split("-")[-1] for l in llms]
        eps_vals = [per_llm_eps[l] for l in llms]
        colors = [LLM_COLORS.get(lab, "gray") for lab in labels]

        bars = ax.bar(range(len(labels)), eps_vals, color=colors, edgecolor="black",
                      linewidth=0.5, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=12)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("ε_i (LLM bias)")
        ax.set_xlabel("LLM Size")
        ax.set_title("(a) Per-LLM Fixed Effects")
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate values
        for i, (v, bar) in enumerate(zip(eps_vals, bars)):
            ax.text(i, v + 0.003 * (1 if v >= 0 else -1),
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=10, fontweight="bold")

    # (b) Residual distribution
    ax = axes[1]
    ax.hist(residuals, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Residual (Observed - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"(b) Residual Distribution (std={np.std(residuals):.4f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.pdf")
    plt.savefig(f"{save_dir}/{filename}.png")
    plt.close()


def fig_noise_floor(noise_result: dict, joint_r2: float, perlm_r2: float,
                     save_dir: str, filename: str = "fig_noise_floor"):
    """Noise floor: R² ceiling from seed variance."""
    max_r2 = noise_result.get("max_r2_mean", 0.976)
    mean_std = noise_result.get("mean_seed_std", 0.049)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Seed std distribution
    ax = axes[0]
    seed_stds = noise_result.get("seed_stds", {})
    if seed_stds:
        configs = sorted(seed_stds.keys())
        stds = [seed_stds[c] for c in configs]
        labels = [c.replace("g9_", "").replace("_d50m", "").replace("3B_", "")
                  for c in configs]

        bars = ax.bar(range(len(stds)), stds, alpha=0.7,
                      color=[COLORS[i % len(COLORS)] for i in range(len(stds))],
                      edgecolor="black", linewidth=0.5)
        ax.axhline(mean_std, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean σ = {mean_std:.3f}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Seed Std (σ)")
        ax.set_title("(a) Per-Config Seed Variance (G9)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # (b) R² ceiling gauge
    ax = axes[1]
    models = ["Standard\nJoint", "Per-LLM\nFixed Effects", "Noise\nCeiling"]
    r2_vals = [joint_r2, perlm_r2, max_r2]
    colors_bar = ["steelblue", "darkorange", "crimson"]

    bars = ax.barh(range(len(models)), r2_vals, color=colors_bar, alpha=0.8,
                   edgecolor="black", linewidth=0.5, height=0.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlabel("R²")
    ax.set_title("(b) R² vs Noise Ceiling")
    ax.set_xlim(0.75, 1.0)
    ax.grid(True, alpha=0.3, axis="x")

    # Percentage annotations
    for i, (v, bar) in enumerate(zip(r2_vals, bars)):
        pct = v / max_r2 * 100 if i < 2 else 100
        ax.text(v + 0.003, i, f"{v:.3f} ({pct:.0f}%)",
                va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.pdf")
    plt.savefig(f"{save_dir}/{filename}.png")
    plt.close()


def fig_t_multi_d(df: pd.DataFrame, save_dir: str,
                   filename: str = "fig_t_sensitivity_multi_d"):
    """T sensitivity at multiple D values (G2 + G13)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # G2: D=552K, G13: D=200K (and D=2M when available)
    g2_3b = df[(df["run_name"].str.startswith("g2")) &
               (df["llm_name"].str.contains("3B"))].sort_values("num_queries")
    g13_200k = df[(df["run_name"].str.startswith("g13")) &
                  (df["run_name"].str.contains("d200k"))].sort_values("num_queries")
    g13_2m = df[(df["run_name"].str.startswith("g13")) &
                (df["run_name"].str.contains("d2m"))].sort_values("num_queries")

    datasets = []
    if len(g13_200k) > 0:
        datasets.append((g13_200k, "D=200K", COLORS[0], "o"))
    if len(g2_3b) > 0:
        datasets.append((g2_3b, "D=552K", COLORS[1], "s"))
    if len(g13_2m) > 0:
        datasets.append((g13_2m, "D=2M", COLORS[2], "D"))

    for subset, label, color, marker in datasets:
        ax.plot(subset["num_queries"], subset["best_val_loss"],
                f"{marker}-", color=color, label=label, markersize=8, linewidth=2)
        # Annotate range
        loss_range = subset["best_val_loss"].max() - subset["best_val_loss"].min()
        ax.annotate(f"range={loss_range:.3f}",
                    xy=(subset["num_queries"].iloc[-1], subset["best_val_loss"].iloc[-1]),
                    textcoords="offset points", xytext=(10, 0), fontsize=9,
                    color=color, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Visual Token Count (T)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("T Sensitivity Across Data Sizes (3B, N_A=M)")
    ax.legend(title="Training Data")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.pdf")
    plt.savefig(f"{save_dir}/{filename}.png")
    plt.close()


def fig_extrapolation(extrap_result: dict, save_dir: str,
                       filename: str = "fig9_extrapolation"):
    """Extrapolation validation: fit on ≤14B, predict 32B."""
    y_true = np.array(extrap_result["y_true"])
    y_pred = np.array(extrap_result["y_pred"])
    mape = extrap_result.get("mape", 0)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, s=100, c="crimson", edgecolors="black",
               linewidth=0.5, zorder=5, label="32B predictions")

    lims = [min(y_true.min(), y_pred.min()) - 0.05,
            max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.5)

    # Error bars (diagonal distance)
    for yt, yp in zip(y_true, y_pred):
        ax.plot([yt, yt], [yt, yp], "r-", alpha=0.3, linewidth=1)

    ax.text(0.05, 0.95, f"MAPE = {mape:.2f}%\nn = {len(y_true)}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Observed Loss (32B)")
    ax.set_ylabel("Predicted Loss (from ≤14B fit)")
    ax.set_title("Extrapolation: 32B Out-of-Distribution")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.pdf")
    plt.savefig(f"{save_dir}/{filename}.png")
    plt.close()


def fig10_d_independence(df: pd.DataFrame, save_dir: str):
    """Figure 10: D-independence of T_opt (G5v2)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    na_col = "adapter_params" if "adapter_params" in df.columns else "adapter_total"

    # Get D levels
    d_col = "num_samples" if "num_samples" in df.columns else "seen_pairs"
    d_levels = sorted(df[d_col].unique())

    for i, d_val in enumerate(d_levels):
        subset = df[df[d_col] == d_val].sort_values("num_queries")
        # Filter to T sweep (adapter_level == M)
        if "adapter_level" in subset.columns:
            subset = subset[subset["adapter_level"] == "M"]
        if len(subset) >= 2:
            d_label = f"D={int(d_val/1e3)}K" if d_val < 1e6 else f"D={d_val/1e6:.0f}M"
            ax.plot(subset["num_queries"], subset["best_val_loss"],
                    "o-", color=COLORS[i], label=d_label, markersize=8)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Visual Token Count (T)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("D-Independence: L(T) at Different Data Sizes (G5v2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig10_d_independence.pdf")
    plt.savefig(f"{save_dir}/fig10_d_independence.png")
    plt.close()


def generate_all_figures(results_dir: str = None, csv_path: str = None,
                         json_path: str = None, save_dir: str = "analysis/figures"):
    """Generate all figures from experiment results."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        from .scaling_fit import load_results
        df = load_results(results_dir)

    if len(df) == 0:
        print("No results to plot.")
        return

    # Load JSON results if available
    fit_results = None
    if json_path:
        with open(json_path) as f:
            fit_results = json.load(f)

    na_col = "adapter_params" if "adapter_params" in df.columns else "adapter_total"
    valid = df[~df["run_name"].str.match(r"^g[05]_")]

    # Figure 1: Base power law (G0v2 data)
    g0v2 = df[df["run_name"].str.startswith("g0v2")]
    if len(g0v2) > 0:
        fig1_base_power_law(g0v2, save_dir)
        print("Figure 1 (base power law) saved.")

    # Figure 2: N_A marginal (G1 data)
    g1 = valid[valid["run_name"].str.startswith("g1")]
    if len(g1) > 0:
        fig2_na_marginal(g1, save_dir)
        print("Figure 2 (N_A marginal) saved.")

    # Figure 3: T marginal (G2 data)
    g2 = valid[valid["run_name"].str.startswith("g2")]
    if len(g2) > 0:
        fig3_t_marginal(g2, save_dir)
        print("Figure 3 (T marginal) saved.")

    # Figure 5: Interaction heatmap (G3 data)
    g3 = valid[valid["run_name"].str.startswith("g3")]
    if len(g3) > 0:
        fig5_interaction_heatmap(g3, save_dir)
        print("Figure 5 (interaction) saved.")

    # Figure 6: ρ invariance (G4 data)
    g4 = valid[valid["run_name"].str.startswith("g4")]
    if len(g4) > 0:
        fig6_rho_invariance(g4, save_dir)
        print("Figure 6 (rho invariance) saved.")

    # Figure 10: D independence (G5v2 data)
    g5v2 = df[df["run_name"].str.startswith("g5v2")]
    if len(g5v2) > 0:
        fig10_d_independence(g5v2, save_dir)
        print("Figure 10 (D independence) saved.")

    # --- Figures from JSON results ---
    if fit_results:
        # Figure 8: Pred vs True (joint law)
        joint_perlm = fit_results.get("joint_per_llm_eps")
        joint = fit_results.get("joint")
        if joint_perlm:
            fig_pred_vs_true_colored(joint_perlm, df, save_dir,
                                      "fig8_pred_vs_true_perlm", joint_info=joint)
            print("Figure 8 (pred vs true, per-LLM eps) saved.")
        if joint and "y_true" in joint:
            # Standard joint
            joint_fit = joint["fits"][0] if "fits" in joint else joint
            fig_pred_vs_true_colored(
                {"y_true": joint["y_true"], "y_pred": joint["y_pred"],
                 "r_squared": joint_fit.get("r_squared", 0),
                 "mape": joint.get("mape", 0), "k": joint_fit.get("k", "?")},
                df, save_dir, "fig8_pred_vs_true", joint_info=joint)
            print("Figure 8 (pred vs true, standard) saved.")

        # Per-LLM residuals
        if joint_perlm:
            fig_per_llm_residuals(joint_perlm, save_dir)
            print("Figure per-LLM residuals saved.")

        # Noise floor
        noise = fit_results.get("noise_floor")
        if noise:
            joint_r2 = 0.803
            perlm_r2 = 0.827
            if joint and "fits" in joint:
                joint_r2 = joint["fits"][0].get("r_squared", 0.803)
            if joint_perlm:
                perlm_r2 = joint_perlm.get("r_squared", 0.827)
            fig_noise_floor(noise, joint_r2, perlm_r2, save_dir)
            print("Figure noise floor saved.")

        # Extrapolation (32B)
        extrap = fit_results.get("extrapolation")
        if extrap and "y_true" in extrap:
            fig_extrapolation(extrap, save_dir)
            print("Figure 9 (extrapolation) saved.")

        # T_opt golden rule
        t_marg = fit_results.get("t_marginal")
        if t_marg:
            t_opt_map = {}
            for llm_name, res in t_marg.items():
                t_opt_info = res.get("T_opt")
                if isinstance(t_opt_info, dict):
                    t_opt_map[llm_name] = t_opt_info.get("T_opt")
                elif isinstance(t_opt_info, (int, float)):
                    t_opt_map[llm_name] = t_opt_info
            if t_opt_map:
                fig4_t_opt_vs_nl(t_opt_map, save_dir)
                print("Figure 4 (T_opt golden rule) saved.")

        # r_A_opt
        na_marg = fit_results.get("na_marginal")
        if na_marg:
            fig7_ra_opt_vs_nl(na_marg, save_dir)
            print("Figure 7 (r_A_opt) saved.")

        # Seed spread
        seed = fit_results.get("seed_stability")
        if seed:
            fig16_seed_spread(seed, save_dir)
            print("Figure 16 (seed spread) saved.")

    # --- Figures from CSV: multi-D T sensitivity ---
    fig_t_multi_d(df, save_dir)
    print("Figure T multi-D sensitivity saved.")

    # Depth ablation (G8)
    g8 = valid[valid["run_name"].str.startswith("g8")]
    if len(g8) > 0 and "num_layers" in g8.columns:
        fig15_depth_ablation(g8, save_dir)
        print("Figure 15 (depth ablation) saved.")

    print(f"\nAll figures saved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default=os.environ.get("VLM_CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--json", type=str, default=None,
                        help="Path to scaling_fit_results.json")
    parser.add_argument("--save_dir", type=str, default="analysis/figures")
    args = parser.parse_args()
    generate_all_figures(results_dir=args.results_dir, csv_path=args.csv,
                         json_path=args.json, save_dir=args.save_dir)
