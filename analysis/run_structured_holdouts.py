"""Structured holdout validation for the Phase 2 selected-LR additive joint law.

Runs four leave-one-X-out validations on the canonical N=60 fit pool:
  - leave-one-N_L (LLM scale) out
  - leave-one-D (seen-pair exposure) out
  - leave-one-T (token count) out
  - leave-one-width (adapter width / level) out

For each axis, we hold out all points at one level, refit the additive joint law
on the remaining points, and report MAPE on the held-out points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from scaling_fit import (  # noqa: E402
    fit_joint_additive_only,
    prepare_joint_fit_data,
    prepare_joint_variant_data,
    _extract_joint_fit_arrays,
    _predict_joint_fit_result,
    _compute_mape,
)


PHASE2_GROUPS = ["g26", "g27", "g28", "g29"]
EXCLUDE_SIZES = ["0.5B"]
CSV_PATH = ROOT / "analysis" / "results_dedup_B_keep_d50m_v2.csv"


def build_pool() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df, _ = prepare_joint_fit_data(df, PHASE2_GROUPS, verbose=False)
    df, _ = prepare_joint_variant_data(
        df,
        exclude_llm_sizes=EXCLUDE_SIZES,
        divergence_threshold=0.5,
        auto_exclude_divergent=True,
        verbose=False,
    )
    return df.copy()


def adapter_level_from_params(p: int) -> str:
    # Phase 2 widths span a 30x range; level by integer-thresholded log
    bins = [(6e6, "XS"), (2e7, "S"), (4e7, "M"), (7e7, "L"), (2e8, "XL")]
    for thresh, lvl in bins:
        if p < thresh:
            return lvl
    return "XL"


def loo_axis(df: pd.DataFrame, axis_col: str, axis_label: str) -> dict:
    levels = sorted(df[axis_col].dropna().unique().tolist(), key=lambda x: float(x))
    rows = []
    all_abs_pct = []
    for lvl in levels:
        train = df[df[axis_col] != lvl]
        test = df[df[axis_col] == lvl]
        if len(train) < 8 or len(test) == 0:
            continue
        fit = fit_joint_additive_only(train, verbose=False)
        if fit is None:
            continue
        _, NL_t, D_t, NA_t, T_t, L_t = _extract_joint_fit_arrays(test)
        y_pred = _predict_joint_fit_result(fit, NL_t, D_t, NA_t, T_t)
        mape = _compute_mape(L_t, y_pred)
        abs_pct = np.abs((np.array(L_t) - y_pred) / np.array(L_t)) * 100
        all_abs_pct.extend(abs_pct.tolist())
        rows.append({
            "level": str(lvl),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "mape": float(mape),
        })
    overall_mape = float(np.mean(all_abs_pct)) if all_abs_pct else None
    return {"axis": axis_label, "rows": rows, "overall_mape": overall_mape}


def main():
    df = build_pool()
    print(f"Pool size N={len(df)}")
    print(f"Columns of interest: llm_size, seen_pairs, num_queries, adapter_params")

    # Add a derived adapter_level column from adapter_params (in case it's missing)
    if "adapter_level" not in df.columns or df["adapter_level"].isna().any():
        df["adapter_level"] = df["adapter_params"].apply(adapter_level_from_params)

    print("\nDistribution by axis:")
    for col, label in [
        ("llm_size", "N_L"),
        ("seen_pairs", "D"),
        ("num_queries", "T"),
        ("adapter_level", "width"),
    ]:
        print(f"  {label}: {sorted(df[col].dropna().unique().tolist())}")

    results = {}
    for col, label in [
        ("llm_size", "Leave-one-N_L-out"),
        ("seen_pairs", "Leave-one-D-out"),
        ("num_queries", "Leave-one-T-out"),
        ("adapter_level", "Leave-one-width-out"),
    ]:
        # llm_size is string; the others are numeric. Patch the sort key.
        levels_raw = df[col].dropna().unique().tolist()
        try:
            levels = sorted(levels_raw, key=lambda x: float(str(x).rstrip("BKMG")))
        except Exception:
            levels = sorted(levels_raw, key=str)
        rows = []
        all_abs_pct = []
        for lvl in levels:
            train = df[df[col] != lvl]
            test = df[df[col] == lvl]
            if len(train) < 8 or len(test) == 0:
                continue
            fit = fit_joint_additive_only(train, verbose=False)
            if fit is None:
                continue
            _, NL_t, D_t, NA_t, T_t, L_t = _extract_joint_fit_arrays(test)
            y_pred = _predict_joint_fit_result(fit, NL_t, D_t, NA_t, T_t)
            mape = _compute_mape(L_t, y_pred)
            abs_pct = np.abs((np.array(L_t) - y_pred) / np.array(L_t)) * 100
            all_abs_pct.extend(abs_pct.tolist())
            rows.append({
                "level": str(lvl),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "mape": float(mape),
            })
        overall = float(np.mean(all_abs_pct)) if all_abs_pct else None
        results[label] = {"per_level": rows, "overall_mape_pct": overall}

        print(f"\n=== {label} ===")
        for r in rows:
            print(f"  {r['level']:<12} n_train={r['n_train']:>3} n_test={r['n_test']:>3} MAPE={r['mape']:.2f}%")
        if overall is not None:
            print(f"  Pooled MAPE across all held-out points: {overall:.2f}%")

    out = ROOT / "analysis" / "results_final_d50m" / "structured_cv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
