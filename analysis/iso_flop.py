"""
Iso-FLOP analysis: compute-optimal adapter design.

Given a compute budget C, find the optimal (T, N_A) that minimizes predicted loss.
Uses the joint scaling law fitted from scaling_fit.py.

Produces:
- Iso-FLOP curves (like Chinchilla Fig.3)
- Optimal (T, N_A) allocation as a function of compute budget
- "How to size your adapter" practical guidelines

Usage:
    python analysis/iso_flop.py
    python analysis/iso_flop.py --results_json analysis/results/scaling_fit_results.json
"""

import argparse
import json
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────
# FLOP computation
# ─────────────────────────────────────────────────────────

# Adapter architecture constants
ADAPTER_CONFIGS = {
    "XS": {"d_model": 256, "n_heads": 4},
    "S":  {"d_model": 512, "n_heads": 8},
    "M":  {"d_model": 768, "n_heads": 8},
    "L":  {"d_model": 1024, "n_heads": 8},
    "XL": {"d_model": 1536, "n_heads": 16},
}

# Qwen2.5 LLM sizes
LLM_PARAMS = {
    "0.5B": 5e8, "1.5B": 1.5e9, "3B": 3e9,
    "7B": 7e9, "14B": 1.4e10, "32B": 3.2e10,
}

LLM_D_LLM = {
    "0.5B": 896, "1.5B": 1536, "3B": 2048,
    "7B": 3584, "14B": 5120, "32B": 5120,
}

D_VISION = 1152  # SigLIP


def _nlm_key(N_L: float) -> str:
    """Convert N_L float to LLM_PARAMS/LLM_D_LLM key string."""
    for key, val in LLM_PARAMS.items():
        if abs(val - N_L) / val < 0.01:
            return key
    # Fallback: try formatting
    gb = N_L / 1e9
    if gb == int(gb):
        return f"{int(gb)}B"
    return f"{gb}B"
T0 = 729         # (384/14)^2 for SigLIP-384


def compute_adapter_params(d_model: int, d_llm: int, num_queries: int = 64, num_layers: int = 2, ff_mult: int = 4) -> int:
    """Compute adapter parameter count (Perceiver Resampler)."""
    d_ff = d_model * ff_mult
    # Input projection: d_vision -> d_model (no bias)
    input_proj = D_VISION * d_model
    # Learnable queries: (1, num_queries, d_model)
    queries = num_queries * d_model
    # Per layer: cross_attn (4 × d_model²) + self_attn (4 × d_model²) + FFN (3 × d_model × d_ff)
    per_layer = 4 * d_model * d_model + 4 * d_model * d_model + 3 * d_model * d_ff
    # LayerNorms: 4 per layer × 2 × d_model (weight + bias)
    # (cross_attn_norm_q, cross_attn_norm_kv, self_attn_norm, ffn_norm)
    per_layer_norm = 4 * 2 * d_model
    # Output projection: d_model -> d_llm (no bias)
    output_proj = d_model * d_llm
    # Norms: input_norm + output_norm = 2 × 2 × d_model
    norms = 2 * 2 * d_model

    total = input_proj + queries + num_layers * (per_layer + per_layer_norm) + output_proj + norms
    return total


def compute_training_flops(
    N_A: int,
    N_L: float,
    T: int,
    D: int,
    batch_size: int = 32,
    seq_len: int = 150,  # average text sequence length
    num_layers: int = 2,
) -> float:
    """
    Estimate total training FLOPs.

    For adapter-only training:
    - Adapter: forward + backward ≈ 3 × 2 × N_A × tokens_per_sample
    - LLM: forward only (frozen) ≈ 2 × N_L × (T + seq_len)
    - Vision encoder: forward only ≈ 2 × N_VE × T₀ (constant, ignored)

    Per sample FLOPs:
        F_adapter = 6 × N_A × (T + T₀)  [forward: 2×, backward: 4×]
        F_llm_fwd = 2 × N_L × (T + seq_len)  [frozen, forward only]
        F_total_per_sample = F_adapter + F_llm_fwd

    Total FLOPs = F_total_per_sample × D
    """
    # Adapter FLOPs (trainable: fwd + bwd = 3× fwd = 6× params)
    adapter_tokens = T + T0  # cross-attention to T₀ vision tokens + self-attention among T queries
    f_adapter = 6 * N_A * adapter_tokens

    # LLM forward FLOPs (frozen, no backward)
    llm_seq = T + seq_len
    f_llm = 2 * N_L * llm_seq

    # Total per sample
    f_per_sample = f_adapter + f_llm

    # Total
    return f_per_sample * D


def compute_adapter_flops_only(N_A: int, T: int, D: int) -> float:
    """
    Compute FLOP contribution from adapter only (excludes frozen LLM/VE).
    This is the compute that's directly controllable by adapter design.

    F_adapter = 6 × N_A × (T + T₀) × D
    """
    return 6 * N_A * (T + T0) * D


# ─────────────────────────────────────────────────────────
# Joint law prediction
# ─────────────────────────────────────────────────────────

def predict_loss_multiplicative(params: dict, N_L: float, D: float, T: float, N_A: float) -> float:
    """Predict loss using multiplicative joint law."""
    a = params["a"]
    alpha = params["alpha"]
    b = params["b"]
    beta = params["beta"]
    c = params["c"]
    gamma = params["gamma"]
    h = params["h"]
    k = params.get("k", 0)
    e = params["e"]
    f = params["f"]
    eps = params["eps"]

    t_term = e * T + f / T
    n_term = 1.0 / N_L**alpha + k / N_A**gamma + h * N_A / N_L
    base = a / N_L**alpha + b / D**beta + c / N_A**gamma

    return t_term * n_term + base + eps


def predict_loss_additive(params: dict, N_L: float, D: float, T: float, N_A: float) -> float:
    """Predict loss using additive joint law."""
    a = params["a"]
    alpha = params["alpha"]
    b = params["b"]
    beta = params["beta"]
    c = params["c"]
    gamma = params["gamma"]
    h = params["h"]
    e = params["e"]
    f = params["f"]
    eps = params["eps"]

    return a / N_L**alpha + b / D**beta + c / N_A**gamma + h * N_A / N_L + e * T + f / T + eps


# ─────────────────────────────────────────────────────────
# Iso-FLOP optimization
# ─────────────────────────────────────────────────────────

def find_optimal_adapter(
    predict_fn,
    params: dict,
    N_L: float,
    D: float,
    flop_budget: float,
    T_range: np.ndarray = None,
    d_model_range: np.ndarray = None,
) -> dict:
    """
    Find optimal (T, N_A) for a given FLOP budget.

    Grid search over T and d_model, keeping only configs within FLOP budget.
    """
    if T_range is None:
        T_range = np.array([4, 8, 16, 32, 64, 128, 256])
    if d_model_range is None:
        d_model_range = np.array([128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536])

    d_llm = LLM_D_LLM.get(_nlm_key(N_L), 2048)

    best = {"loss": float("inf")}

    for T in T_range:
        for d_model in d_model_range:
            N_A = compute_adapter_params(int(d_model), d_llm, num_queries=int(T))
            flops = compute_training_flops(N_A, N_L, int(T), int(D))

            if flops > flop_budget:
                continue

            loss = predict_fn(params, N_L, D, float(T), float(N_A))

            if loss < best["loss"]:
                best = {
                    "loss": loss,
                    "T": int(T),
                    "d_model": int(d_model),
                    "N_A": N_A,
                    "flops": flops,
                    "utilization": flops / flop_budget,
                }

    return best


def compute_iso_flop_curves(
    predict_fn,
    params: dict,
    N_L: float = 3e9,
    D: float = 552544,
    n_budgets: int = 20,
) -> list[dict]:
    """Compute optimal (T, N_A) for a range of FLOP budgets."""

    # Compute FLOP range from smallest to largest adapter config
    d_llm = LLM_D_LLM.get(_nlm_key(N_L), 2048)

    flop_min = compute_training_flops(
        compute_adapter_params(128, d_llm, num_queries=4), N_L, 4, int(D)
    )
    flop_max = compute_training_flops(
        compute_adapter_params(1536, d_llm, num_queries=256), N_L, 256, int(D)
    )

    flop_budgets = np.logspace(np.log10(flop_min), np.log10(flop_max), n_budgets)

    results = []
    for budget in flop_budgets:
        opt = find_optimal_adapter(predict_fn, params, N_L, D, budget)
        if opt["loss"] < float("inf"):
            opt["budget"] = budget
            opt["N_L"] = N_L
            opt["D"] = D
            results.append(opt)

    return results


def compute_iso_flop_grid(
    predict_fn,
    params: dict,
    N_L: float = 3e9,
    D: float = 552544,
) -> list[dict]:
    """
    Compute predicted loss for all (T, d_model) combinations.
    For plotting iso-FLOP contours.
    """
    T_range = [4, 8, 16, 32, 64, 128, 256]
    d_model_range = [256, 512, 768, 1024, 1536]
    d_llm = LLM_D_LLM.get(_nlm_key(N_L), 2048)

    grid = []
    for T in T_range:
        for d_model in d_model_range:
            N_A = compute_adapter_params(d_model, d_llm, num_queries=T)
            flops = compute_training_flops(N_A, N_L, T, int(D))
            loss = predict_fn(params, N_L, D, float(T), float(N_A))

            grid.append({
                "T": T,
                "d_model": d_model,
                "N_A": N_A,
                "flops": flops,
                "predicted_loss": loss,
            })

    return grid


# ─────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Iso-FLOP analysis")
    parser.add_argument("--results_json", type=str,
                        default="analysis/results/scaling_fit_results.json")
    parser.add_argument("--output_dir", type=str,
                        default="analysis/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load joint law parameters
    results_path = Path(args.results_json)
    if results_path.exists():
        with open(results_path) as f:
            fit_results = json.load(f)
    else:
        print(f"Warning: {results_path} not found. Using placeholder parameters.")
        fit_results = {}

    # Extract joint law parameters from scaling_fit_results.json
    joint = fit_results.get("joint", {})
    winner = joint.get("winner", "")
    joint_params = None

    # Try to get params from the fits list (best fit is first)
    fits = joint.get("fits", [])
    if fits and isinstance(fits[0], dict):
        joint_params = fits[0].get("params", None)
        if not winner:
            winner = fits[0].get("name", "")

    if joint_params is None:
        print("No joint law parameters found. Run scaling_fit.py first.")
        print("Using additive placeholder for demonstration.")
        joint_params = {
            "a": 10.0, "alpha": 0.1, "b": 1.0, "beta": 0.1,
            "c": 100.0, "gamma": 0.15, "h": 0.01,
            "e": 0.001, "f": 5.0, "eps": 2.0,
        }
        winner = "additive"

    predict_fn = predict_loss_multiplicative if "multiplicative" in winner else predict_loss_additive

    print(f"Joint law type: {winner}")
    print(f"Parameters: {json.dumps(joint_params, indent=2)}")

    # ── 1. Iso-FLOP optimal curves ──
    print("\n" + "=" * 60)
    print("ISO-FLOP OPTIMAL CURVES")
    print("=" * 60)

    for llm_size in ["3B"]:
        N_L = LLM_PARAMS[llm_size]
        D = 552544

        print(f"\n--- {llm_size}, D={D:,} ---")
        curves = compute_iso_flop_curves(predict_fn, joint_params, N_L, D)

        print(f"{'Budget (TFLOP)':<16} {'T':>4} {'d_model':>8} {'N_A':>12} {'Loss':>8} {'Util':>6}")
        print("-" * 60)
        for c in curves:
            tflop = c["budget"] / 1e12
            print(f"{tflop:>14.2f}   {c['T']:>4} {c['d_model']:>8} "
                  f"{c['N_A']:>12,} {c['loss']:>8.4f} {c['utilization']:>5.1%}")

    # ── 2. Grid for contour plotting ──
    print("\n" + "=" * 60)
    print("LOSS GRID (T × d_model)")
    print("=" * 60)

    N_L = LLM_PARAMS["3B"]
    D = 552544
    grid = compute_iso_flop_grid(predict_fn, joint_params, N_L, D)

    # Print as table
    T_vals = sorted(set(g["T"] for g in grid))
    dm_vals = sorted(set(g["d_model"] for g in grid))

    header = f"{'T \\\\ d_model':>12}" + "".join(f"{dm:>10}" for dm in dm_vals)
    print(header)
    print("-" * len(header))
    for T in T_vals:
        row = f"{T:>12}"
        for dm in dm_vals:
            entry = [g for g in grid if g["T"] == T and g["d_model"] == dm][0]
            row += f"{entry['predicted_loss']:>10.4f}"
        print(row)

    # ── 3. FLOP breakdown ──
    print("\n" + "=" * 60)
    print("FLOP BREAKDOWN (per sample)")
    print("=" * 60)

    d_llm = LLM_D_LLM["3B"]
    for level in ["XS", "S", "M", "L", "XL"]:
        d_model = ADAPTER_CONFIGS[level]["d_model"]
        for T in [16, 64, 128]:
            N_A = compute_adapter_params(d_model, d_llm, num_queries=T)
            f_adapter = 6 * N_A * (T + T0)
            f_llm = 2 * N_L * (T + 150)
            f_total = f_adapter + f_llm
            ratio = f_adapter / f_total * 100
            print(f"  {level} T={T:>3}: adapter={f_adapter/1e9:.2f}G, "
                  f"LLM={f_llm/1e9:.2f}G, total={f_total/1e9:.2f}G "
                  f"(adapter={ratio:.1f}%)")

    # ── 4. Save results ──
    iso_flop_results = {
        "joint_law_type": winner,
        "params": joint_params,
        "iso_flop_curves": {
            "3B": compute_iso_flop_curves(predict_fn, joint_params, LLM_PARAMS["3B"], D),
        },
        "grid_3B": grid,
    }

    # Add curves for other LLM sizes if params support it
    for llm_size in ["0.5B", "7B", "14B"]:
        try:
            curves = compute_iso_flop_curves(
                predict_fn, joint_params, LLM_PARAMS[llm_size], D
            )
            iso_flop_results["iso_flop_curves"][llm_size] = curves
        except Exception:
            pass

    output_path = output_dir / "iso_flop_results.json"
    with open(output_path, "w") as f:
        json.dump(iso_flop_results, f, indent=2, default=lambda x: float(x))
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
