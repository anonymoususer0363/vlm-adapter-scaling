"""
Scaling law fitting and golden rule extraction.

Given experiment results (N_L, D, T, N_A, val_loss), this module:
1. Fits marginal effects (power law, hook, log-hook) with BIC selection
2. Fits joint scaling law (additive vs multiplicative)
3. Extracts golden rules: T_opt, r_A_opt, ρ_opt
4. Generates all paper figures
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize


# ============================================================
# 1. Candidate functional forms
# ============================================================

def power_law_2var(X, a, alpha, b, beta, eps):
    """L(N, D) = a/N^alpha + b/D^beta + eps"""
    N, D = X
    return a / np.power(N, alpha) + b / np.power(D, beta) + eps


# --- N_A marginal candidates ---

def na_power(N_A, c, gamma, iota):
    """(a) Monotone decrease: L(N_A) = c/N_A^gamma + iota"""
    return c / np.power(N_A, gamma) + iota


def na_hook(X, c, gamma, h, iota):
    """(b) Hook: L(N_A) = c/N_A^gamma + h*(N_A/N_L) + iota"""
    N_A, N_L = X
    return c / np.power(N_A, gamma) + h * (N_A / N_L) + iota


def na_log_hook(X, c, h, iota):
    """(c) Log-hook: L(N_A) = -c*log(N_A) + h*(N_A/N_L) + iota"""
    N_A, N_L = X
    return -c * np.log(N_A) + h * (N_A / N_L) + iota


# --- T marginal candidates ---

def t_power(T, f, delta, tau):
    """(a) Monotone decrease: L(T) = f/T^delta + tau"""
    return f / np.power(T, delta) + tau


def t_hook(T, e, f, tau):
    """(b) Hook: L(T) = e*T + f/T + tau → T_opt = sqrt(f/e)"""
    return e * T + f / T + tau


def t_log_hook(T, e, f, tau):
    """(c) Log-hook: L(T) = e*T - f*log(T) + tau → T_opt = f/e"""
    return e * T - f * np.log(T) + tau


def _t_marginal_candidate_defs() -> dict:
    """Canonical candidate set for T marginal fits."""
    return {
        "t_power": (t_power, [1.0, 0.5, 2.0], ([0, 0, 0], [np.inf, 5, np.inf])),
        "t_hook": (t_hook, [0.001, 10.0, 2.0], ([0, 0, 0], [np.inf, np.inf, np.inf])),
        "t_log_hook": (t_log_hook, [0.001, 0.1, 2.0], ([0, 0, 0], [np.inf, np.inf, np.inf])),
    }


# --- Joint law candidates ---

def joint_simple_nl(X, a, alpha, eps):
    """3-param: L = a/N_L^α + ε (base model only, no D/T/N_A)"""
    N_L, D, N_A, T = X
    return a / np.power(N_L, alpha) + eps


def joint_nl_d(X, a, alpha, b, beta, eps):
    """5-param: L = a/N_L^α + b/D^β + ε"""
    N_L, D, N_A, T = X
    return a / np.power(N_L, alpha) + b / np.power(D, beta) + eps


def joint_nl_na(X, a, alpha, c, gamma, eps):
    """5-param: L = a/N_L^α + c/N_A^γ + ε"""
    N_L, D, N_A, T = X
    return a / np.power(N_L, alpha) + c / np.power(N_A, gamma) + eps


def joint_nl_na_t(X, a, alpha, c, gamma, e, f, eps):
    """7-param: L = a/N_L^α + c/N_A^γ + eT + f/T + ε"""
    N_L, D, N_A, T = X
    return a / np.power(N_L, alpha) + c / np.power(N_A, gamma) + e * T + f / T + eps


def joint_additive(X, a, alpha, b, beta, c, gamma, h, e, f, eps):
    """Additive: L = a/N^α + b/D^β + c/N_A^γ + h*N_A/N + eT + f/T + ε"""
    N_L, D, N_A, T = X
    return (a / np.power(N_L, alpha) + b / np.power(D, beta)
            + c / np.power(N_A, gamma) + h * (N_A / N_L)
            + e * T + f / T + eps)


def joint_multiplicative(X, a, alpha, b, beta, c, gamma, h, k, e, f, eps):
    """Multiplicative: L = (eT + f/T)*(1/N^α + k/N_A^γ + h*N_A/N) + a/N^α + b/D^β + c/N_A^γ + ε"""
    N_L, D, N_A, T = X
    structural = e * T + f / T
    scale = 1 / np.power(N_L, alpha) + k / np.power(N_A, gamma) + h * (N_A / N_L)
    base = a / np.power(N_L, alpha) + b / np.power(D, beta) + c / np.power(N_A, gamma)
    return structural * scale + base + eps


# --- Per-LLM fixed-effects joint law ---

def fit_joint_per_llm_eps(
    N_L: np.ndarray,
    D: np.ndarray,
    N_A: np.ndarray,
    T: np.ndarray,
    L: np.ndarray,
    llm_names: np.ndarray,
) -> dict:
    """Joint additive with per-LLM epsilon (fixed effects).

    L = a/N_L^α + b/D^β + c/N_A^γ + h*(N_A/N_L) + eT + f/T + ε_i

    Instead of a single shared epsilon, each LLM size gets its own offset.
    This captures model-specific irreducible loss not explained by the N_L power law.
    """
    unique_llms = sorted(set(llm_names))
    n_llms = len(unique_llms)
    llm_idx = np.array([unique_llms.index(l) for l in llm_names])
    N = len(L)

    def objective(params):
        a, alpha, b, beta, c, gamma, h, e, f = params[:9]
        eps = params[9:9 + n_llms]
        pred = (a / np.power(N_L, alpha) + b / np.power(D, beta)
                + c / np.power(N_A, gamma) + h * (N_A / N_L)
                + e * T + f / T + eps[llm_idx])
        return np.sum((L - pred) ** 2)

    x0 = [4000, 0.45, 25, 0.26, 12, 0.1, 1e-7, 3e-5, 0.05] + [0.0] * n_llms
    bounds = (
        [(0.01, 1e18), (0.01, 5), (0.01, 1e18), (0.01, 5),
         (0.01, 1e18), (0.01, 5), (0, 1e3), (0, 1e3), (0, 1e3)]
        + [(-2, 2)] * n_llms
    )
    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 50000})

    a, alpha, b, beta, c, gamma, h, e, f = res.x[:9]
    eps_values = res.x[9:9 + n_llms]
    y_pred = (a / np.power(N_L, alpha) + b / np.power(D, beta)
              + c / np.power(N_A, gamma) + h * (N_A / N_L)
              + e * T + f / T + eps_values[llm_idx])

    ss_res = np.sum((L - y_pred) ** 2)
    ss_tot = np.sum((L - np.mean(L)) ** 2)
    k = 9 + n_llms
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mape = np.mean(np.abs((L - y_pred) / L)) * 100
    rmse = np.sqrt(ss_res / N)
    bic = compute_bic(N, k, ss_res)
    aic = compute_aic(N, k, ss_res)

    return {
        "shared_params": {
            "a": a, "alpha": alpha, "b": b, "beta": beta,
            "c": c, "gamma": gamma, "h": h, "e": e, "f": f,
        },
        "per_llm_eps": {llm: float(eps_values[i]) for i, llm in enumerate(unique_llms)},
        "r_squared": r_squared,
        "mape": mape,
        "rmse": rmse,
        "bic": bic,
        "aic": aic,
        "k": k,
        "n": N,
        "y_pred": y_pred,
        "y_true": L,
    }


# ============================================================
# 2. Fitting utilities
# ============================================================

def compute_bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion. Lower is better."""
    if rss <= 0 or n <= k:
        return float("inf")
    return n * np.log(rss / n) + k * np.log(n)


def compute_aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion. Lower is better."""
    if rss <= 0 or n <= k:
        return float("inf")
    return n * np.log(rss / n) + 2 * k


@dataclass
class FitResult:
    name: str
    params: dict
    residuals: np.ndarray
    bic: float
    aic: float
    r_squared: float
    rmse: float


def fit_and_compare(
    x_data,
    y_data,
    candidates: dict,  # name -> (func, p0, bounds)
    verbose: bool = True,
) -> list[FitResult]:
    """Fit multiple candidates and rank by BIC."""
    n = len(y_data)
    ss_total = np.sum((y_data - np.mean(y_data)) ** 2)
    results = []

    for name, (func, p0, bounds) in candidates.items():
        try:
            popt, pcov = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
            y_pred = func(x_data, *popt)
            residuals = y_data - y_pred
            rss = np.sum(residuals ** 2)
            k = len(popt)
            bic = compute_bic(n, k, rss)
            aic = compute_aic(n, k, rss)
            r2 = 1 - rss / ss_total if ss_total > 0 else 0
            rmse = np.sqrt(rss / n)

            # Get parameter names from function
            import inspect
            param_names = list(inspect.signature(func).parameters.keys())[1:]  # skip X
            params = dict(zip(param_names, popt))

            result = FitResult(name=name, params=params, residuals=residuals,
                               bic=bic, aic=aic, r_squared=r2, rmse=rmse)
            results.append(result)

            if verbose:
                print(f"  {name}: BIC={bic:.2f}, AIC={aic:.2f}, R²={r2:.6f}, RMSE={rmse:.4f}")

        except Exception as ex:
            if verbose:
                print(f"  {name}: FAILED ({ex})")

    results.sort(key=lambda r: r.bic)
    return results


# ============================================================
# 3. Golden rule extraction
# ============================================================

def extract_t_opt(fit_result: FitResult, t_range: tuple[float, float] = (1, 512)) -> float | None:
    """Extract T_opt from T marginal fit.

    Returns None if the optimum falls outside t_range, which indicates
    the fit did not identify a finite optimum within the tested regime
    (e.g., when T sensitivity is too weak to constrain the hook shape).
    """
    t_opt = None
    if fit_result.name == "t_hook":
        e = fit_result.params["e"]
        f = fit_result.params["f"]
        if e > 0 and f > 0:
            t_opt = np.sqrt(f / e)
    elif fit_result.name == "t_log_hook":
        e = fit_result.params["e"]
        f = fit_result.params["f"]
        if e > 0 and f > 0:
            t_opt = f / e

    if t_opt is not None and not (t_range[0] <= t_opt <= t_range[1]):
        return None
    return t_opt


def extract_ra_opt(fit_result: FitResult, N_L: float = None) -> dict | None:
    """Extract optimal adapter ratio from N_A marginal fit.

    For na_hook: ∂L/∂N_A = 0 → N_A_opt = (c*gamma/h)^(1/(gamma+1))
    For na_log_hook: ∂L/∂N_A = 0 → N_A_opt = c*N_L/h
    """
    if fit_result.name == "na_hook":
        c = fit_result.params["c"]
        gamma = fit_result.params["gamma"]
        h = fit_result.params["h"]
        if h > 0 and c > 0 and gamma > 0:
            N_A_opt = (c * gamma / h) ** (1 / (gamma + 1))
            r_A_opt = N_A_opt / N_L if N_L else None
            return {"N_A_opt": N_A_opt, "r_A_opt": r_A_opt, "c": c, "gamma": gamma, "h": h}
    elif fit_result.name == "na_log_hook":
        c = fit_result.params["c"]
        h = fit_result.params["h"]
        if h > 0 and c > 0 and N_L:
            N_A_opt = c * N_L / h
            r_A_opt = c / h
            return {"N_A_opt": N_A_opt, "r_A_opt": r_A_opt, "c": c, "h": h}
    return None


# ============================================================
# 4. Main analysis pipeline
# ============================================================

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    records = []
    for result_file in Path(results_dir).rglob("result.json"):
        with open(result_file) as f:
            result = json.load(f)

        config_file = result_file.parent / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            result.update(config)

        records.append(result)

    df = pd.DataFrame(records)
    return df


def analyze_marginal_t(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Analyze T marginal effect for each N_L."""
    results = {}

    for n_l in df["llm_name"].unique():
        subset = df[df["llm_name"] == n_l].sort_values("num_queries")
        T = subset["num_queries"].values.astype(float)
        L = subset["best_val_loss"].values

        if verbose:
            print(f"\n--- T marginal for {n_l} ---")

        fits = fit_and_compare(T, L, _t_marginal_candidate_defs(), verbose=verbose)
        if fits:
            best = fits[0]
            t_opt = extract_t_opt(best, t_range=(T.min(), T.max()))
            results[n_l] = {"best_fit": best, "all_fits": fits, "T_opt": t_opt}
            if verbose:
                if t_opt is not None:
                    print(f"  → T_opt = {t_opt:.1f}")
                else:
                    print(f"  → T_opt: not identified (weak T sensitivity)")

    return results


def analyze_marginal_na(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Analyze N_A marginal effect for each N_L."""
    results = {}

    # Support both field names
    na_col = "adapter_params" if "adapter_params" in df.columns else "adapter_total"

    for n_l in df["llm_name"].unique():
        subset = df[df["llm_name"] == n_l].sort_values(na_col)
        N_A = subset[na_col].values.astype(float)
        L = subset["best_val_loss"].values

        if verbose:
            print(f"\n--- N_A marginal for {n_l} ---")

        # Parse N_L value from name
        n_l_val = float(n_l.split("-")[-1].replace("B", "e9").replace("M", "e6"))
        N_L_arr = np.full_like(N_A, n_l_val)

        # Fit power law with just N_A
        fits_power = fit_and_compare(
            N_A, L,
            {"na_power": (na_power, [1e6, 0.3, 2.0], ([0, 0, 0], [np.inf, 5, np.inf]))},
            verbose=verbose,
        )

        # Fit hook and log_hook with (N_A, N_L)
        fits_hook = fit_and_compare(
            (N_A, N_L_arr), L,
            {
                "na_hook": (na_hook, [1e6, 0.3, 1.0, 2.0],
                            ([0, 0, 0, 0], [np.inf, 5, np.inf, np.inf])),
                "na_log_hook": (na_log_hook, [0.01, 1e-9, 5.0],
                                ([0, 0, -np.inf], [np.inf, np.inf, np.inf])),
            },
            verbose=verbose,
        )

        all_fits = fits_power + fits_hook
        all_fits.sort(key=lambda r: r.bic)

        if all_fits:
            best = all_fits[0]
            ra_opt = extract_ra_opt(best, N_L=n_l_val)
            results[n_l] = {"best_fit": best, "all_fits": all_fits,
                            "N_L": n_l_val, "ra_opt": ra_opt}

    return results


def analyze_base_power_law(df: pd.DataFrame, verbose: bool = True) -> FitResult | None:
    """Fit L(N_L, D) = a/N_L^alpha + b/D^beta + eps on G0v2 data."""
    if verbose:
        print("\n--- Base Power Law L(N_L, D) ---")

    df = _parse_llm_params_col(df)
    N_L = df["llm_params"].values.astype(float)
    D = _compute_effective_d(df)
    L = df["best_val_loss"].values.astype(float)

    candidates = {
        "power_law_2var": (
            power_law_2var,
            [1e3, 0.1, 1e3, 0.1, 2.5],
            ([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.inf]),
        ),
    }

    fits = fit_and_compare((N_L, D), L, candidates, verbose=verbose)
    return fits[0] if fits else None


def analyze_interaction(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Test additive vs multiplicative T×N_A interaction on G3 data."""
    if verbose:
        print("\n--- T × N_A Interaction Test ---")

    na_col = "adapter_params" if "adapter_params" in df.columns else "adapter_total"
    T = df["num_queries"].values.astype(float)
    N_A = df[na_col].values.astype(float)
    L = df["best_val_loss"].values.astype(float)

    # Simple additive: L = eT + f/T + c/N_A^gamma + iota
    def additive_TxNA(X, e, f, c, gamma, iota):
        T, N_A = X
        return e * T + f / T + c / np.power(N_A, gamma) + iota

    # Simple multiplicative: L = (eT + f/T) * c/N_A^gamma + iota
    def multiplicative_TxNA(X, e, f, c, gamma, iota):
        T, N_A = X
        return (e * T + f / T) * c / np.power(N_A, gamma) + iota

    candidates = {
        "additive": (additive_TxNA,
                     [0.001, 1.0, 1e4, 0.3, 2.5],
                     ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, 5, np.inf])),
        "multiplicative": (multiplicative_TxNA,
                           [0.0001, 0.1, 1e4, 0.3, 2.5],
                           ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, 5, np.inf])),
    }

    fits = fit_and_compare((T, N_A), L, candidates, verbose=verbose)
    winner = fits[0].name if fits else None

    if verbose and fits:
        print(f"  >>> Winner: {winner} (ΔBIC = {fits[1].bic - fits[0].bic:.2f})" if len(fits) > 1 else "")

    return {"fits": fits, "winner": winner}


def _compute_effective_d(df: pd.DataFrame, epoch_exponent: float = 1.0) -> np.ndarray:
    """Compute effective D from config fields with optional multi-epoch correction.

    D_eff = D_unique × epochs^η   where η = epoch_exponent
        η=1.0: D_eff = seen_pairs (no correction, original behavior)
        η=0.0: D_eff = D_unique (ignore repetitions entirely)
        0<η<1: diminishing returns from repeated data

    D < 552K → num_samples (subsample, always 1 epoch)
    D = 552K → 1 epoch × 552,544
    D > 552K → D_unique × epochs^η
    """
    dataset_size = 552544
    D_unique = np.full(len(df), float(dataset_size))
    epochs = np.ones(len(df))

    num_samples = df["num_samples"] if "num_samples" in df.columns else pd.Series(np.nan, index=df.index)
    num_epochs = df["num_epochs"] if "num_epochs" in df.columns else pd.Series(1, index=df.index)

    ns_vals = num_samples.fillna(dataset_size).values.astype(float)
    ne_vals = num_epochs.fillna(1).values.astype(float)

    for i in range(len(D_unique)):
        ns = ns_vals[i]
        ne = ne_vals[i]

        if ns < dataset_size:
            # Subsample: always 1 epoch of fewer samples
            D_unique[i] = ns
            epochs[i] = 1.0
        elif ne > 1:
            # Multi-epoch: D_unique stays at dataset_size, epochs > 1
            D_unique[i] = dataset_size
            epochs[i] = ne
        else:
            D_unique[i] = dataset_size
            epochs[i] = 1.0

    return D_unique * np.power(epochs, epoch_exponent)



    
def _parse_llm_params_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has numeric llm_params column."""
    if "llm_params" in df.columns and df["llm_params"].notna().any():
        return df

    def parse_params(name):
        s = name.split("-")[-1]
        if s.endswith("B"):
            return float(s[:-1]) * 1e9
        elif s.endswith("M"):
            return float(s[:-1]) * 1e6
        return float(s)

    df = df.copy()
    df["llm_params"] = df["llm_name"].apply(parse_params)
    return df


def _ensure_group_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a group column parsed from run_name when needed."""
    if "group" in df.columns and df["group"].notna().any():
        return df

    def parse_group(run_name: str) -> str:
        if not isinstance(run_name, str):
            return ""
        if run_name.startswith("rerun"):
            return "rerun"
        if run_name.startswith("g"):
            return run_name.split("_")[0]
        return ""

    df = df.copy()
    df["group"] = df["run_name"].apply(parse_group)
    return df


def _ensure_llm_size_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a coarse llm_size column such as 0.5B, 3B, 14B."""
    if "llm_size" in df.columns and df["llm_size"].notna().any():
        return df

    def parse_size(name: str) -> str:
        if not isinstance(name, str):
            return ""
        for size in ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]:
            if size in name:
                return size
        return ""

    df = df.copy()
    df["llm_size"] = df["llm_name"].apply(parse_size)
    return df


def _ensure_adapter_type_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has adapter_type with perceiver as the default."""
    df = df.copy()
    if "adapter_type" in df.columns:
        df["adapter_type"] = df["adapter_type"].fillna("perceiver")
    else:
        df["adapter_type"] = "perceiver"
    return df


def _ensure_depth_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has an effective adapter depth column."""
    if "effective_depth" in df.columns and df["effective_depth"].notna().any():
        return df

    df = df.copy()
    depth_series = None
    for col in ["adapter_num_layers", "vision_adapter_num_layers"]:
        if col in df.columns and df[col].notna().any():
            depth_series = pd.to_numeric(df[col], errors="coerce")
            break

    if depth_series is None:
        depth_series = pd.to_numeric(
            df["run_name"].astype(str).str.extract(r"depth(\d+)")[0],
            errors="coerce",
        )

    df["effective_depth"] = depth_series.fillna(2).astype(float)
    return df


def prepare_joint_fit_data(df: pd.DataFrame, fit_groups: list[str], verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Select the canonical joint-fit dataset based on group and architecture filters."""
    fit_groups = list(fit_groups)
    df = _ensure_group_col(df)
    df = _ensure_adapter_type_col(df)
    df = _ensure_depth_col(df)
    df = _ensure_llm_size_col(df)

    group_mask = df["group"].isin(fit_groups)
    perceiver_mask = df["adapter_type"].eq("perceiver")
    depth_mask = df["effective_depth"].eq(2)

    selected = df[group_mask & perceiver_mask & depth_mask].copy()
    summary = {
        "fit_groups": fit_groups,
        "n_input": int(len(df)),
        "n_group_match": int(group_mask.sum()),
        "n_selected": int(len(selected)),
        "group_counts": {str(k): int(v) for k, v in selected["group"].value_counts().sort_index().items()},
        "excluded_non_perceiver": sorted(df[group_mask & ~perceiver_mask]["run_name"].tolist()),
        "excluded_non_depth2": sorted(df[group_mask & perceiver_mask & ~depth_mask]["run_name"].tolist()),
    }

    if verbose:
        print("\n--- Joint Fit Dataset Selection ---")
        print(f"  Fit groups: {', '.join(fit_groups)}")
        print(f"  Selected {summary['n_selected']} / {summary['n_input']} experiments")
        if summary["group_counts"]:
            print("  Per-group counts:")
            for group_name, count in summary["group_counts"].items():
                print(f"    {group_name}: {count}")
        if summary["excluded_non_perceiver"]:
            print(f"  Excluded non-perceiver runs: {len(summary['excluded_non_perceiver'])}")
        if summary["excluded_non_depth2"]:
            print(f"  Excluded depth!=2 runs: {len(summary['excluded_non_depth2'])}")

    return selected, summary


def exclude_divergent_runs(
    df: pd.DataFrame,
    threshold: float = 0.5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Exclude runs whose final loss is substantially worse than best loss."""
    df = _ensure_group_col(df)
    df = _ensure_llm_size_col(df)
    df = df.copy()

    final_loss = pd.to_numeric(df.get("final_val_loss"), errors="coerce")
    best_loss = pd.to_numeric(df.get("best_val_loss"), errors="coerce")
    gap = final_loss - best_loss
    divergent_mask = gap > threshold

    excluded = df[divergent_mask].copy()
    excluded["divergence_gap"] = gap[divergent_mask]
    excluded = excluded.sort_values("divergence_gap", ascending=False)

    excluded_records = []
    for _, row in excluded.iterrows():
        excluded_records.append({
            "run_name": row.get("run_name"),
            "group": row.get("group"),
            "llm_name": row.get("llm_name"),
            "llm_size": row.get("llm_size"),
            "best_val_loss": float(row["best_val_loss"]) if pd.notna(row.get("best_val_loss")) else None,
            "final_val_loss": float(row["final_val_loss"]) if pd.notna(row.get("final_val_loss")) else None,
            "divergence_gap": float(row["divergence_gap"]),
        })

    if verbose and excluded_records:
        print(f"  Excluding {len(excluded_records)} divergent runs (final-best > {threshold:.3f}):")
        for record in excluded_records:
            print(f"    {record['run_name']}: gap={record['divergence_gap']:.4f}")

    return df[~divergent_mask].copy(), excluded_records


def _compute_mape(y_true, y_pred) -> float | None:
    """Compute mean absolute percentage error in percent."""
    if y_true is None or y_pred is None:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return None
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def _llm_size_sort_key(size: str) -> float:
    """Sort helper for coarse LLM size labels such as 1.5B, 7B, 14B."""
    if not isinstance(size, str):
        return float("inf")
    try:
        if size.endswith("B"):
            return float(size[:-1])
        if size.endswith("M"):
            return float(size[:-1]) / 1000.0
    except ValueError:
        return float("inf")
    return float("inf")


def _joint_additive_candidate_defs() -> dict:
    """Single-candidate additive joint law used for fixed-form validation."""
    return {
        "joint_additive": (
            joint_additive,
            [1e3, 0.1, 1e3, 0.1, 1e4, 0.3, 1e-9, 0.001, 1.0, 2.5],
            ([0] * 9 + [0], [np.inf, 2, np.inf, 2, np.inf, 5, np.inf, np.inf, np.inf, np.inf]),
        ),
    }


def _joint_candidate_defs(include_multiplicative: bool = True) -> dict:
    """Canonical joint-law candidate set used across analyses."""
    candidates = _joint_additive_candidate_defs()
    if include_multiplicative:
        candidates["joint_multiplicative"] = (
            joint_multiplicative,
            [1e3, 0.1, 1e3, 0.1, 1e4, 0.3, 1e-9, 1.0, 0.001, 1.0, 2.5],
            ([0] * 10 + [0], [np.inf, 2, np.inf, 2, np.inf, 5, np.inf, np.inf, np.inf, np.inf, np.inf]),
        )
    return candidates


def _extract_joint_fit_arrays(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the canonical joint-fit arrays from a dataframe."""
    na_col = "adapter_params" if "adapter_params" in df.columns else "adapter_total"
    df = _parse_llm_params_col(df)
    N_L = df["llm_params"].values.astype(float)
    D = _compute_effective_d(df)
    N_A = df[na_col].values.astype(float)
    T = df["num_queries"].values.astype(float)
    L = df["best_val_loss"].values.astype(float)
    return df, N_L, D, N_A, T, L


def _predict_joint_fit_result(
    fit_result: FitResult,
    N_L: np.ndarray,
    D: np.ndarray,
    N_A: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Predict losses from a fitted joint-law model."""
    if fit_result.name == "joint_additive":
        param_order = ["a", "alpha", "b", "beta", "c", "gamma", "h", "e", "f", "eps"]
        return joint_additive((N_L, D, N_A, T), *[fit_result.params[k] for k in param_order])
    if fit_result.name == "joint_multiplicative":
        param_order = ["a", "alpha", "b", "beta", "c", "gamma", "h", "k", "e", "f", "eps"]
        return joint_multiplicative((N_L, D, N_A, T), *[fit_result.params[k] for k in param_order])
    raise ValueError(f"Unsupported joint fit type: {fit_result.name}")


def fit_joint_additive_only(df: pd.DataFrame, verbose: bool = False) -> FitResult | None:
    """Fit the additive joint law only, used for bootstrap and hold-out validation."""
    _, N_L, D, N_A, T, L = _extract_joint_fit_arrays(df)
    fits = fit_and_compare(
        (N_L, D, N_A, T),
        L,
        _joint_additive_candidate_defs(),
        verbose=verbose,
    )
    return fits[0] if fits else None


def run_leave_one_scale_out(
    df: pd.DataFrame,
    holdout_sizes: list[str] | None = None,
    base_exclude_llm_sizes: list[str] | None = None,
    divergence_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Hold out one LLM size at a time and measure prediction error on the held-out scale."""
    df = _ensure_group_col(df)
    df = _ensure_llm_size_col(df)

    canonical_df, canonical_meta = prepare_joint_variant_data(
        df,
        exclude_llm_sizes=base_exclude_llm_sizes,
        divergence_threshold=divergence_threshold,
        auto_exclude_divergent=True,
        verbose=False,
    )
    if holdout_sizes is None:
        holdout_sizes = ["1.5B", "3B", "7B", "14B"]
    available_sizes = [
        size for size in holdout_sizes if size in set(canonical_df["llm_size"].dropna().tolist())
    ]

    rows = []
    if verbose:
        print("\n--- Leave-One-Scale-Out (LLM) ---")
        print(f"  Canonical pool after filtering: {len(canonical_df)} points")
        print(f"  {'Hold-out':<8} {'n_train':>7} {'n_test':>6} {'Model':<20} {'MAPE':>8} {'MAE':>8}")
        print("  " + "-" * 68)

    for holdout_size in available_sizes:
        heldout_df = canonical_df[canonical_df["llm_size"] == holdout_size].copy()
        train_df = canonical_df[canonical_df["llm_size"] != holdout_size].copy()
        if len(heldout_df) == 0 or len(train_df) <= 10:
            continue

        best_fit = fit_joint_additive_only(train_df, verbose=False)
        if best_fit is None:
            continue
        heldout_df, N_L_test, D_test, N_A_test, T_test, L_test = _extract_joint_fit_arrays(heldout_df)
        y_pred = _predict_joint_fit_result(best_fit, N_L_test, D_test, N_A_test, T_test)
        mape = _compute_mape(L_test, y_pred)
        mae = float(np.mean(np.abs(L_test - y_pred)))
        row = {
            "holdout_llm_size": holdout_size,
            "n_train": int(len(train_df)),
            "n_test": int(len(heldout_df)),
            "model": best_fit.name,
            "train_r_squared": float(best_fit.r_squared),
            "mape": float(mape) if mape is not None else None,
            "mae": mae,
            "predictions": [
                {
                    "run_name": heldout_df.iloc[i].get("run_name"),
                    "true": float(L_test[i]),
                    "pred": float(y_pred[i]),
                    "abs_err": float(abs(L_test[i] - y_pred[i])),
                }
                for i in range(len(heldout_df))
            ],
        }
        rows.append(row)
        if verbose:
            mape_str = f"{row['mape']:.2f}%" if row["mape"] is not None else "N/A"
            print(
                f"  {holdout_size:<8} {row['n_train']:>7d} {row['n_test']:>6d} "
                f"{row['model']:<20} {mape_str:>8} {row['mae']:>8.4f}"
            )

    avg_mape = float(np.mean([row["mape"] for row in rows if row["mape"] is not None])) if rows else None
    worst_holdout = max(rows, key=lambda row: row["mape"] if row["mape"] is not None else -np.inf)["holdout_llm_size"] if rows else None
    if verbose and avg_mape is not None:
        print(f"  Mean hold-out MAPE: {avg_mape:.2f}%")
        print(f"  Worst hold-out scale: {worst_holdout}")

    return {
        "canonical_filter": canonical_meta,
        "rows": rows,
        "mean_mape": avg_mape,
        "worst_holdout": worst_holdout,
    }


def bootstrap_joint_parameter_ci(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Bootstrap 95% CIs for the additive joint-law parameters."""
    if n_bootstrap <= 0:
        return {"error": "n_bootstrap must be positive"}

    df_fit, N_L, D, N_A, T, L = _extract_joint_fit_arrays(df)
    point_fit = fit_joint_additive_only(df_fit, verbose=False)
    if point_fit is None:
        return {"error": "point_fit_failed"}

    rng = np.random.default_rng(seed)
    param_names = list(point_fit.params.keys())
    samples = {name: [] for name in param_names}
    success = 0
    failures = 0
    attempts = 0
    max_attempts = max(n_bootstrap * 3, n_bootstrap + 10)

    while success < n_bootstrap and attempts < max_attempts:
        idx = rng.integers(0, len(L), size=len(L))
        boot_df = df_fit.iloc[idx].copy()
        boot_fit = fit_joint_additive_only(boot_df, verbose=False)
        attempts += 1
        if boot_fit is None:
            failures += 1
            continue
        for name in param_names:
            samples[name].append(float(boot_fit.params[name]))
        success += 1

    param_summary = {}
    for name in param_names:
        values = np.asarray(samples[name], dtype=float)
        if len(values) == 0:
            param_summary[name] = {
                "point_estimate": float(point_fit.params[name]),
                "ci_lower": None,
                "ci_upper": None,
                "includes_zero": None,
                "n_success": 0,
            }
            continue
        ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
        param_summary[name] = {
            "point_estimate": float(point_fit.params[name]),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "includes_zero": bool(ci_lower <= 0.0 <= ci_upper),
            "n_success": int(len(values)),
        }

    if verbose:
        print("\n--- Bootstrap CI (joint_additive) ---")
        print(f"  Requested={n_bootstrap}, successful={success}, failures={failures}, attempts={attempts}")
        print(f"  {'Param':<10} {'Point':>12} {'CI low':>12} {'CI high':>12} {'0 in CI':>8}")
        print("  " + "-" * 62)
        for name in param_names:
            row = param_summary[name]
            low = "N/A" if row["ci_lower"] is None else f"{row['ci_lower']:.6g}"
            high = "N/A" if row["ci_upper"] is None else f"{row['ci_upper']:.6g}"
            zero_flag = "N/A" if row["includes_zero"] is None else ("yes" if row["includes_zero"] else "no")
            print(f"  {name:<10} {row['point_estimate']:>12.6g} {low:>12} {high:>12} {zero_flag:>8}")

    return {
        "n_requested": int(n_bootstrap),
        "n_success": int(success),
        "n_failures": int(failures),
        "seed": int(seed),
        "parameters": param_summary,
    }


def analyze_t_equivalence(
    df_g2: pd.DataFrame,
    seed_sigma: float = 0.049,
    df_g25: pd.DataFrame | None = None,
    verbose: bool = True,
) -> dict:
    """Compare T sweep variability against the seed-noise reference sigma."""
    df_g2 = _ensure_llm_size_col(df_g2)
    rows = []

    def _sample_std(values: np.ndarray) -> float:
        ddof = 1 if len(values) > 1 else 0
        return float(np.std(values, ddof=ddof))

    def _ratio(value: float | None) -> float | None:
        if value is None or seed_sigma <= 0:
            return None
        return float(value / seed_sigma)

    def summarize_ranges(frame: pd.DataFrame, label: str) -> list[dict]:
        summary_rows = []
        for llm_size in sorted(frame["llm_size"].dropna().unique(), key=_llm_size_sort_key):
            subset = frame[frame["llm_size"] == llm_size].sort_values("num_queries").copy()
            if len(subset) < 2:
                continue

            losses = subset["best_val_loss"].values.astype(float)
            t_values = subset["num_queries"].values.astype(float)
            best_idx = int(np.argmin(losses))
            worst_idx = int(np.argmax(losses))
            t_range = float(losses.max() - losses.min())
            loss_std = _sample_std(losses)

            practical_mask = (t_values >= 16) & (t_values <= 128)
            practical_t = t_values[practical_mask]
            practical_losses = losses[practical_mask]
            practical_range = None
            practical_std = None
            practical_best_t = None
            practical_worst_t = None
            if len(practical_losses) >= 2:
                practical_range = float(practical_losses.max() - practical_losses.min())
                practical_std = _sample_std(practical_losses)
                practical_best_t = float(practical_t[int(np.argmin(practical_losses))])
                practical_worst_t = float(practical_t[int(np.argmax(practical_losses))])

            t_fits = fit_and_compare(t_values, losses, _t_marginal_candidate_defs(), verbose=False)
            best_fit = t_fits[0] if t_fits else None

            range_ratio = _ratio(t_range)
            std_ratio = _ratio(loss_std)
            practical_range_ratio = _ratio(practical_range)
            practical_std_ratio = _ratio(practical_std)

            summary_rows.append({
                "regime": label,
                "llm_size": llm_size,
                "n_points": int(len(subset)),
                "t_range": t_range,
                "range_over_seed_sigma": range_ratio,
                "loss_std": loss_std,
                "std_over_seed_sigma": std_ratio,
                "best_t": float(t_values[best_idx]),
                "worst_t": float(t_values[worst_idx]),
                "below_seed_noise": bool(range_ratio < 1.0) if range_ratio is not None else None,
                "std_below_seed_noise": bool(std_ratio < 1.0) if std_ratio is not None else None,
                "practical_t_window": [16.0, 128.0],
                "practical_n_points": int(len(practical_losses)),
                "practical_t_range": practical_range,
                "practical_range_over_seed_sigma": practical_range_ratio,
                "practical_loss_std": practical_std,
                "practical_std_over_seed_sigma": practical_std_ratio,
                "practical_best_t": practical_best_t,
                "practical_worst_t": practical_worst_t,
                "practical_below_seed_noise": bool(practical_range_ratio < 1.0) if practical_range_ratio is not None else None,
                "practical_std_below_seed_noise": bool(practical_std_ratio < 1.0) if practical_std_ratio is not None else None,
                "best_t_fit": best_fit.name if best_fit else None,
                "best_t_fit_r_squared": float(best_fit.r_squared) if best_fit else None,
            })
        return summary_rows

    rows.extend(summarize_ranges(df_g2, label="g2_1ep"))
    g25_rows = []
    if df_g25 is not None and len(df_g25) > 0:
        df_g25 = _ensure_llm_size_col(df_g25)
        g25_rows = summarize_ranges(df_g25, label="g25_3ep")
        rows.extend(g25_rows)

    if verbose:
        print("\n--- T Equivalence Test ---")
        print(f"  Reference seed sigma: {seed_sigma:.3f}")
        print(
            f"  {'Regime':<10} {'LLM':<6} {'range/σ':>8} {'std/σ':>8} {'16-128/σ':>10} {'fit':<12} {'R²':>7}"
        )
        print("  " + "-" * 76)
        for row in rows:
            range_ratio = "N/A" if row["range_over_seed_sigma"] is None else f"{row['range_over_seed_sigma']:.2f}"
            std_ratio = "N/A" if row["std_over_seed_sigma"] is None else f"{row['std_over_seed_sigma']:.2f}"
            practical_ratio = (
                "N/A"
                if row["practical_range_over_seed_sigma"] is None
                else f"{row['practical_range_over_seed_sigma']:.2f}"
            )
            fit_name = row["best_t_fit"] or "N/A"
            fit_r2 = "N/A" if row["best_t_fit_r_squared"] is None else f"{row['best_t_fit_r_squared']:.2f}"
            print(
                f"  {row['regime']:<10} {row['llm_size']:<6} {range_ratio:>8} {std_ratio:>8} {practical_ratio:>10} "
                f"{fit_name:<12} {fit_r2:>7}"
            )

    g2_only = [row for row in rows if row["regime"] == "g2_1ep"]
    g2_range_ratios = [row["range_over_seed_sigma"] for row in g2_only if row["range_over_seed_sigma"] is not None]
    g2_std_ratios = [row["std_over_seed_sigma"] for row in g2_only if row["std_over_seed_sigma"] is not None]
    g2_practical_range_ratios = [
        row["practical_range_over_seed_sigma"]
        for row in g2_only
        if row["practical_range_over_seed_sigma"] is not None
    ]

    return {
        "seed_sigma": float(seed_sigma),
        "g2_rows": g2_only,
        "g25_rows": g25_rows,
        "mean_ratio_g2": float(np.mean(g2_range_ratios)) if g2_range_ratios else None,
        "max_ratio_g2": float(np.max(g2_range_ratios)) if g2_range_ratios else None,
        "mean_std_ratio_g2": float(np.mean(g2_std_ratios)) if g2_std_ratios else None,
        "max_std_ratio_g2": float(np.max(g2_std_ratios)) if g2_std_ratios else None,
        "mean_practical_ratio_g2": float(np.mean(g2_practical_range_ratios)) if g2_practical_range_ratios else None,
        "max_practical_ratio_g2": float(np.max(g2_practical_range_ratios)) if g2_practical_range_ratios else None,
        "all_g2_below_seed_noise": bool(all(row["below_seed_noise"] for row in g2_only)) if g2_only else None,
        "all_g2_std_below_seed_noise": bool(all(row["std_below_seed_noise"] for row in g2_only)) if g2_only else None,
        "all_g2_practical_below_seed_noise": (
            bool(all(row["practical_below_seed_noise"] for row in g2_only if row["practical_below_seed_noise"] is not None))
            if any(row["practical_below_seed_noise"] is not None for row in g2_only)
            else None
        ),
    }

def summarize_effective_params(fit_result: FitResult, zero_threshold: float = 1e-10) -> dict:
    """Report how many fitted parameters are effectively active."""
    effectively_zero = []
    effective_names = []

    for name, value in fit_result.params.items():
        value_f = float(value)
        if abs(value_f) < zero_threshold:
            effectively_zero.append({"name": name, "value": value_f})
        else:
            effective_names.append(name)

    return {
        "zero_threshold": float(zero_threshold),
        "total_params": int(len(fit_result.params)),
        "effective_params": int(len(effective_names)),
        "effective_param_names": effective_names,
        "effectively_zero_params": effectively_zero,
    }


def _format_effective_param_report(report: dict) -> str:
    """Format effective-parameter information for console output."""
    zero_parts = [f"{item['name']}={item['value']:.2e}" for item in report.get("effectively_zero_params", [])]
    zero_str = ", ".join(zero_parts) if zero_parts else "none"
    return (
        f"{report.get('effective_params', 0)}/{report.get('total_params', 0)} effective "
        f"(zero-ish < {report.get('zero_threshold', 0.0):.0e}: {zero_str})"
    )


def prepare_joint_variant_data(
    df: pd.DataFrame,
    exclude_llm_sizes: list[str] | None = None,
    divergence_threshold: float = 0.5,
    auto_exclude_divergent: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Apply optional size exclusions and divergence filtering to a joint-fit pool."""
    df = _ensure_group_col(df)
    df = _ensure_llm_size_col(df)
    df = df.copy()

    exclude_llm_sizes = sorted({size for size in (exclude_llm_sizes or []) if size})
    metadata = {
        "exclude_llm_sizes": exclude_llm_sizes,
        "divergence_threshold": float(divergence_threshold),
        "auto_exclude_divergent": bool(auto_exclude_divergent),
        "n_input": int(len(df)),
        "n_excluded_by_size": 0,
        "excluded_by_size": [],
        "n_excluded_divergent": 0,
        "excluded_divergent": [],
    }

    if exclude_llm_sizes:
        size_mask = df["llm_size"].isin(exclude_llm_sizes)
        excluded_by_size = df[size_mask].copy()
        excluded_by_size = excluded_by_size.sort_values(["llm_size", "run_name"])
        metadata["excluded_by_size"] = [
            {
                "run_name": row.get("run_name"),
                "group": row.get("group"),
                "llm_name": row.get("llm_name"),
                "llm_size": row.get("llm_size"),
            }
            for _, row in excluded_by_size.iterrows()
        ]
        metadata["n_excluded_by_size"] = len(metadata["excluded_by_size"])
        if verbose and metadata["n_excluded_by_size"]:
            print(
                f"  Excluding {metadata['n_excluded_by_size']} runs by LLM size: "
                f"{', '.join(exclude_llm_sizes)}"
            )
        df = df[~size_mask].copy()

    metadata["n_after_size_filter"] = int(len(df))

    if auto_exclude_divergent:
        df, excluded_divergent = exclude_divergent_runs(
            df, threshold=divergence_threshold, verbose=verbose
        )
        metadata["excluded_divergent"] = excluded_divergent
        metadata["n_excluded_divergent"] = len(excluded_divergent)

    metadata["n_points"] = int(len(df))
    return df, metadata


def _select_recommended_joint_variant(summaries: list[dict]) -> str | None:
    """Pick the recommended variant, prioritizing lower MAPE."""
    valid = [row for row in summaries if row.get("mape") is not None]
    if not valid:
        return None

    def sort_key(row: dict) -> tuple:
        bic = row.get("bic")
        r2 = row.get("r_squared")
        return (
            float(row["mape"]),
            float(bic) if bic is not None else float("inf"),
            -float(r2) if r2 is not None else float("inf"),
            -int(row.get("n_points", 0)),
        )

    return sorted(valid, key=sort_key)[0]["label"]


def _summarize_joint_result(label: str, result: dict) -> dict:
    """Create a compact summary row for a joint-law fit result."""
    summary = {
        "label": label,
        "n_points": int(result.get("n_points", 0)),
        "n_excluded_by_size": int(result.get("n_excluded_by_size", 0)),
        "n_excluded_divergent": int(result.get("n_excluded_divergent", 0)),
        "n_excluded_total": int(result.get("n_excluded_by_size", 0)) + int(result.get("n_excluded_divergent", 0)),
        "winner": None,
        "bic": None,
        "aic": None,
        "r_squared": None,
        "rmse": None,
        "mape": None,
        "total_params": None,
        "effective_params": None,
        "effectively_zero_params": [],
    }

    fits = result.get("fits", [])
    if fits:
        best = fits[0]
        mape = result.get("mape")
        if mape is None:
            mape = _compute_mape(result.get("y_true"), result.get("y_pred"))
        param_report = summarize_effective_params(best)
        summary.update({
            "winner": best.name,
            "bic": float(best.bic),
            "aic": float(best.aic),
            "r_squared": float(best.r_squared),
            "rmse": float(best.rmse),
            "mape": float(mape) if mape is not None else None,
            "total_params": int(param_report["total_params"]),
            "effective_params": int(param_report["effective_params"]),
            "effectively_zero_params": param_report["effectively_zero_params"],
        })

    return summary


def analyze_joint(
    df: pd.DataFrame,
    verbose: bool = True,
    divergence_threshold: float = 0.5,
    auto_exclude_divergent: bool = True,
    exclude_llm_sizes: list[str] | None = None,
) -> dict:
    """Fit joint scaling law on the requested fit set."""
    if verbose:
        print("\n--- Joint Scaling Law ---")

    na_col = "adapter_params" if "adapter_params" in df.columns else "adapter_total"
    df = _parse_llm_params_col(df)
    df = _ensure_group_col(df)
    df = _ensure_llm_size_col(df)

    df_fit, filter_meta = prepare_joint_variant_data(
        df,
        exclude_llm_sizes=exclude_llm_sizes,
        divergence_threshold=divergence_threshold,
        auto_exclude_divergent=auto_exclude_divergent,
        verbose=verbose,
    )

    result = dict(filter_meta)
    if len(df_fit) > 0:
        result["group_counts"] = {
            str(k): int(v) for k, v in df_fit["group"].value_counts().sort_index().items()
        }
    else:
        result["group_counts"] = {}

    if len(df_fit) <= 10:
        if verbose:
            print("  Not enough data points for joint fitting after filtering.")
        result["fits"] = []
        return result

    N_L = df_fit["llm_params"].values.astype(float)
    D = _compute_effective_d(df_fit)
    N_A = df_fit[na_col].values.astype(float)
    T = df_fit["num_queries"].values.astype(float)
    L = df_fit["best_val_loss"].values.astype(float)

    if verbose:
        print(
            f"  Data points: {len(L)}, N_L range: [{N_L.min():.0e}, {N_L.max():.0e}], "
            f"D range: [{D.min():.0e}, {D.max():.0e}]"
        )

    fits = fit_and_compare((N_L, D, N_A, T), L, _joint_candidate_defs(), verbose=verbose)

    result["fits"] = fits
    if fits:
        best = fits[0]
        result["winner"] = best.name
        y_pred = _predict_joint_fit_result(best, N_L, D, N_A, T)
        result["y_pred"] = y_pred
        result["y_true"] = L
        result["mape"] = _compute_mape(L, y_pred)
        result["winner_effective_params"] = summarize_effective_params(best)

        if verbose:
            print(f"  >>> Winner: {best.name} (R²={best.r_squared:.4f}, MAPE={result['mape']:.2f}%)")
            print(f"  >>> Effective params: {_format_effective_param_report(result['winner_effective_params'])}")
            if len(fits) > 1:
                print(f"  >>> ΔBIC = {fits[1].bic - fits[0].bic:.2f}")

    return result


def search_epoch_exponent(
    df: pd.DataFrame,
    eta_range: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Grid-search the epoch exponent η that maximizes joint-law R².

    D_eff = D_unique × epochs^η.  Sweeps η from 0 to 1.
    """
    if eta_range is None:
        eta_range = np.arange(0.0, 1.05, 0.05)

    na_col = "adapter_params" if "adapter_params" in df.columns else "adapter_total"
    df = _parse_llm_params_col(df)

    N_L = df["llm_params"].values.astype(float)
    N_A = df[na_col].values.astype(float)
    T = df["num_queries"].values.astype(float)
    L = df["best_val_loss"].values.astype(float)

    best_eta, best_r2, best_mape = 1.0, -np.inf, np.inf
    rows = []

    for eta in eta_range:
        D = _compute_effective_d(df, epoch_exponent=float(eta))
        candidates = {
            "joint_additive": (
                joint_additive,
                [1e3, 0.1, 1e3, 0.1, 1e4, 0.3, 1e-9, 0.001, 1.0, 2.5],
                ([0]*9 + [0], [np.inf, 2, np.inf, 2, np.inf, 5, np.inf, np.inf, np.inf, np.inf]),
            ),
        }
        fits = fit_and_compare((N_L, D, N_A, T), L, candidates, verbose=False)
        if fits:
            r2 = fits[0].r_squared
            y_pred = joint_additive(
                (N_L, D, N_A, T),
                *[fits[0].params[k] for k in ["a", "alpha", "b", "beta", "c", "gamma", "h", "e", "f", "eps"]],
            )
            mape = float(np.mean(np.abs((L - y_pred) / L)) * 100)
            rows.append({"eta": round(float(eta), 2), "r_squared": r2, "mape": mape, "bic": fits[0].bic})
            if r2 > best_r2:
                best_eta, best_r2, best_mape = float(eta), r2, mape

    if verbose:
        print(f"\n--- Epoch Exponent Search ---")
        print(f"  {'η':>5} {'R²':>10} {'MAPE':>10} {'BIC':>12}")
        for row in rows:
            marker = " ← best" if abs(row["eta"] - best_eta) < 0.01 else ""
            print(f"  {row['eta']:>5.2f} {row['r_squared']:>10.4f} {row['mape']:>9.2f}% {row['bic']:>12.2f}{marker}")
        print(f"  Best η={best_eta:.2f}, R²={best_r2:.4f}, MAPE={best_mape:.2f}%")

    return {
        "best_eta": best_eta,
        "best_r2": best_r2,
        "best_mape": best_mape,
        "grid": rows,
    }


def analyze_rho_invariance(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Test ρ=T/T₀ invariance from G4 data.

    G4 has 2 resolutions × 5 T values. Test: does L(ρ) collapse across resolutions?
    If L depends only on ρ (not on T and T₀ separately), the two curves should overlap.
    """
    if verbose:
        print("\n--- ρ Invariance Analysis (G4) ---")

    # image_size → T₀ mapping (SigLIP patch14)
    T0_MAP = {384: 729, 224: 256}

    df = df.copy()
    df["image_size"] = df["image_size"].astype(int)
    df["T0"] = df["image_size"].map(T0_MAP).fillna(729).astype(float)
    df["T"] = df["num_queries"].astype(float)
    df["rho"] = df["T"] / df["T0"]

    results = {"data": []}

    # Fit L(ρ) separately for each resolution
    for res in sorted(df["image_size"].unique()):
        subset = df[df["image_size"] == res].sort_values("T")
        t0 = T0_MAP[res]
        if verbose:
            print(f"\n  Resolution {res} (T₀={t0}):")
            for _, row in subset.iterrows():
                print(f"    T={int(row['T']):>4d}, ρ={row['rho']:.4f}, L={row['best_val_loss']:.4f}")
        results["data"].extend(subset[["T", "T0", "rho", "best_val_loss", "image_size"]].to_dict("records"))

    # Compare: for same T but different resolutions
    if verbose:
        print(f"\n  Same-T comparison (L difference across resolutions):")
    deltas = []
    for t_val in sorted(df["T"].unique()):
        t_data = df[df["T"] == t_val].sort_values("image_size")
        if len(t_data) == 2:
            l_224 = t_data[t_data["image_size"] == 224]["best_val_loss"].values[0]
            l_384 = t_data[t_data["image_size"] == 384]["best_val_loss"].values[0]
            delta = abs(l_224 - l_384)
            deltas.append(delta)
            if verbose:
                print(f"    T={int(t_val):>4d}: L_224={l_224:.4f}, L_384={l_384:.4f}, Δ={delta:.4f}")

    results["mean_delta"] = np.mean(deltas) if deltas else None
    if verbose and deltas:
        print(f"\n  Mean |ΔL| across resolutions: {np.mean(deltas):.4f}")
        print(f"  → {'Partial invariance' if np.mean(deltas) < 0.1 else 'Resolution matters'}")

    return results


def analyze_d_independence(df_d: pd.DataFrame, t_results_per_d: dict = None,
                           verbose: bool = True) -> dict:
    """Check if T_opt and r_A_opt are stable across D values (G5v2)."""
    if verbose:
        print("\n--- D Independence Analysis (G5v2) ---")

    results = {}

    # Group by D level (num_samples or actual D)
    d_col = "num_samples" if "num_samples" in df_d.columns else "seen_pairs"

    # Drop NaN D values
    df_d = df_d.dropna(subset=[d_col])

    for d_val in sorted(df_d[d_col].unique()):
        subset = df_d[df_d[d_col] == d_val]
        d_label = f"D={int(d_val)}"

        # T sweep subset (fixed N_A=M)
        t_subset = subset[subset["adapter_level"] == "M"] if "adapter_level" in subset.columns else subset
        if len(t_subset) >= 3:
            t_fits = analyze_marginal_t(t_subset, verbose=False)
            for llm, res in t_fits.items():
                t_opt = res.get("T_opt")
                if verbose:
                    print(f"  {d_label}, {llm}: T_opt = {t_opt:.1f}" if t_opt else f"  {d_label}, {llm}: T_opt = N/A")
                results.setdefault(d_label, {})[llm] = {"T_opt": t_opt}

    return results


def validate_extrapolation(df_fit: pd.DataFrame, df_test: pd.DataFrame,
                           joint_result: dict, verbose: bool = True) -> dict:
    """Validate joint law by fitting on smaller LLMs and predicting larger ones."""
    if verbose:
        print("\n--- Extrapolation Validation ---")

    if not joint_result or "fits" not in joint_result or not joint_result["fits"]:
        if verbose:
            print("  No joint law fit available.")
        return {}

    best = joint_result["fits"][0]
    na_col = "adapter_params" if "adapter_params" in df_test.columns else "adapter_total"

    if "llm_params" not in df_test.columns:
        def parse_params(name):
            s = name.split("-")[-1]
            if s.endswith("B"):
                return float(s[:-1]) * 1e9
            elif s.endswith("M"):
                return float(s[:-1]) * 1e6
            return float(s)
        df_test = df_test.copy()
        df_test["llm_params"] = df_test["llm_name"].apply(parse_params)

    N_L = df_test["llm_params"].values.astype(float)
    D = _compute_effective_d(df_test)
    N_A = df_test[na_col].values.astype(float)
    T = df_test["num_queries"].values.astype(float)
    L = df_test["best_val_loss"].values.astype(float)

    y_pred = _predict_joint_fit_result(best, N_L, D, N_A, T)

    mape = np.mean(np.abs((L - y_pred) / L)) * 100
    mae = np.mean(np.abs(L - y_pred))

    if verbose:
        print(f"  Test set: {len(df_test)} experiments")
        print(f"  MAPE = {mape:.2f}%, MAE = {mae:.4f}")
        for i in range(len(df_test)):
            rn = df_test.iloc[i].get("run_name", f"exp{i}")
            print(f"    {rn}: true={L[i]:.4f}, pred={y_pred[i]:.4f}, err={abs(L[i]-y_pred[i]):.4f}")

    return {"mape": mape, "mae": mae, "y_true": L, "y_pred": y_pred}


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for the mean."""
    means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True))
                      for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return np.percentile(means, 100 * alpha), np.percentile(means, 100 * (1 - alpha))


def cross_validate_fit(
    x_data,
    y_data,
    candidates: dict,
    k: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """k-fold cross-validation for scaling law model selection.

    Returns list of dicts with cv_mape, cv_rmse, train_r2 for each candidate.
    """
    n = len(y_data)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // k

    cv_results = []

    for name, (func, p0, bounds) in candidates.items():
        fold_mapes = []
        fold_rmses = []
        fold_r2s = []

        for fold in range(k):
            test_idx = indices[fold * fold_size: (fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)

            if len(train_idx) < len(p0) + 1:
                continue

            # Select train/test data
            if isinstance(x_data, tuple):
                x_train = tuple(x[train_idx] for x in x_data)
                x_test = tuple(x[test_idx] for x in x_data)
            else:
                x_train = x_data[train_idx]
                x_test = x_data[test_idx]
            y_train = y_data[train_idx]
            y_test = y_data[test_idx]

            try:
                popt, _ = curve_fit(func, x_train, y_train, p0=p0, bounds=bounds, maxfev=10000)
                y_pred_test = func(x_test, *popt)
                y_pred_train = func(x_train, *popt)

                mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
                rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
                ss_res = np.sum((y_train - y_pred_train) ** 2)
                ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                fold_mapes.append(mape)
                fold_rmses.append(rmse)
                fold_r2s.append(r2)
            except Exception:
                continue

        if fold_mapes:
            result = {
                "name": name,
                "cv_mape": float(np.mean(fold_mapes)),
                "cv_mape_std": float(np.std(fold_mapes)),
                "cv_rmse": float(np.mean(fold_rmses)),
                "cv_rmse_std": float(np.std(fold_rmses)),
                "train_r2": float(np.mean(fold_r2s)),
                "n_folds": len(fold_mapes),
            }
            cv_results.append(result)
            if verbose:
                print(f"  {name}: CV MAPE={result['cv_mape']:.2f}%±{result['cv_mape_std']:.2f}, "
                      f"CV RMSE={result['cv_rmse']:.4f}, train R²={result['train_r2']:.4f}")

    cv_results.sort(key=lambda r: r["cv_mape"])
    return cv_results


def analyze_depth(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Analyze depth ablation from G8 data.

    G8 varies adapter_num_layers ∈ {1, 2, 4, 6} at fixed T=64, N_A_level=M.
    Tests whether deeper adapters improve performance.
    """
    if verbose:
        print("\n--- Depth Ablation (G8) ---")

    results = {}

    # Parse depth from run_name if no column exists
    depth_col = None
    for col in ["adapter_num_layers", "vision_adapter_num_layers"]:
        if col in df.columns and df[col].notna().any():
            depth_col = col
            break
    if depth_col is None:
        # Parse from run_name: e.g., "g8_3B_T64_M_d50m_depth4_s42" → 4
        import re
        df = df.copy()
        df["depth"] = df["run_name"].str.extract(r"depth(\d+)").astype(float)
        # Default depth=2 for runs without explicit depth tag
        df["depth"] = df["depth"].fillna(2)
        depth_col = "depth"

    for n_l in sorted(df["llm_name"].unique()):
        subset = df[df["llm_name"] == n_l].sort_values(depth_col)
        if len(subset) < 2:
            continue

        depths = subset[depth_col].values.astype(float)
        losses = subset["best_val_loss"].values.astype(float)

        # Simple analysis: depth vs loss
        best_idx = np.argmin(losses)
        best_depth = int(depths[best_idx])
        best_loss = losses[best_idx]

        # Diminishing returns: gain from depth=1→2 vs 2→4 vs 4→6
        gains = {}
        depth_vals = sorted(set(depths))
        for i in range(1, len(depth_vals)):
            d_prev, d_curr = depth_vals[i-1], depth_vals[i]
            l_prev = subset[subset[depth_col] == d_prev]["best_val_loss"].values[0]
            l_curr = subset[subset[depth_col] == d_curr]["best_val_loss"].values[0]
            gains[f"{int(d_prev)}→{int(d_curr)}"] = l_prev - l_curr

        results[n_l] = {
            "best_depth": best_depth,
            "best_loss": best_loss,
            "depths": depths.tolist(),
            "losses": losses.tolist(),
            "gains": gains,
        }

        if verbose:
            print(f"  {n_l}: best_depth={best_depth}, best_loss={best_loss:.4f}")
            for step, gain in gains.items():
                print(f"    {step}: ΔL = {gain:.4f} ({'↓ better' if gain > 0 else '↑ worse'})")

    return results


def analyze_lora_comparison(df_frozen: pd.DataFrame, df_lora: pd.DataFrame,
                            verbose: bool = True) -> dict:
    """Compare frozen LLM vs LoRA-unfrozen from G6 data.

    Matches configs on (llm_name, num_queries, adapter_level) and compares losses.
    """
    if verbose:
        print("\n--- LoRA vs Frozen Comparison (G6) ---")

    results = {"comparisons": [], "summary": {}}

    # Merge on common config keys
    key_cols = ["llm_name", "num_queries", "adapter_level"]

    for _, lora_row in df_lora.iterrows():
        match = df_frozen
        for col in key_cols:
            if col in lora_row and col in match.columns:
                match = match[match[col] == lora_row[col]]

        if len(match) == 0:
            continue

        frozen_loss = match["best_val_loss"].values[0]
        lora_loss = lora_row["best_val_loss"]
        improvement = frozen_loss - lora_loss

        comp = {
            "config": f"{lora_row['llm_name'].split('-')[-1]} T={int(lora_row['num_queries'])} {lora_row.get('adapter_level', 'M')}",
            "frozen_loss": float(frozen_loss),
            "lora_loss": float(lora_loss),
            "improvement": float(improvement),
        }
        results["comparisons"].append(comp)

        if verbose:
            print(f"  {comp['config']}: frozen={frozen_loss:.4f}, LoRA={lora_loss:.4f}, "
                  f"Δ={improvement:.4f} ({'✓' if improvement > 0 else '✗'})")

    if results["comparisons"]:
        improvements = [c["improvement"] for c in results["comparisons"]]
        results["summary"] = {
            "n_comparisons": len(improvements),
            "mean_improvement": float(np.mean(improvements)),
            "all_positive": all(i > 0 for i in improvements),
        }
        if verbose:
            print(f"\n  Mean improvement: {np.mean(improvements):.4f}")
            print(f"  All LoRA better: {results['summary']['all_positive']}")

    return results


def generate_bic_summary(all_results: dict, verbose: bool = True) -> list[dict]:
    """Generate BIC model comparison summary table across all analyses."""
    if verbose:
        print("\n--- BIC Model Selection Summary ---")

    rows = []

    # T marginal
    if "t_marginal" in all_results:
        for n_l, res in all_results["t_marginal"].items():
            for fit in res.get("all_fits", []):
                rows.append({
                    "analysis": "T marginal",
                    "subset": n_l.split("-")[-1],
                    "model": fit.name,
                    "k": len(fit.params),
                    "BIC": round(fit.bic, 2),
                    "AIC": round(fit.aic, 2),
                    "R2": round(fit.r_squared, 6),
                    "RMSE": round(fit.rmse, 4),
                })

    # N_A marginal
    if "na_marginal" in all_results:
        for n_l, res in all_results["na_marginal"].items():
            for fit in res.get("all_fits", []):
                rows.append({
                    "analysis": "N_A marginal",
                    "subset": n_l.split("-")[-1],
                    "model": fit.name,
                    "k": len(fit.params),
                    "BIC": round(fit.bic, 2),
                    "AIC": round(fit.aic, 2),
                    "R2": round(fit.r_squared, 6),
                    "RMSE": round(fit.rmse, 4),
                })

    # Interaction
    if "interaction" in all_results:
        for fit in all_results["interaction"].get("fits", []):
            rows.append({
                "analysis": "T×N_A interaction",
                "subset": "G3 (3B)",
                "model": fit.name,
                "k": len(fit.params),
                "BIC": round(fit.bic, 2),
                "AIC": round(fit.aic, 2),
                "R2": round(fit.r_squared, 6),
                "RMSE": round(fit.rmse, 4),
            })

    # Joint
    if "joint" in all_results:
        for fit in all_results["joint"].get("fits", []):
            rows.append({
                "analysis": "Joint law",
                "subset": "All valid",
                "model": fit.name,
                "k": len(fit.params),
                "BIC": round(fit.bic, 2),
                "AIC": round(fit.aic, 2),
                "R2": round(fit.r_squared, 6),
                "RMSE": round(fit.rmse, 4),
            })

    if verbose:
        print(f"  {'Analysis':<20} {'Subset':<10} {'Model':<20} {'k':>3} {'BIC':>10} {'R²':>10}")
        print("  " + "-" * 75)
        for r in rows:
            print(f"  {r['analysis']:<20} {r['subset']:<10} {r['model']:<20} {r['k']:>3} "
                  f"{r['BIC']:>10.2f} {r['R2']:>10.6f}")

    return rows


def analyze_seed_stability(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Analyze seed-to-seed variation from G9 data."""
    if verbose:
        print("\n--- Seed Stability (G9) ---")

    results = {}

    # Group by config (everything except seed)
    if "seed" not in df.columns:
        # Extract seed from run_name
        df = df.copy()
        df["seed"] = df["run_name"].str.extract(r"s(\d+)$").astype(float)

    # Group by base config (run_name without seed)
    df["base_config"] = df["run_name"].str.replace(r"_s\d+$", "", regex=True)

    for config, group in df.groupby("base_config"):
        if len(group) < 2:
            continue
        losses = group["best_val_loss"].values
        mean_l = np.mean(losses)
        std_l = np.std(losses)
        ci_low, ci_high = bootstrap_ci(losses)
        spread = np.max(losses) - np.min(losses)

        results[config] = {
            "mean": mean_l, "std": std_l, "spread": spread,
            "ci_95": (ci_low, ci_high), "n_seeds": len(group),
            "values": losses.tolist(),
        }

        if verbose:
            print(f"  {config}: mean={mean_l:.4f}±{std_l:.4f}, spread={spread:.4f}, "
                  f"95%CI=[{ci_low:.4f}, {ci_high:.4f}]")

    return results


def compute_noise_floor(
    seed_results: dict,
    total_var: float,
    exclude_05b: bool = True,
    verbose: bool = True,
) -> dict:
    """Compute noise floor and max achievable R² from seed variance.

    Uses G9 seed data to estimate irreducible noise variance, then computes
    the theoretical maximum R² achievable by any model.
    """
    stds = []
    stds_by_config = {}
    for config, data in seed_results.items():
        if exclude_05b and "0.5B" in config:
            continue
        if data["n_seeds"] >= 2:
            stds.append(data["std"])
            stds_by_config[config] = data["std"]

    if not stds:
        return {"error": "no valid seed data"}

    mean_std = np.mean(stds)
    median_std = np.median(stds)

    noise_var_mean = mean_std ** 2
    noise_var_median = median_std ** 2
    max_r2_mean = 1 - noise_var_mean / total_var
    max_r2_median = 1 - noise_var_median / total_var

    result = {
        "seed_stds": stds_by_config,
        "mean_seed_std": float(mean_std),
        "median_seed_std": float(median_std),
        "noise_var_mean": float(noise_var_mean),
        "noise_var_median": float(noise_var_median),
        "total_var": float(total_var),
        "max_r2_mean": float(max_r2_mean),
        "max_r2_median": float(max_r2_median),
    }

    if verbose:
        print(f"\n--- Noise Floor Analysis ---")
        print(f"  Seed stds (excl 0.5B): {[f'{s:.4f}' for s in stds]}")
        print(f"  Mean seed σ: {mean_std:.4f}, Median seed σ: {median_std:.4f}")
        print(f"  Total variance of fit data: {total_var:.6f}")
        print(f"  Noise variance (median): {noise_var_median:.6f}")
        print(f"  Max achievable R² (median): {max_r2_median:.4f}")
        print(f"  Max achievable R² (mean):   {max_r2_mean:.4f}")

    return result


def _serialize_fit_result(fr: 'FitResult') -> dict:
    """Convert a FitResult to a JSON-serializable dict."""
    return {
        "name": fr.name,
        "params": {k: float(v) for k, v in fr.params.items()},
        "bic": float(fr.bic),
        "aic": float(fr.aic),
        "r_squared": float(fr.r_squared),
        "rmse": float(fr.rmse),
        "effective_param_report": summarize_effective_params(fr),
    }


def _serialize_value(v):
    """Convert a value to JSON-serializable form."""
    if isinstance(v, FitResult):
        return _serialize_fit_result(v)
    elif isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, (np.floating, np.integer)):
        return float(v)
    elif isinstance(v, dict):
        return {str(kk): _serialize_value(vv) for kk, vv in v.items()}
    elif isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    elif isinstance(v, (int, float, str, bool, type(None))):
        return v
    else:
        return str(v)



def save_results_json(all_results: dict, output_path: str):
    """Save all analysis results to JSON (serializable subset)."""
    serializable = _serialize_value(all_results)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\nResults saved: {output_path}")



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default=os.environ.get("VLM_CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--csv", type=str, default=None,
                        help="Load from CSV instead of results_dir")
    parser.add_argument("--output_dir", type=str, default="analysis/results")
    parser.add_argument("--fit_groups", nargs="+",
                        default=["g0v2", "g1", "g2", "g3", "g4", "g5v2"],
                        help="Groups to include in the joint-law fitting pool")
    parser.add_argument("--divergence_threshold", type=float, default=0.5,
                        help="Exclude runs with final_val_loss - best_val_loss above this threshold")
    parser.add_argument("--exclude_llm_sizes", "--exclude_size", nargs="+", default=[],
                        help="LLM sizes to exclude from the canonical joint fit (e.g. 0.5B)")
    parser.add_argument("--leave_one_out", action="store_true",
                        help="Run leave-one-LLM-out validation on the canonical joint-fit pool")
    parser.add_argument("--bootstrap", nargs="?", const=1000, default=0, type=int,
                        help="Bootstrap CI for additive joint-law parameters; default 1000 when the flag is provided without a value")
    parser.add_argument("--bootstrap_seed", type=int, default=42,
                        help="Random seed for bootstrap resampling")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.csv:
        print(f"Loading from CSV: {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        print("Loading results from checkpoints...")
        df = load_results(args.results_dir)
    print(f"Loaded {len(df)} experiments")

    if len(df) == 0:
        print("No results found. Run experiments first.")
        exit()

    df = _ensure_group_col(df)
    df = _ensure_llm_size_col(df)
    df = _ensure_adapter_type_col(df)
    df = _ensure_depth_col(df)

    # Filter valid groups for non-joint analyses (exclude G0/G5 which have D bug)
    valid = df[~df["run_name"].str.match(r"^g[05]_")]
    all_results = {}

    # 1. T marginal (G2)
    g2 = valid[valid["run_name"].str.startswith("g2_")]
    if len(g2) > 0:
        print("\n" + "=" * 60)
        print("T MARGINAL ANALYSIS (G2)")
        print("=" * 60)
        all_results["t_marginal"] = analyze_marginal_t(g2)

    # 2. N_A marginal (G1)
    g1 = valid[valid["run_name"].str.startswith("g1_")]
    if len(g1) > 0:
        print("\n" + "=" * 60)
        print("N_A MARGINAL ANALYSIS (G1)")
        print("=" * 60)
        all_results["na_marginal"] = analyze_marginal_na(g1)

    # 3. Base power law (G0v2)
    g0v2 = df[df["run_name"].str.startswith("g0v2_")]
    if len(g0v2) > 0:
        print("\n" + "=" * 60)
        print("BASE POWER LAW (G0v2)")
        print("=" * 60)
        all_results["base_power_law"] = analyze_base_power_law(g0v2)

    # 4. T×N_A interaction (G3)
    g3 = valid[valid["run_name"].str.startswith("g3_")]
    if len(g3) > 0:
        print("\n" + "=" * 60)
        print("T × N_A INTERACTION (G3)")
        print("=" * 60)
        all_results["interaction"] = analyze_interaction(g3)

    # 5. Seed stability (G9)
    g9 = valid[valid["run_name"].str.startswith("g9_")]
    if len(g9) > 0:
        print("\n" + "=" * 60)
        print("SEED STABILITY (G9)")
        print("=" * 60)
        all_results["seed_stability"] = analyze_seed_stability(g9)

    g25 = valid[valid["run_name"].str.startswith("g25_")]

    # 5b. T equivalence test (G2 vs seed noise)
    if len(g2) > 0:
        g2_equiv = g2[~g2["llm_size"].isin(args.exclude_llm_sizes)] if args.exclude_llm_sizes else g2
        g25_equiv = g25[~g25["llm_size"].isin(args.exclude_llm_sizes)] if args.exclude_llm_sizes else g25
        print("\n" + "=" * 60)
        print("T EQUIVALENCE TEST")
        print("=" * 60)
        all_results["t_equivalence"] = analyze_t_equivalence(
            g2_equiv,
            seed_sigma=0.049,
            df_g25=g25_equiv if len(g25_equiv) > 0 else None,
        )

    # 6. ρ invariance (G4)
    g4 = valid[valid["run_name"].str.startswith("g4_")]
    if len(g4) > 0:
        print("\n" + "=" * 60)
        print("ρ INVARIANCE (G4)")
        print("=" * 60)
        all_results["rho_invariance"] = analyze_rho_invariance(g4)

    # 7. D independence (G5v2)
    g5v2 = df[df["run_name"].str.startswith("g5v2_")]
    if len(g5v2) > 0:
        print("\n" + "=" * 60)
        print("D INDEPENDENCE (G5v2)")
        print("=" * 60)
        all_results["d_independence"] = analyze_d_independence(g5v2)

    # 8. Depth ablation (G8)
    g8 = valid[valid["run_name"].str.startswith("g8_")]
    if len(g8) > 0:
        print("\n" + "=" * 60)
        print("DEPTH ABLATION (G8)")
        print("=" * 60)
        all_results["depth"] = analyze_depth(g8)

    # 9. LoRA comparison (G6 vs matched frozen configs)
    g6 = valid[valid["run_name"].str.startswith("g6_")]
    if len(g6) > 0:
        if "group" in valid.columns:
            frozen_pool = valid[valid["group"].isin(["g1", "g2"])]
        else:
            frozen_pool = valid[valid["run_name"].str.match(r"^g[12]_")]
        if len(frozen_pool) > 0:
            print("\n" + "=" * 60)
            print("LoRA vs FROZEN (G6)")
            print("=" * 60)
            all_results["lora_comparison"] = analyze_lora_comparison(frozen_pool, g6)

    # 10. Joint law fit set preparation
    joint_pool, joint_fit_selection = prepare_joint_fit_data(df, args.fit_groups, verbose=True)
    all_results["joint_fit_selection"] = joint_fit_selection
    joint_main_data, joint_main_filter = prepare_joint_variant_data(
        joint_pool,
        exclude_llm_sizes=args.exclude_llm_sizes,
        divergence_threshold=args.divergence_threshold,
        auto_exclude_divergent=True,
        verbose=False,
    )
    all_results["joint_main_filter"] = joint_main_filter

    # 10a. Main joint law: canonical fit set with requested exclusions
    if len(joint_pool) > 10:
        print("\n" + "=" * 60)
        print("JOINT SCALING LAW")
        print("=" * 60)
        if args.exclude_llm_sizes:
            print(f"  Canonical size exclusions: {', '.join(args.exclude_llm_sizes)}")
        all_results["joint"] = analyze_joint(
            joint_pool,
            divergence_threshold=args.divergence_threshold,
            auto_exclude_divergent=True,
            exclude_llm_sizes=args.exclude_llm_sizes,
        )

    # 10aa. Leave-one-scale-out validation on the canonical pool
    if args.leave_one_out and len(joint_pool) > 10:
        print("\n" + "=" * 60)
        print("LEAVE-ONE-SCALE-OUT (LLM)")
        print("=" * 60)
        all_results["leave_one_scale_out"] = run_leave_one_scale_out(
            joint_pool,
            base_exclude_llm_sizes=args.exclude_llm_sizes,
            divergence_threshold=args.divergence_threshold,
        )

    # 10ab. Bootstrap CI for additive joint-law parameters
    if args.bootstrap > 0 and len(joint_main_data) > 10:
        print("\n" + "=" * 60)
        print("BOOTSTRAP CI")
        print("=" * 60)
        all_results["joint_bootstrap_ci"] = bootstrap_joint_parameter_ci(
            joint_main_data,
            n_bootstrap=args.bootstrap,
            seed=args.bootstrap_seed,
        )

    # 10b. Sensitivity: all data vs divergent excluded vs 0.5B excluded
    if len(joint_pool) > 10:
        print("\n" + "=" * 60)
        print("JOINT SENSITIVITY (0.5B / DIVERGENCE)")
        print("=" * 60)

        joint_full = analyze_joint(joint_pool, verbose=False, auto_exclude_divergent=False)
        joint_exclude_divergent = analyze_joint(
            joint_pool,
            verbose=False,
            divergence_threshold=args.divergence_threshold,
            auto_exclude_divergent=True,
        )
        joint_exclude_05b = analyze_joint(
            joint_pool,
            verbose=False,
            auto_exclude_divergent=False,
            exclude_llm_sizes=["0.5B"],
        )

        joint_summaries = [
            _summarize_joint_result("all_included", joint_full),
            _summarize_joint_result("exclude_divergent", joint_exclude_divergent),
            _summarize_joint_result("exclude_0.5B", joint_exclude_05b),
        ]
        recommended_variant = _select_recommended_joint_variant(joint_summaries)
        best_by_bic = min(
            (row for row in joint_summaries if row["bic"] is not None),
            key=lambda row: row["bic"],
            default=None,
        )
        best_by_r2 = max(
            (row for row in joint_summaries if row["r_squared"] is not None),
            key=lambda row: row["r_squared"],
            default=None,
        )

        print(f"  {'Variant':<20} {'n':>4} {'Excl':>5} {'BIC':>10} {'R²':>10} {'RMSE':>10} {'MAPE':>10}")
        print("  " + "-" * 80)
        for row in joint_summaries:
            bic = f"{row['bic']:.2f}" if row["bic"] is not None else "N/A"
            r2 = f"{row['r_squared']:.4f}" if row["r_squared"] is not None else "N/A"
            rmse = f"{row['rmse']:.4f}" if row["rmse"] is not None else "N/A"
            mape = f"{row['mape']:.2f}%" if row["mape"] is not None else "N/A"
            print(f"  {row['label']:<20} {row['n_points']:>4} {row['n_excluded_total']:>5} {bic:>10} {r2:>10} {rmse:>10} {mape:>10}")
        if recommended_variant is not None:
            print(f"  Recommended by MAPE: {recommended_variant}")
        if best_by_bic is not None:
            print(f"  Best by BIC: {best_by_bic['label']}")
        if best_by_r2 is not None:
            print(f"  Best by R²: {best_by_r2['label']}")

        all_results["joint_sensitivity"] = {
            "fit_groups": args.fit_groups,
            "divergence_threshold": args.divergence_threshold,
            "canonical_exclude_llm_sizes": args.exclude_llm_sizes,
            "variants": {row["label"]: row for row in joint_summaries},
            "recommended_variant": recommended_variant,
            "best_by_bic": best_by_bic["label"] if best_by_bic is not None else None,
            "best_by_r_squared": best_by_r2["label"] if best_by_r2 is not None else None,
        }

    # 10c. Simplified forms + CV on the canonical fit set
    if len(joint_main_data) > 10:
        print("\n" + "=" * 60)
        print("JOINT LAW: SIMPLIFIED FORMS + CROSS-VALIDATION")
        print("=" * 60)

        na_col = "adapter_params" if "adapter_params" in joint_main_data.columns else "adapter_total"
        jd = _parse_llm_params_col(joint_main_data)
        N_L = jd["llm_params"].values.astype(float)
        D = _compute_effective_d(jd)
        N_A = jd[na_col].values.astype(float)
        T = jd["num_queries"].values.astype(float)
        L = jd["best_val_loss"].values.astype(float)

        simplified_candidates = {
            "simple_NL": (
                joint_simple_nl,
                [1e3, 0.1, 2.5],
                ([0, 0, 0], [np.inf, 2, np.inf]),
            ),
            "NL_D": (
                joint_nl_d,
                [1e3, 0.1, 1e3, 0.1, 2.5],
                ([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.inf]),
            ),
            "NL_NA": (
                joint_nl_na,
                [1e3, 0.1, 1e4, 0.3, 2.5],
                ([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 5, np.inf]),
            ),
            "NL_NA_T": (
                joint_nl_na_t,
                [1e3, 0.1, 1e4, 0.3, 0.001, 1.0, 2.5],
                ([0, 0, 0, 0, 0, 0, 0], [np.inf, 2, np.inf, 5, np.inf, np.inf, np.inf]),
            ),
            "full_additive": (
                joint_additive,
                [1e3, 0.1, 1e3, 0.1, 1e4, 0.3, 1e-9, 0.001, 1.0, 2.5],
                ([0]*9 + [0], [np.inf, 2, np.inf, 2, np.inf, 5, np.inf, np.inf, np.inf, np.inf]),
            ),
            "full_multiplicative": (
                joint_multiplicative,
                [1e3, 0.1, 1e3, 0.1, 1e4, 0.3, 1e-9, 1.0, 0.001, 1.0, 2.5],
                ([0]*10 + [0], [np.inf, 2, np.inf, 2, np.inf, 5, np.inf, np.inf, np.inf, np.inf, np.inf]),
            ),
        }

        print("\n  BIC Comparison (simplified forms):")
        simplified_fits = fit_and_compare((N_L, D, N_A, T), L, simplified_candidates)
        simplified_effective = {fit.name: summarize_effective_params(fit) for fit in simplified_fits}

        print("\n  Effective parameter counts:")
        for fit in simplified_fits:
            report = simplified_effective[fit.name]
            print(f"  {fit.name}: {_format_effective_param_report(report)}")

        print("\n  5-fold Cross-validation:")
        cv_results = cross_validate_fit((N_L, D, N_A, T), L, simplified_candidates, k=5)

        all_results["joint_simplified"] = {
            "bic_fits": simplified_fits,
            "cv_results": cv_results,
            "effective_params": simplified_effective,
            "best_by_bic": simplified_fits[0].name if simplified_fits else None,
            "best_by_cv": cv_results[0]["name"] if cv_results else None,
        }

    # 10d. Per-LLM fixed-effects joint law
    if len(joint_main_data) > 10:
        print("\n" + "=" * 60)
        print("JOINT LAW: PER-LLM FIXED EFFECTS")
        print("=" * 60)

        na_col = "adapter_params" if "adapter_params" in joint_main_data.columns else "adapter_total"
        jd_fe = _parse_llm_params_col(joint_main_data)
        N_L_fe = jd_fe["llm_params"].values.astype(float)
        D_fe = _compute_effective_d(jd_fe)
        N_A_fe = jd_fe[na_col].values.astype(float)
        T_fe = jd_fe["num_queries"].values.astype(float)
        L_fe = jd_fe["best_val_loss"].values.astype(float)
        llm_names_fe = jd_fe["llm_name"].values

        fe_result = fit_joint_per_llm_eps(N_L_fe, D_fe, N_A_fe, T_fe, L_fe, llm_names_fe)

        print(f"  N = {fe_result['n']}, k = {fe_result['k']}")
        print(f"  R² = {fe_result['r_squared']:.4f}, MAPE = {fe_result['mape']:.2f}%")
        print(f"  BIC = {fe_result['bic']:.2f}, AIC = {fe_result['aic']:.2f}")
        print(f"  Shared params:")
        for k_name, v in fe_result["shared_params"].items():
            print(f"    {k_name} = {v:.6g}")
        print(f"  Per-LLM eps:")
        for llm, eps_val in fe_result["per_llm_eps"].items():
            short = llm.split("-")[-1]
            print(f"    {short}: ε = {eps_val:+.4f}")

        # Compare with standard joint law
        if "joint" in all_results and "fits" in all_results["joint"] and all_results["joint"]["fits"]:
            std_r2 = all_results["joint"]["fits"][0].r_squared
            delta_r2 = fe_result["r_squared"] - std_r2
            print(f"\n  Δ R² vs standard joint: {delta_r2:+.4f}")
            print(f"  Standard joint R²: {std_r2:.4f} → Per-LLM eps R²: {fe_result['r_squared']:.4f}")

        all_results["joint_per_llm_eps"] = fe_result

    # 10e. Noise floor analysis (requires seed stability data)
    if "seed_stability" in all_results and len(joint_main_data) > 10:
        print("\n" + "=" * 60)
        print("NOISE FLOOR ANALYSIS")
        print("=" * 60)
        total_var_fit = float(np.var(L_fe, ddof=1))
        noise_floor = compute_noise_floor(
            all_results["seed_stability"],
            total_var_fit,
            exclude_05b=True,
            verbose=True,
        )

        # Add noise-adjusted R² for all joint models
        max_r2 = noise_floor.get("max_r2_median", 1.0)
        noise_adj = {"max_r2": max_r2}
        if "joint" in all_results and "fits" in all_results["joint"] and all_results["joint"]["fits"]:
            std_r2 = all_results["joint"]["fits"][0].r_squared
            noise_adj["standard_joint"] = {
                "r_squared": float(std_r2),
                "noise_adjusted_r2": float(std_r2 / max_r2) if max_r2 > 0 else None,
            }
        if "joint_per_llm_eps" in all_results:
            fe_r2 = all_results["joint_per_llm_eps"]["r_squared"]
            noise_adj["per_llm_eps"] = {
                "r_squared": float(fe_r2),
                "noise_adjusted_r2": float(fe_r2 / max_r2) if max_r2 > 0 else None,
            }

        noise_floor["noise_adjusted_r2"] = noise_adj
        if max_r2 > 0:
            print(f"\n  Noise-adjusted R² (relative to ceiling {max_r2:.4f}):")
            for model_name, data in noise_adj.items():
                if isinstance(data, dict) and "noise_adjusted_r2" in data:
                    print(f"    {model_name}: R²={data['r_squared']:.4f} → "
                          f"adjusted={data['noise_adjusted_r2']:.4f}")

        all_results["noise_floor"] = noise_floor

    # 11. Extrapolation validation (predict G7 from main joint fit)
    g7 = valid[valid["run_name"].str.startswith("g7_")]
    if len(g7) > 0 and "joint" in all_results:
        print("\n" + "=" * 60)
        print("EXTRAPOLATION (G7)")
        print("=" * 60)
        all_results["extrapolation"] = validate_extrapolation(joint_main_data, g7, all_results["joint"])

    # 12. BIC summary table
    print("\n" + "=" * 60)
    print("BIC SUMMARY")
    print("=" * 60)
    bic_table = generate_bic_summary(all_results)

    if bic_table:
        bic_df = pd.DataFrame(bic_table)
        bic_path = Path(args.output_dir) / "bic_summary.csv"
        bic_df.to_csv(bic_path, index=False)
        print(f"\nBIC table saved: {bic_path}")

    save_results_json(all_results, str(Path(args.output_dir) / "scaling_fit_results.json"))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
