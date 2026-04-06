"""
egarch_model.py
===============
Fits EGARCH(1,1) models with Student-t errors to each BRICS currency return
series and extracts volatility metrics and inverse-variance basket weights.

EGARCH(1,1) variance equation (Nelson, 1991):

    ln(σ²_t) = ω
             + α · [|z_{t-1}| − E|z_{t-1}|]    ← magnitude effect
             + γ · z_{t-1}                        ← leverage / sign effect
             + β · ln(σ²_{t-1})                  ← persistence

    z_t = ε_t / σ_t  (standardised residuals, Student-t distributed)

Why EGARCH over standard GARCH?
  1. γ parameter captures asymmetry: depreciations raise volatility by more
     than equivalent appreciations — critical for emerging market FX.
  2. Log-variance specification ensures σ²_t > 0 without non-negativity
     constraints on parameters.
  3. Student-t errors handle the fat tails observed in BRICS FX returns
     (excess kurtosis up to 781 for RUB, see Table 1 in the paper).

Usage:
    python src/egarch_model.py        # requires fx_returns.csv in data/processed/
    # or import:
    from src.egarch_model import fit_egarch, compute_volatility_metrics, derive_weight_constraints
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from arch import arch_model

ANNUAL = 252   # trading days per year


# ---------------------------------------------------------------------------
# EGARCH fitting
# ---------------------------------------------------------------------------

def fit_egarch(returns: pd.DataFrame,
               dist: str = "t") -> dict:
    """
    Fit EGARCH(1,1) with Student-t errors to each column of `returns`.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns in %, one column per currency.
    dist : str
        Error distribution for arch_model. Default "t" (Student-t).

    Returns
    -------
    results : dict
        Keys are currency codes. Each value is a dict with:
          "result"         — arch ModelResult object
          "cond_vol_daily" — daily conditional volatility series (%)
          "cond_vol_ann"   — annualised conditional volatility series (%)
          "params"         — estimated parameter Series
          "aic"            — Akaike information criterion
          "bic"            — Bayesian information criterion
    """
    print("\n[EGARCH] Fitting EGARCH(1,1) models (Student-t errors)...")
    results = {}

    def _get(p, *keys):
        for k in keys:
            if k in p.index:
                return p[k]
        return float("nan")

    for col in returns.columns:
        r  = returns[col].dropna()
        am = arch_model(r, vol="EGARCH", p=1, q=1,
                        dist=dist, mean="Constant")
        res = am.fit(disp="off",
                     options={"maxiter": 500},
                     show_warning=False)

        cond_vol_daily = res.conditional_volatility        # σ_t  (daily %)
        cond_vol_ann   = cond_vol_daily * np.sqrt(ANNUAL)  # annualised %

        results[col] = {
            "result":          res,
            "cond_vol_daily":  cond_vol_daily,
            "cond_vol_ann":    cond_vol_ann,
            "params":          res.params,
            "aic":             res.aic,
            "bic":             res.bic,
        }

        p = res.params
        print(f"\n  --- {col} ---")
        print(f"    ω (omega) = {p['omega']:+.6f}")
        print(f"    α (alpha) = {_get(p,'alpha[1]','alpha'):+.6f}   "
              f"[magnitude effect]")
        print(f"    γ (gamma) = {_get(p,'gamma[1]','gamma'):+.6f}   "
              f"[leverage effect; < 0 → depreciations spike vol]")
        print(f"    β (beta)  = {_get(p,'beta[1]', 'beta') :+.6f}   "
              f"[persistence]")
        print(f"    AIC = {res.aic:.1f}   BIC = {res.bic:.1f}")

    return results


# ---------------------------------------------------------------------------
# Volatility metrics
# ---------------------------------------------------------------------------

def compute_volatility_metrics(egarch_results: dict) -> pd.DataFrame:
    """
    Extract stability metrics from fitted EGARCH models and rank currencies.

    Metrics extracted per currency
    ------------------------------
    1. Mean Conditional Variance (daily %²)  ← PRIMARY ranking criterion
       = average of σ²_t across the full sample
       A lower value indicates a more stable currency.

    2. Mean Conditional Volatility (ann. %)
       = sqrt(mean_cond_var × 252)
       Directly comparable to annualised realised volatility.

    3. Unconditional (long-run) Variance
       For EGARCH: E[ln σ²] = ω/(1−β)  →  σ²_LR = exp(ω/(1−β))
       Gives the theoretical steady-state variance level.

    4. Persistence = |α| + |β|
       Measures how long a volatility shock persists.
       Values > 1 indicate an explosive process (e.g., RUB post-2022).

    5. Half-life (days)
       = ln(0.5) / ln(|β|)
       Days for a shock to decay to half its initial impact.

    Stability rank is assigned in ascending order of Mean Conditional Variance
    (rank 1 = most stable = INR in our sample).

    Parameters
    ----------
    egarch_results : dict
        Output of fit_egarch().

    Returns
    -------
    metrics : pd.DataFrame
        Sorted from most stable (rank 1) to least stable, with all metrics.
    """
    print("\n[METRICS] Computing volatility metrics and stability ranking...")

    def _get(p, *keys):
        for k in keys:
            if k in p.index:
                return p[k]
        return float("nan")

    rows = []
    for col, d in egarch_results.items():
        res   = d["result"]
        p     = res.params

        cond_var_daily    = res.conditional_volatility ** 2
        mean_cond_var     = cond_var_daily.mean()
        mean_cond_vol_ann = np.sqrt(mean_cond_var * ANNUAL)

        beta        = abs(_get(p, "beta[1]",  "beta"))
        alpha       = abs(_get(p, "alpha[1]", "alpha"))
        omega       = p["omega"]
        persistence = beta + alpha
        half_life   = (np.log(0.5) / np.log(beta)
                       if 0 < beta < 1 else np.inf)

        if abs(1 - beta) > 1e-6:
            unconditional_var = np.exp(omega / (1 - beta))
        else:
            unconditional_var = np.nan   # near unit-root: undefined

        rows.append({
            "Currency":                  col,
            "Mean Cond. Var (daily %²)": mean_cond_var,
            "Mean Cond. Vol (ann. %)":   mean_cond_vol_ann,
            "Unconditional Var":         unconditional_var,
            "Persistence (α+β)":         persistence,
            "Half-life (days)":          half_life,
        })

    metrics = pd.DataFrame(rows).set_index("Currency")
    metrics["Stability Rank"] = (
        metrics["Mean Cond. Var (daily %²)"]
        .rank(ascending=True)
        .astype(int)
    )
    metrics = metrics.sort_values("Stability Rank")

    print("\n  Volatility metrics — sorted most stable → least stable:")
    print(metrics.to_string(float_format="{:.4f}".format))
    return metrics


# ---------------------------------------------------------------------------
# Inverse-variance weight derivation
# ---------------------------------------------------------------------------

def derive_weight_constraints(metrics: pd.DataFrame) -> dict:
    """
    Translate EGARCH stability ranking into inverse-variance basket weights.

    Inverse-variance weighting
    --------------------------
    Given conditional variances σ²_i (mean over the sample), the
    closed-form minimum-variance weights assuming a diagonal covariance
    matrix (zero cross-currency correlations) are:

        raw_w_i = 1 / σ²_i
        w_i     = raw_w_i / Σ_j raw_w_j      (normalised, Σ w_i = 1)

    This is a natural stability-focused allocation: currencies with lower
    volatility receive proportionally higher basket weights.

    The ordering constraint
    -----------------------
    The stability rank implies: w[rank 1] ≥ w[rank 2] ≥ ... ≥ w[rank 5],
    which is automatically satisfied by inverse-variance weights when
    currencies are ranked by ascending mean conditional variance.

    Parameters
    ----------
    metrics : pd.DataFrame
        Output of compute_volatility_metrics(); must be sorted by rank.

    Returns
    -------
    constraints : dict with keys:
        "ordered_currencies" — list of currency codes, best → worst
        "iv_weights"         — pd.Series of normalised IV weights
        "mean_cond_var"      — pd.Series of mean conditional variances
    """
    print("\n[WEIGHTS] Deriving inverse-variance basket weights...")

    ordered   = metrics.index.tolist()
    mean_vars = metrics["Mean Cond. Var (daily %²)"]

    inv_var    = 1.0 / mean_vars
    iv_weights = inv_var / inv_var.sum()
    metrics    = metrics.copy()
    metrics["IV Weight"] = iv_weights

    print("\n  Inverse-variance weights (normalised, Σ = 1):")
    for cur in ordered:
        print(f"    w_{cur} = {iv_weights[cur]:.4f}   "
              f"({iv_weights[cur] * 100:.2f}%)")
    print(f"    Sum    = {iv_weights.sum():.6f}  ✓")

    print("\n  Ordering constraint (from EGARCH stability rank):")
    print("    " + " ≥ ".join([f"w_{c}" for c in ordered]))

    return {
        "ordered_currencies": ordered,
        "iv_weights":         iv_weights,
        "mean_cond_var":      mean_vars,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_metrics(metrics: pd.DataFrame,
                   constraints: dict,
                   out_dir: str = None) -> None:
    """Save volatility metrics and weights tables to CSV."""
    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed"
        )
    os.makedirs(out_dir, exist_ok=True)

    metrics.to_csv(os.path.join(out_dir, "egarch_volatility_metrics.csv"))

    wt_df = pd.DataFrame({
        "Currency":       constraints["ordered_currencies"],
        "Stability Rank": range(1, 6),
        "Mean Cond Var":  [constraints["mean_cond_var"][c]
                           for c in constraints["ordered_currencies"]],
        "IV Weight":      [constraints["iv_weights"][c]
                           for c in constraints["ordered_currencies"]],
        "IV Weight (%)":  [constraints["iv_weights"][c] * 100
                           for c in constraints["ordered_currencies"]],
    })
    wt_df.to_csv(os.path.join(out_dir, "egarch_weights.csv"), index=False)
    print(f"\n[METRICS] Exported to '{out_dir}/'")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed"
    )
    returns_path = os.path.join(data_dir, "fx_returns.csv")

    if not os.path.exists(returns_path):
        raise FileNotFoundError(
            f"'{returns_path}' not found. "
            "Run data_download.py first, or egarch_analysis.py for the full pipeline."
        )

    returns        = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    egarch_results = fit_egarch(returns)
    metrics        = compute_volatility_metrics(egarch_results)
    constraints    = derive_weight_constraints(metrics)
    export_metrics(metrics, constraints)
    print("\n[EGARCH] Model fitting and metric extraction complete.")
