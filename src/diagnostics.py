"""
diagnostics.py
==============
Pre-fit and post-fit diagnostic tests for the BRICS EGARCH analysis.

Pre-fit diagnostics (run before model estimation):
  - Descriptive statistics: mean, std, skewness, excess kurtosis
  - Jarque-Bera test  → tests normality; we expect rejection (fat tails in FX)
  - ARCH-LM test      → confirms conditional heteroskedasticity; justifies GARCH family

Post-fit diagnostics (run on standardised residuals after fitting):
  - Ljung-Box on z_t   → tests mean-equation adequacy (no residual autocorrelation)
  - Ljung-Box on z²_t  → tests variance-equation adequacy (no remaining ARCH effects)
  - Leverage γ significance
  - Persistence (α+β) and shock half-life

Usage:
    python src/diagnostics.py          # requires fx_returns.csv in data/processed/
    # or import directly:
    from src.diagnostics import run_preflight_diagnostics, post_fit_diagnostics
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera


# ---------------------------------------------------------------------------
# Pre-fit diagnostics
# ---------------------------------------------------------------------------

def run_preflight_diagnostics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics and pre-fit tests for each return series.

    Tests performed
    ---------------
    Jarque-Bera (JB):
        H0: series is normally distributed.
        We expect rejection because FX returns exhibit excess kurtosis and
        skewness, motivating Student-t errors in the EGARCH model.

    ARCH-LM (Lagrange Multiplier, 10 lags):
        H0: no ARCH effects (constant conditional variance).
        Rejection justifies the use of GARCH-family models.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns (in %) with one column per currency.

    Returns
    -------
    diag : pd.DataFrame
        Table of descriptive statistics and test p-values.
    """
    print("\n[PRE-FIT] Running pre-fit diagnostics (Jarque-Bera + ARCH-LM)...")
    rows = []
    for col in returns.columns:
        r = returns[col].dropna().values
        jb_stat, jb_p, _, _ = jarque_bera(r)
        lm_stat, lm_p, _, _ = het_arch(r, nlags=10)
        rows.append({
            "Currency":     col,
            "Mean (%)":     np.mean(r),
            "Std (%)":      np.std(r),
            "Skewness":     stats.skew(r),
            "Ex. Kurtosis": stats.kurtosis(r),
            "JB p-value":   jb_p,
            "ARCH-LM p":    lm_p,
            "ARCH Effect":  "YES ***" if lm_p < 0.05 else "NO",
        })
    diag = pd.DataFrame(rows).set_index("Currency")
    print(diag.to_string(float_format="{:.4f}".format))
    return diag


# ---------------------------------------------------------------------------
# Post-fit diagnostics
# ---------------------------------------------------------------------------

def post_fit_diagnostics(egarch_results: dict) -> pd.DataFrame:
    """
    Validate each fitted EGARCH model using standardised residuals.

    Standardised residuals are z_t = ε_t / σ_t. A well-specified model
    should produce z_t with no remaining serial correlation in either
    levels or squares.

    Checks
    ------
    - StdResid mean ≈ 0, std ≈ 1  (sanity check on residuals)
    - Ljung-Box on z_t    (10 lags) — mean equation adequacy
    - Ljung-Box on z²_t   (10 lags) — variance equation adequacy
    - Leverage γ          (sign of γ: negative = depreciations spike vol more)
    - Persistence (α+β)   (>1 indicates explosive variance process)
    - Half-life (days)    (days for shock to decay by 50%)

    Parameters
    ----------
    egarch_results : dict
        Output of egarch_model.fit_egarch(); keys are currency codes,
        values contain 'result' (arch ModelResult).

    Returns
    -------
    diag : pd.DataFrame
        Post-fit diagnostic table.
    """
    print("\n[POST-FIT] Running post-fit diagnostics on standardised residuals...")
    rows = []

    def _get(p, *keys):
        for k in keys:
            if k in p.index:
                return p[k]
        return float("nan")

    for col, d in egarch_results.items():
        res       = d["result"]
        p         = res.params
        std_resid = (res.resid / res.conditional_volatility).dropna()

        lb_lev = acorr_ljungbox(std_resid,    lags=10, return_df=True)
        lb_sq  = acorr_ljungbox(std_resid**2, lags=10, return_df=True)

        beta        = abs(_get(p, "beta[1]",  "beta"))
        alpha       = abs(_get(p, "alpha[1]", "alpha"))
        persistence = beta + alpha
        half_life   = (np.log(0.5) / np.log(beta)
                       if 0 < beta < 1 else np.inf)

        rows.append({
            "Currency":           col,
            "StdResid Mean":      std_resid.mean(),
            "StdResid Std":       std_resid.std(),
            "LB(10) p (z_t)":     lb_lev["lb_pvalue"].iloc[-1],
            "LB(10) p (z²_t)":   lb_sq["lb_pvalue"].iloc[-1],
            "Leverage γ":         _get(p, "gamma[1]", "gamma"),
            "Persistence (α+β)":  persistence,
            "Half-life (days)":   half_life,
        })

    diag = pd.DataFrame(rows).set_index("Currency")
    print(diag.to_string(float_format="{:.4f}".format))
    return diag


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    returns_path = os.path.join(data_dir, "fx_returns.csv")

    if not os.path.exists(returns_path):
        raise FileNotFoundError(
            f"'{returns_path}' not found. "
            "Run data_download.py first, or run egarch_analysis.py for the full pipeline."
        )

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    pre_diag = run_preflight_diagnostics(returns)
    print("\n[PRE-FIT] Complete. Post-fit diagnostics require EGARCH results.")
    print("          Run egarch_analysis.py for the full pipeline.")
