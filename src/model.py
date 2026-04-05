"""
model.py

Step 3 — Fit EGARCH(1,1) per Currency

Fits a Nelson (1991) EGARCH(1,1) model with Student-t errors to the
daily log-return series of each BRICS currency.  The fitted results
(arch ARCHModelResult objects, conditional volatilities, parameters,
AIC/BIC) are returned as a dictionary keyed by currency code and
consumed downstream by analysis.py.

EGARCH(1,1) variance equation:

    ln(σ²_t) = ω
             + α · [|z_{t-1}| − E|z_{t-1}|]    ← magnitude effect
             + γ · z_{t-1}                        ← leverage / sign effect
             + β · ln(σ²_{t-1})                  ← persistence

    z_t = ε_t / σ_t   (standardised residuals)

Key advantages over standard GARCH:
  - γ term captures asymmetry: depreciations spike vol more than appreciations
  - Log-variance never goes negative — no positivity constraints needed

Usage (standalone):
    python src/model.py

Imported by:
    notebooks/exploratory_analysis.ipynb
    src/analysis.py (via main pipeline)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from arch import arch_model

from data_cleaning import CURRENCIES, START_DATE, END_DATE

# SHARED CONFIGURATION

DIST   = "t"    # Student-t errors — fat tails in FX returns
ANNUAL = 252    # trading days per year

# HELPER

def get_param(params: pd.Series, *keys) -> float:
    for k in keys:
        if k in params.index:
            return params[k]
    return float("nan")

# STEP 3 — FIT EGARCH(1,1) PER CURRENCY

def fit_egarch(returns: pd.DataFrame) -> dict:
    print("\n[3] Fitting EGARCH(1,1) models...")
    results = {}

    for col in returns.columns:
        r  = returns[col].dropna()
        am = arch_model(r, vol="EGARCH", p=1, q=1,
                        dist=DIST, mean="Constant")
        res = am.fit(disp="off", options={"maxiter": 500},
                     show_warning=False)

        cond_vol_daily = res.conditional_volatility          # σ_t  (daily %)
        cond_vol_ann   = cond_vol_daily * np.sqrt(ANNUAL)   # annualised %

        results[col] = {
            "result"        : res,
            "cond_vol_daily": cond_vol_daily,
            "cond_vol_ann"  : cond_vol_ann,
            "params"        : res.params,
            "aic"           : res.aic,
            "bic"           : res.bic,
        }

        p = res.params
        print(f"\n  --- {col} ---")
        print(f"    ω (omega)  = {p['omega']:.6f}")
        print(f"    α (alpha)  = {get_param(p,'alpha[1]','alpha'):.6f}"
              f"   magnitude effect")
        print(f"    γ (gamma)  = {get_param(p,'gamma[1]','gamma'):.6f}"
              f"   leverage effect")
        print(f"    β (beta)   = {get_param(p,'beta[1]','beta'):.6f}"
              f"   persistence")
        print(f"    AIC={res.aic:.1f}   BIC={res.bic:.1f}")

    return results

# STANDALONE ENTRY POINT

if __name__ == "__main__":
    from data_cleaning import run as fetch_data

    _, _, returns = fetch_data(save=False)
    egarch_results = fit_egarch(returns)
    print("\n[Done] model.py complete.")
