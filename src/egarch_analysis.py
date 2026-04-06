"""
egarch_analysis.py
==================
Full BRICS UNIT EGARCH pipeline — single entry point.

Runs all nine steps in sequence and saves outputs to results/.

Steps
-----
1.  Download FX data (BRL, RUB, INR, CNY, ZAR) and gold from Yahoo Finance
2.  Pre-fit diagnostics  (Jarque-Bera, ARCH-LM)
3.  Fit EGARCH(1,1) models per currency (Student-t errors)
4.  Post-fit diagnostics (Ljung-Box on z_t and z²_t)
5.  Compute volatility metrics (mean conditional variance, annualised vol,
    unconditional variance, persistence, half-life)
6.  Derive inverse-variance basket weights
7.  Simulate the BRICS Currency Unit (UNIT) index
8.  Generate all plots
9.  Export all tables to results/tables/ and data/processed/

Usage
-----
    python src/egarch_analysis.py

Requirements
------------
    pip install -r requirements.txt
    Internet connection required (FX and gold data fetched live).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project root to path so sub-modules resolve correctly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.data_download  import download_fx, download_gold, CURRENCIES, START_DATE, END_DATE
from src.diagnostics    import run_preflight_diagnostics, post_fit_diagnostics
from src.egarch_model   import fit_egarch, compute_volatility_metrics, derive_weight_constraints
from src.unit_simulation import (
    simulate_unit_value,
    plot_conditional_volatility,
    plot_stability_ranking,
    plot_unit_value,
    plot_individual_currencies,
    plot_egarch_diagnostics,
)

ANNUAL    = 252
FIG_DIR   = os.path.join(ROOT, "results", "figures")
TABLE_DIR = os.path.join(ROOT, "results", "tables")
PROC_DIR  = os.path.join(ROOT, "data", "processed")


def export_all(metrics: pd.DataFrame,
               constraints: dict,
               pre_diag: pd.DataFrame,
               post_diag: pd.DataFrame) -> None:
    """Save all result tables to results/tables/ and data/processed/."""
    for d in [TABLE_DIR, PROC_DIR]:
        os.makedirs(d, exist_ok=True)

    metrics.to_csv(os.path.join(TABLE_DIR, "egarch_volatility_metrics.csv"))
    metrics.to_csv(os.path.join(PROC_DIR,  "egarch_volatility_metrics.csv"))
    pre_diag.to_csv(os.path.join(TABLE_DIR, "prefit_diagnostics.csv"))
    pre_diag.to_csv(os.path.join(PROC_DIR,  "prefit_diagnostics.csv"))
    post_diag.to_csv(os.path.join(TABLE_DIR, "postfit_diagnostics.csv"))
    post_diag.to_csv(os.path.join(PROC_DIR,  "postfit_diagnostics.csv"))

    wt_df = pd.DataFrame({
        "Currency":       constraints["ordered_currencies"],
        "Stability Rank": range(1, len(constraints["ordered_currencies"]) + 1),
        "Mean Cond Var":  [constraints["mean_cond_var"][c]
                           for c in constraints["ordered_currencies"]],
        "IV Weight":      [constraints["iv_weights"][c]
                           for c in constraints["ordered_currencies"]],
        "IV Weight (%)":  [constraints["iv_weights"][c] * 100
                           for c in constraints["ordered_currencies"]],
    })
    wt_df.to_csv(os.path.join(TABLE_DIR, "egarch_weights.csv"), index=False)
    wt_df.to_csv(os.path.join(PROC_DIR,  "egarch_weights.csv"), index=False)

    print(f"\n[EXPORT] Tables saved to:")
    print(f"  {TABLE_DIR}/")
    print(f"  {PROC_DIR}/")
    for f in ["egarch_volatility_metrics.csv", "prefit_diagnostics.csv",
              "postfit_diagnostics.csv", "egarch_weights.csv"]:
        print(f"    {f}")


def print_final_summary(constraints: dict) -> None:
    """Print a formatted summary of the EGARCH-derived UNIT weights."""
    ordered = constraints["ordered_currencies"]
    weights = constraints["iv_weights"]

    print("\n" + "=" * 65)
    print("  FINAL WEIGHT CONSTRAINTS FOR BRICS UNIT BASKET")
    print("=" * 65)
    print("\n  UNIT = 0.6 × [Σ wᵢ · FXᵢ] + 0.4 × Gold")
    print("\n  EGARCH-derived ordering constraint:")
    print("    " + " ≥ ".join([f"w_{c}" for c in ordered]))
    print("\n  Inverse-variance weights (analytically derived solution):")
    for c in ordered:
        w = weights[c]
        bar = "█" * int(w * 50)
        print(f"    w_{c} = {w:.4f}  ({w * 100:.2f}%)  {bar}")
    print(f"\n  Σ weights = {weights.sum():.6f}  ✓")
    print("\n  Note: RUB weight ≈ 0 reflects the post-2022 sanctions")
    print("  structural break — an empirical result, not an assumption.\n")


def main():
    print("=" * 65)
    print("  BRICS CURRENCY UNIT — EGARCH VOLATILITY PIPELINE")
    print("  Singhal, Sanghani & Jain (2025–26), CFM, BITS Pilani")
    print("=" * 65)

    os.makedirs(FIG_DIR,   exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(PROC_DIR,  exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Data download
    # ------------------------------------------------------------------
    prices, returns = download_fx(CURRENCIES, START_DATE, END_DATE)
    gold_prices     = download_gold(START_DATE, END_DATE)

    # Save raw downloads
    prices.to_csv(os.path.join(PROC_DIR, "fx_prices.csv"))
    returns.to_csv(os.path.join(PROC_DIR, "fx_returns.csv"))
    gold_prices.to_csv(os.path.join(PROC_DIR, "gold_prices.csv"), header=True)

    # ------------------------------------------------------------------
    # Step 2 — Pre-fit diagnostics
    # ------------------------------------------------------------------
    pre_diag = run_preflight_diagnostics(returns)

    # ------------------------------------------------------------------
    # Step 3 — Fit EGARCH(1,1)
    # ------------------------------------------------------------------
    egarch_results = fit_egarch(returns, dist="t")

    # ------------------------------------------------------------------
    # Step 4 — Post-fit diagnostics
    # ------------------------------------------------------------------
    post_diag = post_fit_diagnostics(egarch_results)

    # ------------------------------------------------------------------
    # Step 5 — Volatility metrics & stability ranking
    # ------------------------------------------------------------------
    metrics = compute_volatility_metrics(egarch_results)

    # ------------------------------------------------------------------
    # Step 6 — Inverse-variance weights
    # ------------------------------------------------------------------
    constraints = derive_weight_constraints(metrics)

    # ------------------------------------------------------------------
    # Step 7 — Simulate UNIT index
    # ------------------------------------------------------------------
    unit_df = simulate_unit_value(prices, gold_prices,
                                   weights=constraints["iv_weights"])

    # ------------------------------------------------------------------
    # Step 8 — Plots
    # ------------------------------------------------------------------
    print("\n[PLOTS] Generating figures...")

    plot_individual_currencies(
        prices,
        save_path=os.path.join(FIG_DIR, "brics_individual_fx.png")
    )
    plot_conditional_volatility(
        egarch_results, metrics,
        save_path=os.path.join(FIG_DIR, "brics_conditional_vol.png")
    )
    plot_stability_ranking(
        metrics, constraints,
        save_path=os.path.join(FIG_DIR, "brics_stability_ranking.png")
    )
    plot_unit_value(
        unit_df,
        save_path=os.path.join(FIG_DIR, "brics_unit_index.png")
    )
    plot_egarch_diagnostics(
        egarch_results,
        save_prefix=os.path.join(FIG_DIR, "brics_diag")
    )

    # ------------------------------------------------------------------
    # Step 9 — Export results
    # ------------------------------------------------------------------
    export_all(metrics, constraints, pre_diag, post_diag)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print_final_summary(constraints)

    print("[PIPELINE] Complete. All outputs in results/ and data/processed/")


if __name__ == "__main__":
    main()
