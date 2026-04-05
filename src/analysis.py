"""
analysis.py

Steps 2, 4–9 — Diagnostics, Metrics, Weights, Simulation, Plots, Export

Runs the full analytical pipeline downstream of EGARCH fitting:

  Step 2  — Pre-fit diagnostics (ARCH-LM, Jarque-Bera)
  Step 4  — Post-fit diagnostics (Ljung-Box on z_t and z²_t)
  Step 5  — Volatility metrics and stability ranking
  Step 6  — Weight constraints from EGARCH ranking
  Step 7  — BRICS UNIT index simulation
  Step 8  — Plots (conditional vol, stability ranking, UNIT index,
              per-currency diagnostic grids)
  Step 9  — Export results to results/tables/ and results/figures/

Usage (full pipeline from scratch):
    python src/analysis.py

Imported by:
    notebooks/exploratory_analysis.ipynb
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf

from data_cleaning import CURRENCIES, GOLD_TICKER, START_DATE, END_DATE
from model import ANNUAL, get_param

FIGURES_DIR = os.path.join("results", "figures")
TABLES_DIR  = os.path.join("results", "tables")

# STEP 2 — PRE-FIT DIAGNOSTICS

def run_preflight_diagnostics(returns: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] Pre-fit diagnostics (ARCH-LM + Normality tests)...")
    rows = []
    for col in returns.columns:
        r = returns[col].dropna().values
        jb_stat, jb_p, _, _ = jarque_bera(r)
        lm_stat, lm_p, _, _ = het_arch(r, nlags=10)
        rows.append({
            "Currency"    : col,
            "Mean (%)"    : np.mean(r),
            "Std (%)"     : np.std(r),
            "Skewness"    : stats.skew(r),
            "Ex.Kurtosis" : stats.kurtosis(r),
            "JB p-value"  : jb_p,
            "ARCH-LM p"   : lm_p,
            "ARCH effect" : "YES ***" if lm_p < 0.05 else "NO",
        })
    diag = pd.DataFrame(rows).set_index("Currency")
    print(diag.to_string(float_format="{:.4f}".format))
    return diag

# STEP 4 — POST-FIT DIAGNOSTICS

def post_fit_diagnostics(egarch_results: dict) -> pd.DataFrame:
    print("\n[4] Post-fit diagnostics on standardised residuals...")
    rows = []

    for col, d in egarch_results.items():
        res       = d["result"]
        p         = res.params
        std_resid = (res.resid / res.conditional_volatility).dropna()

        lb_lev = acorr_ljungbox(std_resid,    lags=10, return_df=True)
        lb_sq  = acorr_ljungbox(std_resid**2, lags=10, return_df=True)

        beta        = abs(get_param(p, "beta[1]",  "beta"))
        alpha       = abs(get_param(p, "alpha[1]", "alpha"))
        persistence = beta + alpha
        half_life   = np.log(0.5) / np.log(beta) if 0 < beta < 1 else np.inf

        rows.append({
            "Currency"           : col,
            "StdResid Mean"      : std_resid.mean(),
            "StdResid Std"       : std_resid.std(),
            "LB(10) p  (z_t)"    : lb_lev["lb_pvalue"].iloc[-1],
            "LB(10) p  (z²_t)"   : lb_sq["lb_pvalue"].iloc[-1],
            "Leverage γ"         : get_param(p, "gamma[1]", "gamma"),
            "Persistence (α+β)"  : persistence,
            "Half-life (days)"   : half_life,
        })

    diag = pd.DataFrame(rows).set_index("Currency")
    print(diag.to_string(float_format="{:.4f}".format))
    return diag

# STEP 5 — VOLATILITY METRICS & STABILITY RANKING

def compute_volatility_metrics(egarch_results: dict) -> pd.DataFrame:
    print("\n[5] Computing volatility metrics and stability ranking...")

    rows = []
    for col, d in egarch_results.items():
        res = d["result"]
        p   = res.params

        cond_var_daily    = res.conditional_volatility ** 2
        mean_cond_var     = cond_var_daily.mean()
        mean_cond_vol_ann = np.sqrt(mean_cond_var * ANNUAL)

        beta        = abs(get_param(p, "beta[1]",  "beta"))
        alpha       = abs(get_param(p, "alpha[1]", "alpha"))
        omega       = p["omega"]
        persistence = beta + alpha
        half_life   = np.log(0.5) / np.log(beta) if 0 < beta < 1 else np.inf

        unconditional_var = (
            np.exp(omega / (1 - beta)) if abs(1 - beta) > 1e-6 else np.nan
        )

        rows.append({
            "Currency"                 : col,
            "Mean Cond. Var (daily %²)": mean_cond_var,
            "Mean Cond. Vol (ann. %)"  : mean_cond_vol_ann,
            "Unconditional Var"        : unconditional_var,
            "Persistence (α+β)"        : persistence,
            "Half-life (days)"         : half_life,
        })

    metrics = pd.DataFrame(rows).set_index("Currency")
    metrics["Stability Rank"] = (
        metrics["Mean Cond. Var (daily %²)"]
        .rank(ascending=True).astype(int)
    )
    metrics = metrics.sort_values("Stability Rank")

    print("\n  Volatility Metrics — sorted most stable → least stable:")
    print(metrics.to_string(float_format="{:.4f}".format))
    return metrics

# STEP 6 — WEIGHT CONSTRAINTS FROM EGARCH RANKING

def derive_weight_constraints(metrics: pd.DataFrame) -> dict:
    print("\n[6] Deriving weight constraints...")

    ordered   = metrics.index.tolist()
    mean_vars = metrics["Mean Cond. Var (daily %²)"]
    inv_var   = 1.0 / mean_vars
    iv_weights = inv_var / inv_var.sum()
    metrics["IV Weight"] = iv_weights

    print("\n  Inverse-variance weights (seed / reference solution):")
    for cur in ordered:
        print(f"    w_{cur} = {iv_weights[cur]:.4f}   ({iv_weights[cur]*100:.1f}%)")
    print(f"    Sum    = {iv_weights.sum():.6f}  ✓")

    print("\n  Ordering constraint (from EGARCH stability rank):")
    print("    " + " >= ".join([f"w_{c}" for c in ordered]))

    return {
        "ordered_currencies": ordered,
        "iv_weights"        : iv_weights,
        "mean_cond_var"     : mean_vars,
    }

# STEP 7 — BRICS UNIT SIMULATION

def simulate_unit_value(prices: pd.DataFrame,
                        gold_prices: pd.Series,
                        weights: pd.Series) -> pd.DataFrame:
    print("\n[7] Simulating BRICS UNIT value...")

    gold_aligned = gold_prices.reindex(prices.index, method="ffill").dropna()
    common_idx   = prices.index.intersection(gold_aligned.index)
    fx           = prices.loc[common_idx]
    gold         = gold_aligned.loc[common_idx]

    fx_norm   = fx   / fx.iloc[0]
    gold_norm = gold / gold.iloc[0]

    basket = (fx_norm * weights).sum(axis=1)
    unit   = 0.6 * basket + 0.4 * gold_norm

    result   = pd.DataFrame({"UNIT": unit, "Basket": basket, "Gold": gold_norm})
    unit_ret = np.log(unit / unit.shift(1)).dropna() * 100
    print(f"    UNIT annualised vol  : {unit_ret.std() * np.sqrt(ANNUAL):.4f}%")
    print(f"    UNIT total return    : {(unit.iloc[-1] - 1)*100:.2f}%")
    return result

# STEP 8 — PLOTS

def plot_conditional_volatility(egarch_results: dict,
                                metrics: pd.DataFrame,
                                save_path: str = None) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        "EGARCH(1,1) Conditional Volatility — BRICS Currencies vs USD\n"
        "(Annualised %)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

    for ax, (col, d), color in zip(axes, egarch_results.items(), colors):
        rank = metrics.loc[col, "Stability Rank"]
        vol  = d["cond_vol_ann"]
        ax.fill_between(vol.index, vol.values, alpha=0.25, color=color)
        ax.plot(vol.index, vol.values, color=color, linewidth=0.8)
        ax.set_ylabel("Vol (%)", fontsize=9)
        ax.set_title(
            f"{col}  |  Stability Rank #{rank}  |  "
            f"Mean Ann. Vol = {metrics.loc[col,'Mean Cond. Vol (ann. %)']:.2f}%",
            fontsize=10, loc="left",
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {save_path}")
    plt.show()


def plot_stability_ranking(metrics: pd.DataFrame,
                           constraints: dict,
                           save_path: str = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BRICS Currency Stability Summary (EGARCH-derived)",
                 fontsize=13, fontweight="bold")
    ordered = metrics.index.tolist()
    colors  = ["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]

    vols = metrics.loc[ordered, "Mean Cond. Vol (ann. %)"]
    bars = ax1.bar(ordered, vols.values, color=colors, edgecolor="black", linewidth=0.7)
    ax1.set_title("Mean Annualised Conditional Volatility")
    ax1.set_ylabel("Volatility (%)")
    for b, v in zip(bars, vols.values):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    wts   = constraints["iv_weights"][ordered] * 100
    bars2 = ax2.bar(ordered, wts.values, color=colors, edgecolor="black", linewidth=0.7)
    ax2.set_title("Inverse-Variance Weights (Σ = 100%)")
    ax2.set_ylabel("Weight (%)")
    for b, v in zip(bars2, wts.values):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {save_path}")
    plt.show()


def plot_unit_value(unit_df: pd.DataFrame, save_path: str = None) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(unit_df.index, unit_df["UNIT"],   color="#2c3e50", lw=1.8,
            label="BRICS UNIT")
    ax.plot(unit_df.index, unit_df["Basket"], color="#3498db", lw=1.0,
            ls="--", alpha=0.7, label="Currency Basket (60%)")
    ax.plot(unit_df.index, unit_df["Gold"],   color="#f39c12", lw=1.0,
            ls="--", alpha=0.7, label="Gold (40%)")
    ax.axhline(1.0, color="grey", lw=0.7, ls=":")
    ax.set_title("Simulated BRICS UNIT Index  (Base = 1.0 at Start Date)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Index Value")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {save_path}")
    plt.show()


def plot_egarch_diagnostics(egarch_results: dict,
                            save_prefix: str = None) -> None:
    for col, d in egarch_results.items():
        res       = d["result"]
        std_resid = (res.resid / res.conditional_volatility).dropna()

        fig = plt.figure(figsize=(13, 8))
        fig.suptitle(f"EGARCH(1,1) Diagnostics — {col}",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        ax1.plot(std_resid.index, std_resid.values, lw=0.6, color="#2c3e50")
        ax1.axhline(0,  color="red",  lw=0.8, ls="--")
        ax1.axhline( 2, color="grey", lw=0.6, ls=":")
        ax1.axhline(-2, color="grey", lw=0.6, ls=":")
        ax1.set_title("Standardised Residuals  z_t")

        plot_acf(std_resid.values, lags=20, ax=ax2, zero=False)
        ax2.set_title("ACF of z_t  (should be ~0 at all lags)")

        ax3.plot(std_resid.index, std_resid.values**2, lw=0.6, color="#e74c3c")
        ax3.set_title("Squared Standardised Residuals  z²_t")

        plot_acf(std_resid.values**2, lags=20, ax=ax4, zero=False)
        ax4.set_title("ACF of z²_t  (should be ~0 — no remaining ARCH)")

        plt.tight_layout()
        if save_prefix:
            path = f"{save_prefix}_{col}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"    Saved: {path}")
        plt.show()

# STEP 9 — EXPORT

def export_results(metrics: pd.DataFrame,
                   constraints: dict,
                   pre_diag: pd.DataFrame,
                   post_diag: pd.DataFrame,
                   tables_dir: str = TABLES_DIR) -> None:
    """
    Files written:
        egarch_volatility_metrics.csv
        prefit_diagnostics.csv
        postfit_diagnostics.csv
        egarch_weights.csv
    """
    os.makedirs(tables_dir, exist_ok=True)

    metrics.to_csv(os.path.join(tables_dir, "egarch_volatility_metrics.csv"))
    pre_diag.to_csv(os.path.join(tables_dir, "prefit_diagnostics.csv"))
    post_diag.to_csv(os.path.join(tables_dir, "postfit_diagnostics.csv"))

    wt_df = pd.DataFrame({
        "Currency"      : constraints["ordered_currencies"],
        "Stability Rank": range(1, 6),
        "Mean Cond Var" : [constraints["mean_cond_var"][c]
                           for c in constraints["ordered_currencies"]],
        "IV Weight"     : [constraints["iv_weights"][c]
                           for c in constraints["ordered_currencies"]],
        "IV Weight (%)" : [constraints["iv_weights"][c] * 100
                           for c in constraints["ordered_currencies"]],
    })
    wt_df.to_csv(os.path.join(tables_dir, "egarch_weights.csv"), index=False)

    print(f"\n[9] Results saved to '{tables_dir}/'")
    for f in ["egarch_volatility_metrics.csv", "prefit_diagnostics.csv",
              "postfit_diagnostics.csv", "egarch_weights.csv"]:
        print(f"    {f}")

# MAIN — FULL PIPELINE ORCHESTRATOR

def main():
    print("=" * 65)
    print("  BRICS CURRENCY UNIT — EGARCH VOLATILITY PIPELINE")
    print("=" * 65)

    # --- Step 1: Data ---
    from data_cleaning import run as fetch_data
    prices, gold, returns = fetch_data()

    # --- Step 2: Pre-fit diagnostics ---
    pre_diag = run_preflight_diagnostics(returns)

    # --- Step 3: Fit EGARCH(1,1) ---
    from model import fit_egarch
    egarch_results = fit_egarch(returns)

    # --- Step 4: Post-fit diagnostics ---
    post_diag = post_fit_diagnostics(egarch_results)

    # --- Step 5: Volatility metrics & ranking ---
    metrics = compute_volatility_metrics(egarch_results)

    # --- Step 6: Weight constraints ---
    constraints = derive_weight_constraints(metrics)

    # --- Step 7: Simulate UNIT ---
    unit_df = simulate_unit_value(prices, gold,
                                  weights=constraints["iv_weights"])

    # --- Step 8: Plots ---
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("\n[8] Generating plots...")
    plot_conditional_volatility(
        egarch_results, metrics,
        save_path=os.path.join(FIGURES_DIR, "brics_conditional_vol.png"),
    )
    plot_stability_ranking(
        metrics, constraints,
        save_path=os.path.join(FIGURES_DIR, "brics_stability_ranking.png"),
    )
    plot_unit_value(
        unit_df,
        save_path=os.path.join(FIGURES_DIR, "brics_unit_index.png"),
    )
    plot_egarch_diagnostics(
        egarch_results,
        save_prefix=os.path.join(FIGURES_DIR, "brics_diag"),
    )

    # --- Step 9: Export ---
    export_results(metrics, constraints, pre_diag, post_diag)

    # --- Final summary ---
    print("\n" + "=" * 65)
    print("  FINAL WEIGHT CONSTRAINTS FOR OPTIMISER")
    print("=" * 65)
    print("\n  UNIT = 0.6 × [Σ w_i · FX_i] + 0.4 × Gold")
    print("\n  EGARCH-derived ordering constraint:")
    ordered = constraints["ordered_currencies"]
    print("    " + " >= ".join([f"w_{c}" for c in ordered]))
    print("\n  Inverse-variance seed weights (one valid solution):")
    for c in ordered:
        w = constraints["iv_weights"][c]
        print(f"    w_{c} = {w:.4f}  ({w*100:.1f}%)")
    print(f"\n  Sum = {constraints['iv_weights'].sum():.6f}  ✓")
    print("\n  Feed the ordering constraint + Σw=1 + w_i≥0 into your")
    print("  delta-UNIT=0 linear system to filter the infinite set.\n")


if __name__ == "__main__":
    main()
