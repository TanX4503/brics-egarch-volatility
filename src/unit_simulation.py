"""
unit_simulation.py
==================
Constructs and simulates the BRICS Currency Unit (UNIT) index.

UNIT composition
----------------
    UNIT_t = 0.6 × Σᵢ (wᵢ × FXᵢ_t / FXᵢ_0)   +   0.4 × (Gold_t / Gold_0)

All components are normalised to 1.0 at the first trading date so that
price-level differences across currencies do not affect the composition.

The 60% currency basket uses EGARCH-derived inverse-variance weights.
The 40% gold anchor (COMEX GC=F) provides a store-of-value component
with low correlation to BRICS currency movements, enhancing diversification
and dampening the impact of currency-specific shocks.

Usage:
    python src/unit_simulation.py     # requires fx_prices.csv & gold_prices.csv
    # or import:
    from src.unit_simulation import simulate_unit_value, print_unit_summary
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_acf

ANNUAL         = 252       # trading days per year
BASKET_WEIGHT  = 0.60      # currency basket share of UNIT
GOLD_WEIGHT    = 0.40      # gold anchor share of UNIT

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


# ---------------------------------------------------------------------------
# UNIT simulation
# ---------------------------------------------------------------------------

def simulate_unit_value(prices: pd.DataFrame,
                         gold_prices: pd.Series,
                         weights: pd.Series) -> pd.DataFrame:
    """
    Simulate the BRICS Currency Unit (UNIT) index over the sample period.

    Parameters
    ----------
    prices : pd.DataFrame
        FX close prices, one column per BRICS currency.
    gold_prices : pd.Series
        COMEX Gold Futures (GC=F) close prices.
    weights : pd.Series
        IV weights indexed by currency code (must sum to 1).

    Returns
    -------
    result : pd.DataFrame
        Columns: "UNIT", "Basket" (currency component), "Gold" (gold component)
        All normalised to 1.0 at the start date.
    """
    print("\n[UNIT] Simulating BRICS Currency Unit index...")

    gold_aligned = gold_prices.reindex(prices.index, method="ffill").dropna()
    common_idx   = prices.index.intersection(gold_aligned.index)
    fx           = prices.loc[common_idx]
    gold         = gold_aligned.loc[common_idx]

    fx_norm      = fx   / fx.iloc[0]
    gold_norm    = gold / gold.iloc[0]

    basket = (fx_norm * weights).sum(axis=1)
    unit   = BASKET_WEIGHT * basket + GOLD_WEIGHT * gold_norm

    result = pd.DataFrame({
        "UNIT":   unit,
        "Basket": basket,
        "Gold":   gold_norm,
    })

    print_unit_summary(result, weights)
    return result


def print_unit_summary(unit_df: pd.DataFrame, weights: pd.Series) -> None:
    """Print performance summary for the simulated UNIT index."""
    unit     = unit_df["UNIT"]
    unit_ret = np.log(unit / unit.shift(1)).dropna() * 100
    ann_vol  = unit_ret.std() * np.sqrt(ANNUAL)
    tot_ret  = (unit.iloc[-1] - 1) * 100

    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║         BRICS UNIT Index — Summary           ║")
    print(f"  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Annualised Volatility : {ann_vol:7.2f}%             ║")
    print(f"  ║  Total Return          : {tot_ret:+7.2f}%             ║")
    print(f"  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Composition (60% FX basket + 40% Gold)      ║")
    for c in weights.index:
        w_basket = weights[c] * 100
        w_unit   = weights[c] * BASKET_WEIGHT * 100
        print(f"  ║    {c:>3s}: basket={w_basket:6.2f}%  unit={w_unit:5.2f}%         ║")
    print(f"  ╚══════════════════════════════════════════════╝")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_conditional_volatility(egarch_results: dict,
                                 metrics: pd.DataFrame,
                                 save_path: str = None) -> None:
    """
    Plot annualised conditional volatility for all five BRICS currencies.

    Highlights three key events visible in the data:
    - COVID-19 shock (March–April 2020): simultaneous spike across all series
    - Russia–Ukraine conflict (February 2022): explosive RUB volatility
    - South Africa's periodic political crises (ZAR spikes)
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        "EGARCH(1,1) Conditional Volatility — BRICS Currencies vs USD\n"
        "(Annualised %)",
        fontsize=14, fontweight="bold", y=0.98
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
            f"Mean Ann. Vol = {metrics.loc[col, 'Mean Cond. Vol (ann. %)']:.2f}%",
            fontsize=10, loc="left"
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_stability_ranking(metrics: pd.DataFrame,
                            constraints: dict,
                            save_path: str = None) -> None:
    """
    Side-by-side bar charts: (left) mean annualised volatility,
    (right) inverse-variance basket weights.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "BRICS Currency Stability Summary (EGARCH-derived)",
        fontsize=13, fontweight="bold"
    )
    ordered = metrics.index.tolist()
    colors  = ["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]

    # Left: volatility
    vols = metrics.loc[ordered, "Mean Cond. Vol (ann. %)"]
    bars = ax1.bar(ordered, vols.values, color=colors,
                   edgecolor="black", linewidth=0.7)
    ax1.set_title("Comparison of Annualised Conditional Volatilities (EGARCH)",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("Annualised Volatility (%)")
    for b, v in zip(bars, vols.values):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 f"{v:.2f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Right: weights
    wts  = constraints["iv_weights"][ordered] * 100
    bars2 = ax2.bar(ordered, wts.values, color=colors,
                    edgecolor="black", linewidth=0.7)
    ax2.set_title("Inverse-Variance Weights (Σ = 100%)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylabel("Weight (%)")
    for b, v in zip(bars2, wts.values):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_unit_value(unit_df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot the simulated UNIT index alongside its two components
    (currency basket and gold).
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(unit_df.index, unit_df["UNIT"],
            color="#2c3e50", lw=1.8, label="BRICS UNIT")
    ax.plot(unit_df.index, unit_df["Basket"],
            color="#3498db", lw=1.0, ls="--", alpha=0.7,
            label="Currency Basket (60%)")
    ax.plot(unit_df.index, unit_df["Gold"],
            color="#f39c12", lw=1.0, ls="--", alpha=0.7,
            label="Gold (40%)")
    ax.axhline(1.0, color="grey", lw=0.7, ls=":")
    ax.set_title("Simulated BRICS UNIT Index  (Base = 1.0 at Start Date)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Index Value")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_individual_currencies(prices: pd.DataFrame,
                                save_path: str = None) -> None:
    """
    Plot normalised (base 1.0) FX price series for all five currencies,
    stacked in subplots (one per currency).
    """
    norm = prices / prices.iloc[0]
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 18), sharex=True)
    fig.suptitle(
        "Individual BRICS Currency Performance (Base 1.0)",
        fontsize=16, fontweight="bold", y=0.92
    )
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

    for i, col in enumerate(prices.columns):
        ax = axes[i]
        ax.plot(norm[col], color=colors[i], lw=1.5)
        ax.set_title(f"{col} vs USD", loc="left",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylabel("Price Ratio")

    plt.xlabel("Date")
    plt.tight_layout(rect=[0, 0.03, 1, 0.91])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_egarch_diagnostics(egarch_results: dict,
                             save_prefix: str = None) -> None:
    """
    For each currency, produce a 2×2 diagnostic grid:
      [0,0] Standardised residuals z_t
      [0,1] ACF of z_t   (should show no significant autocorrelation)
      [1,0] Squared standardised residuals z²_t
      [1,1] ACF of z²_t  (should show no remaining ARCH effects)
    """
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

        ax3.plot(std_resid.index, std_resid.values**2,
                 lw=0.6, color="#e74c3c")
        ax3.set_title("Squared Standardised Residuals  z²_t")

        plot_acf(std_resid.values**2, lags=20, ax=ax4, zero=False)
        ax4.set_title("ACF of z²_t  (should be ~0 — no remaining ARCH)")

        plt.tight_layout()
        if save_prefix:
            path = f"{save_prefix}_{col}.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed"
    )
    prices_path = os.path.join(data_dir, "fx_prices.csv")
    gold_path   = os.path.join(data_dir, "gold_prices.csv")
    weights_path = os.path.join(data_dir, "egarch_weights.csv")

    for p in [prices_path, gold_path, weights_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"'{p}' not found. Run egarch_analysis.py for the full pipeline."
            )

    prices      = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    gold_prices = pd.read_csv(gold_path,   index_col=0, parse_dates=True).squeeze()
    wt_df       = pd.read_csv(weights_path)
    weights     = pd.Series(
        wt_df["IV Weight"].values, index=wt_df["Currency"].values
    )

    unit_df = simulate_unit_value(prices, gold_prices, weights)
    plot_unit_value(unit_df,
                    save_path=os.path.join(FIG_DIR, "brics_unit_index.png"))
    plot_individual_currencies(prices,
                               save_path=os.path.join(FIG_DIR, "brics_individual_fx.png"))
    print("\n[UNIT] Simulation complete.")
