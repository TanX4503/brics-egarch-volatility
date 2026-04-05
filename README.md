# BRICS Currency Unit — EGARCH Volatility Analysis

## Project Title

BRICS Currency Unit (UNIT): EGARCH-Based Volatility Estimation and Stability-Weighted Basket Construction

## Overview

This project models the time-varying volatility of the five BRICS currency exchange rates — Brazilian Real (BRL), Russian Ruble (RUB), Indian Rupee (INR), Chinese Yuan (CNY), and South African Rand (ZAR) — using EGARCH(1,1) models fitted to daily USD exchange rate returns over 2017–2024. The estimated volatility dynamics are used to rank currencies by stability and derive weight constraints for a proposed BRICS Currency Unit (UNIT) index, defined as a composite of a 60% currency basket and 40% gold.

## Research Question

Can EGARCH-derived conditional volatility estimates provide a principled, data-driven basis for constructing stability-ranked currency weights in a BRICS Currency Unit, and what ordering and magnitude constraints do they imply for the basket optimiser?

## Methodology

The pipeline is implemented across three source modules and runs in nine sequential steps:

**Step 1 — Data Download & Log Returns** (`src/data_cleaning.py`): Daily FX closing prices and gold futures (GC=F) are fetched from Yahoo Finance via `yfinance`. Log returns are computed and scaled to percentage points. Raw prices are saved to `data/raw/` and processed returns to `data/processed/`.

**Step 2 — Pre-fit Diagnostics** (`src/analysis.py`): The Jarque-Bera test and the ARCH-LM test are run on each return series. Significant ARCH-LM p-values (< 0.05) confirm time-varying volatility in all five currencies, justifying a GARCH-family model.

**Step 3 — EGARCH(1,1) Fitting** (`src/model.py`): A Nelson (1991) EGARCH(1,1) model with Student-t errors is fitted independently to each currency:

$$\ln(\sigma^2_t) = \omega + \alpha[|z_{t-1}| - E|z_{t-1}|] + \gamma z_{t-1} + \beta \ln(\sigma^2_{t-1})$$

The log-variance formulation avoids positivity constraints; the $\gamma$ term captures leverage effects (depreciations spiking volatility more than appreciations).

**Step 4 — Post-fit Diagnostics** (`src/analysis.py`): Standardised residuals $z_t = \varepsilon_t / \sigma_t$ are checked for mean ≈ 0, std ≈ 1, and absence of serial correlation in $z_t$ and $z_t^2$ via Ljung-Box tests (lags = 10).

**Step 5 — Volatility Metrics & Stability Ranking** (`src/analysis.py`): Three metrics are computed per currency — mean conditional variance (primary ranking criterion), unconditional long-run variance $\exp(\omega/(1-\beta))$, and persistence $(\alpha + \beta)$. Currencies are ranked ascending by mean conditional variance (lower = more stable).

**Step 6 — Weight Constraints** (`src/analysis.py`): The EGARCH stability ranking translates into an ordering constraint ($w_{\text{rank 1}} \geq w_{\text{rank 2}} \geq \ldots \geq w_{\text{rank 5}}$). Inverse-variance weights are derived as a closed-form seed solution under a diagonal covariance assumption.

**Step 7 — BRICS UNIT Simulation** (`src/analysis.py`): The UNIT index is simulated as:

$$\text{UNIT}_t = 0.6 \times \sum_i w_i \frac{\text{FX}_{i,t}}{\text{FX}_{i,0}} + 0.4 \times \frac{\text{Gold}_t}{\text{Gold}_0}$$

All components are normalised to 1.0 at the start date.

**Step 8 — Plots** (`src/analysis.py`): Four figure sets are generated and saved to `results/figures/`: conditional volatility time series per currency, stability ranking bar charts with IV weights, the simulated UNIT index vs its components, and per-currency 2×2 diagnostic grids.

**Step 9 — Export** (`src/analysis.py`): Volatility metrics, diagnostic tables, and weight outputs are saved as CSVs to `results/tables/`.

## Dataset Description

| Field | Detail |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| Period | 2017-01-01 to 2024-12-31 |
| BRL | `BRL=X` — Brazilian Real / USD |
| RUB | `RUB=X` — Russian Ruble / USD |
| INR | `INR=X` — Indian Rupee / USD |
| CNY | `CNY=X` — Chinese Yuan / USD |
| ZAR | `ZAR=X` — South African Rand / USD |
| Gold | `GC=F` — COMEX Gold Futures |
| Observations | ~2,080 daily return observations per series (after differencing) |
| Storage | Data is downloaded live at runtime — no data files are bundled in this repository |

## Instructions to Run the Code

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/brics-egarch-volatility.git
cd brics-egarch-volatility
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3a. Run the full pipeline (script)

```bash
python src/analysis.py
```

This runs all nine steps end-to-end. Data is downloaded, models are fitted, plots are saved to `results/figures/`, and tables are saved to `results/tables/`.

### 3b. Run step-by-step (notebook)

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

The notebook imports from the three `src/` modules and runs each step in a separate cell with inline outputs and plots.

### 3c. Run individual modules

```bash
python src/data_cleaning.py   # Step 1 only — downloads and saves data
python src/model.py           # Steps 1 + 3 — downloads data and fits EGARCH
```

## Results Summary

All five currencies passed the ARCH-LM test at p < 0.001, confirming ARCH effects and justifying EGARCH modelling.

**Stability Ranking (most stable → least stable):**

| Rank | Currency | Mean Ann. Vol (%) | IV Weight |
|------|----------|-------------------|-----------|
| 1 | INR | 6.31% | 46.7% |
| 2 | CNY | 6.90% | 39.0% |
| 3 | ZAR | 15.79% | 7.4% |
| 4 | BRL | 16.42% | 6.9% |
| 5 | RUB | 24,328.61%* | 0.0% |

\* RUB mean conditional variance is heavily distorted by the 2022 sanctions shock and resulting market dislocation.

**EGARCH-derived ordering constraint:**

```
w_INR >= w_CNY >= w_ZAR >= w_BRL >= w_RUB
```

**Simulated UNIT Index (2017–2024):**
- Annualised volatility: 13.62%
- Total return: +63.39%

## Repository Structure

```
brics-egarch-volatility/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                        # fx_prices.csv, gold_prices.csv (runtime)
│   └── processed/                  # fx_log_returns.csv (runtime)
│
├── src/
│   ├── data_cleaning.py            # Step 1  — download, log returns, save/load
│   ├── model.py                    # Step 3  — EGARCH(1,1) fitting
│   └── analysis.py                 # Steps 2, 4–9 — diagnostics, metrics,
│                                   #   weights, simulation, plots, export, main()
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Step-by-step notebook, imports from src/
│
├── results/
│   ├── figures/                    # PNG plots (runtime)
│   └── tables/                     # CSV outputs (runtime)
│
└── paper/
    └── final_report.pdf (To be added)
```

## Dependencies

```
arch>=8.0.0
yfinance>=0.2.0
statsmodels>=0.13.0
pandas>=1.4.0
numpy>=1.22.3
matplotlib>=3.5.0
scipy>=1.8.0
```

Full version-pinned list is in `requirements.txt`.

## Authors
Economics and Finance Association
