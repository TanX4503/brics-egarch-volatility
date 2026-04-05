# BRICS Currency Unit — EGARCH Volatility Analysis

> **Research Paper:** *Volatility Analysis of a Potential BRICS Currency*
> **Model:** EGARCH(1,1) with Student-t errors · **Supplementary Layer:** Macroeconomic Vulnerability Index
> **Authors:** Economics and Finance Association

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Goals](#2-research-goals)
3. [Methodology](#3-methodology)
4. [What the Code Does](#4-what-the-code-does)
5. [Macroeconomic Parameter Analysis](#5-macroeconomic-parameter-analysis)
6. [Results Summary](#6-results-summary)
7. [Repository Structure](#7-repository-structure)
8. [File Reference](#8-file-reference)
9. [Instructions to Run](#9-instructions-to-run)
10. [Dependencies](#10-dependencies)

---

## 1. Project Overview

This repository contains the full analytical codebase, raw data, processed outputs, and results for a research paper investigating the **feasibility of a unified BRICS Currency Unit (UNIT)** — a proposed composite monetary instrument for the BRICS bloc (Brazil, Russia, India, China, South Africa).

The project proceeds along two parallel tracks:

| Track | Method | Goal |
|---|---|---|
| **Financial Volatility** | EGARCH(1,1) on daily FX returns (2017–2024) | Estimate conditional volatility, rank currencies by stability, derive basket weights |
| **Macroeconomic Vulnerability** | Standardised macro indicators + composite index | Quantify long-run structural risk across fiscal, trade, inflation, and energy dimensions |

Together, these two tracks provide both a *short-run* (market-based) and a *long-run* (fundamental) assessment of which BRICS currencies are most suitable to anchor a shared monetary unit.

---

## 2. Research Goals

### Primary Goals

- **Estimate time-varying volatility** of each BRICS currency (BRL, RUB, INR, CNY, ZAR) against the USD using the EGARCH(1,1) model, which accounts for leverage effects (asymmetric responses to positive vs. negative return shocks).
- **Rank currencies by financial stability** using EGARCH-derived conditional variance metrics, and translate this ranking into inverse-variance basket weights.
- **Simulate a BRICS Currency Unit (UNIT)** index as a weighted composite of the five currency components (60% weight) and gold (40% weight), benchmarked from 2017.
- **Build a Macroeconomic Vulnerability Index** for each BRICS nation using a multi-dimensional panel of indicators — inflation, GDP, public debt, energy import dependence, and international trade — to provide structural context for volatility findings.

---

## 3. Methodology

### 3.1 EGARCH(1,1) Model

The log-variance equation of the EGARCH(1,1) model is:

$$\ln(\sigma^2_t) = \omega + \alpha \left[|z_{t-1}| - \mathbb{E}|z_{t-1}|\right] + \gamma z_{t-1} + \beta \ln(\sigma^2_{t-1})$$

Where:
- $\sigma^2_t$ — conditional variance at time $t$
- $z_t = \varepsilon_t / \sigma_t$ — standardised residuals
- $\omega$ — intercept (long-run log variance)
- $\alpha$ — magnitude effect (symmetric response to shock size)
- $\gamma$ — leverage effect (asymmetric response: $\gamma < 0$ means negative shocks inflate volatility more)
- $\beta$ — persistence of volatility shocks

Key advantages over standard GARCH: the log-variance specification avoids non-negativity constraints, and the $\gamma$ term explicitly captures the asymmetric shock-volatility relationship observed in FX markets (currency depreciations spike volatility more than equivalent appreciations).

### 3.2 BRICS Currency Unit (UNIT) Index

$$\text{UNIT}_t = 0.6 \times \sum_{i} w_i \frac{\text{FX}_{i,t}}{\text{FX}_{i,0}} + 0.4 \times \frac{\text{Gold}_t}{\text{Gold}_0}$$

All components are normalised to 1.0 at the start date (2017-01-01). The 60/40 currency-gold split mirrors commodity-backed reserve proposals discussed in recent BRICS summits.

### 3.3 Macroeconomic Vulnerability Index

Country-specific macro indicators are z-score standardised and aggregated into a composite vulnerability score (0–1 scale) across five structural "poles":

| Pole | Indicators |
|---|---|
| **Inflation** | Annual inflation rate |
| **Output** | GDP (current USD) |
| **Fiscal** | General government debt-to-GDP ratio |
| **Energy** | Net energy imports as % of energy use |
| **Trade** | Exports, Imports, International Trade volume |

The `BRICS_Vulnerability_Index.xlsx` workbook documents the full methodology, intermediate z-scores, pole scores, and the final composite index for each country-year from 1999 to 2023.

---

## 4. What the Code Does

The analytical pipeline is implemented across three source modules (`src/`) and runs in **nine sequential steps**:

### Step 1 — Data Download & Log Returns (`src/data_cleaning.py`)

- Fetches daily FX closing prices for all five BRICS currencies and COMEX Gold Futures (`GC=F`) from Yahoo Finance via `yfinance` for the period **2017-01-01 to 2024-12-31**.
- Computes log returns: $r_t = 100 \times \ln(P_t / P_{t-1})$, scaled to percentage points.
- Saves raw prices to `data/raw/` and processed returns to `data/processed/`.

### Step 2 — Pre-fit Diagnostics (`src/analysis.py`)

- Runs the **Jarque-Bera test** on each return series to assess normality (all series expected to reject normality due to fat tails).
- Runs the **ARCH-LM test** (10 lags) to confirm the presence of autoregressive conditional heteroskedasticity — i.e., that time-varying volatility is statistically justified before fitting a GARCH-family model.

### Step 3 — EGARCH(1,1) Fitting (`src/model.py`)

- Fits an independent EGARCH(1,1) model with Student-t distributed errors to each of the five currency return series using the `arch` library.
- Prints the four key parameters ($\omega, \alpha, \gamma, \beta$), AIC, and BIC for each currency.
- Returns conditional daily volatility $\sigma_t$ and annualised volatility $\sigma_t \times \sqrt{252}$.

### Step 4 — Post-fit Diagnostics (`src/analysis.py`)

- Computes standardised residuals $z_t = \varepsilon_t / \sigma_t$ for each fitted model.
- Verifies $\mathbb{E}[z_t] \approx 0$ and $\text{Std}(z_t) \approx 1$.
- Runs **Ljung-Box tests** (10 lags) on both $z_t$ and $z_t^2$ to confirm the absence of remaining serial correlation and ARCH effects in residuals.

### Step 5 — Volatility Metrics & Stability Ranking (`src/analysis.py`)

Computes three metrics per currency and ranks them ascending by mean conditional variance:

| Metric | Description |
|---|---|
| Mean Conditional Variance | Primary ranking criterion — lower = more stable |
| Unconditional Long-run Variance | $\exp(\omega / (1-\beta))$ — theoretical steady-state variance |
| Persistence | $\alpha + \beta$ — how long shocks persist |
| Half-life | $\ln(0.5) / \ln(\beta)$ — days for shock to decay by 50% |

### Step 6 — Weight Constraints (`src/analysis.py`)

- Translates the EGARCH stability ranking into a **monotone ordering constraint**: $w_{\text{rank 1}} \geq w_{\text{rank 2}} \geq \cdots \geq w_{\text{rank 5}}$.
- Derives **inverse-variance (IV) weights** as the closed-form seed solution under a diagonal covariance assumption (no cross-currency correlations).

### Step 7 — BRICS UNIT Simulation (`src/analysis.py`)

- Simulates the UNIT index from 2017 to 2024 using EGARCH-derived IV weights for the currency basket and the gold price index as the commodity component.
- Computes annualised volatility and total return of the simulated UNIT index.

### Step 8 — Plots (`src/analysis.py`)

Generates and saves four figure sets to `results/figures/`:

| Figure | Description |
|---|---|
| `brics_conditional_vol.png` | Time-series of annualised conditional volatility for all five currencies (2017–2024) |
| `brics_stability_ranking.png` | Bar chart of mean conditional volatility per currency with IV weights labelled |
| `brics_unit_index.png` | Simulated UNIT index vs. individual currency components vs. gold (all normalised to 1.0) |
| `brics_diag_BRL/CNY/INR/RUB/ZAR.png` | 2×2 diagnostic grids per currency: return series, conditional volatility, ACF of residuals, ACF of squared residuals |

### Step 9 — Export (`src/analysis.py`)

Saves all tabular results to `results/tables/` as CSV files. See [File Reference](#8-file-reference) for details.

---

## 5. Macroeconomic Parameter Analysis

In addition to the EGARCH model, the research incorporates a panel of **macroeconomic indicators** for each BRICS nation (World Bank data, 1999–2023) to assess structural economic vulnerability.

### Indicators Collected (per country)

| Indicator | Variable Name |
|---|---|
| Inflation Rate (%) | `inflation_rate` |
| GDP (current USD) | `gdp` |
| Debt-to-GDP Ratio (%) | `debt_to_gdp` |
| Energy Import Dependence (% of energy use) | `ENERGY IMPORT (% of energy use)` |
| Exports (current USD) | `Exports` |
| Imports (current USD) | `Imports` |
| International Trade Volume (USD) | `International Trade` |

### Vulnerability Index Workbook

The `BRICS_Vulnerability_Index.xlsx` workbook contains four sheets:

| Sheet | Contents |
|---|---|
| **Vulnerability Index** | Final composite scores (0–1) per country-year, 1999–2023. Higher = more vulnerable. |
| **Pole Scores** | Intermediate aggregated scores per structural pole before final averaging |
| **Z-Scores (Intermediate)** | Raw standardised indicator values used in construction |
| **Methodology Notes** | Full description of normalisation, equal-weight pole averaging, and peer-group percentile ranking |

---

## 6. Results Summary

### EGARCH Diagnostics

All five currencies passed the ARCH-LM test at **p < 0.001**, confirming the presence of ARCH effects and justifying EGARCH modelling. All return series strongly reject normality (Jarque-Bera p < 0.001), consistent with the fat-tailed Student-t error distribution used.

### Stability Ranking

| Rank | Currency | Mean Ann. Vol (%) | Persistence (α+β) | IV Weight (%) |
|------|----------|-------------------|-------------------|---------------|
| 1 | INR (Indian Rupee) | 6.31% | 1.191 | 46.66% |
| 2 | CNY (Chinese Yuan) | 6.90% | 1.355 | 39.01% |
| 3 | ZAR (South African Rand) | 15.79% | 0.983 | 7.45% |
| 4 | BRL (Brazilian Real) | 16.42% | 1.083 | 6.88% |
| 5 | RUB (Russian Ruble) | 24,431%* | 2.262 | ≈ 0.00% |

> \* The RUB conditional variance is heavily distorted by the February 2022 sanctions shock and ensuing market dislocation, which caused a structural break absorbed by the EGARCH model as extreme persistent variance.

### EGARCH-Derived Ordering Constraint

```
w_INR  >=  w_CNY  >=  w_ZAR  >=  w_BRL  >=  w_RUB
```

### Simulated UNIT Index (2017–2024)

| Metric | Value |
|---|---|
| Annualised Volatility | 13.62% |
| Total Return | +63.39% |
| Gold Allocation | 40% |
| Currency Basket | 60% (IV-weighted per above) |

---

## 7. Repository Structure

```
brics-egarch-volatility/
|
+-- README.md                               <- This file
+-- requirements.txt                        <- Python package dependencies
|
+-- data/
|   +-- raw/
|   |   +-- fx_prices.csv                  <- Daily USD FX prices, 2017-2024 (Yahoo Finance)
|   |   +-- gold_prices.csv                <- Daily COMEX Gold Futures prices (GC=F)
|   |
|   +-- processed/
|   |   +-- fx_log_returns.csv             <- Daily log returns (%), all 5 currencies
|   |
|   +-- macro/
|       +-- BRAZIL_macro.csv               <- Brazil macroeconomic indicators, 1999-2023
|       +-- CHINA_macro.csv                <- China macroeconomic indicators, 1999-2023
|       +-- INDIA_macro.csv                <- India macroeconomic indicators, 1999-2023
|       +-- RUSSIA_macro.csv               <- Russia macroeconomic indicators, 1999-2023
|       +-- SOUTHAFRICA_macro.csv          <- South Africa macroeconomic indicators, 1999-2023
|       +-- BRICS_Vulnerability_Index.xlsx <- Composite vulnerability index workbook
|
+-- src/
|   +-- data_cleaning.py                   <- Step 1: data download, log returns, save/load
|   +-- model.py                           <- Step 3: EGARCH(1,1) fitting
|   +-- analysis.py                        <- Steps 2, 4-9: diagnostics, metrics,
|                                              weights, simulation, plots, export, main()
|
+-- notebooks/
|   +-- exploratory_analysis.ipynb         <- Step-by-step notebook (imports from src/)
|
+-- results/
|   +-- figures/
|   |   +-- brics_conditional_vol.png      <- Conditional volatility time series (all 5)
|   |   +-- brics_stability_ranking.png    <- Stability ranking bar chart with IV weights
|   |   +-- brics_unit_index.png           <- Simulated UNIT index vs components
|   |   +-- brics_diag_BRL.png             <- Brazil 2x2 diagnostic grid
|   |   +-- brics_diag_CNY.png             <- China 2x2 diagnostic grid
|   |   +-- brics_diag_INR.png             <- India 2x2 diagnostic grid
|   |   +-- brics_diag_RUB.png             <- Russia 2x2 diagnostic grid
|   |   +-- brics_diag_ZAR.png             <- South Africa 2x2 diagnostic grid
|   |
|   +-- tables/
|       +-- egarch_volatility_metrics.csv  <- Per-currency variance, vol, persistence, weight
|       +-- egarch_weights.csv             <- Stability rank and IV weights (clean export)
|       +-- prefit_diagnostics.csv         <- JB and ARCH-LM test results (Step 2)
|       +-- postfit_diagnostics.csv        <- Ljung-Box residual diagnostics (Step 4)
|
+-- paper/
    +-- final_report.pdf                   <- Research paper (to be added)
```

---

## 8. File Reference

### Source Code

| File | Purpose |
|---|---|
| `src/data_cleaning.py` | Downloads FX + gold data from Yahoo Finance, computes log returns, saves raw and processed data. Exposes `CURRENCIES`, `START_DATE`, `END_DATE` constants imported by other modules. |
| `src/model.py` | Fits EGARCH(1,1) with Student-t errors per currency using the `arch` library. Returns dictionary of fitted results, conditional volatilities, and parameters. Can be run standalone. |
| `src/analysis.py` | Full analytical pipeline: pre-fit and post-fit diagnostics, volatility metrics, stability ranking, IV weights, UNIT index simulation, all plots, and CSV exports. Entry point: `main()`. |

### Data Files

| File | Format | Description |
|---|---|---|
| `data/raw/fx_prices.csv` | CSV | Daily closing FX rates (BRL, RUB, INR, CNY, ZAR vs USD), 2017–2024 |
| `data/raw/gold_prices.csv` | CSV | Daily COMEX Gold Futures closing prices (GC=F), 2017–2024 |
| `data/processed/fx_log_returns.csv` | CSV | Daily log returns (%) derived from FX prices, all 5 currencies |
| `data/macro/BRAZIL_macro.csv` | CSV | Brazil: inflation, GDP, debt/GDP, energy imports, trade volume (1999–2023) |
| `data/macro/CHINA_macro.csv` | CSV | China: inflation, GDP, debt/GDP, energy imports, trade volume (1999–2023) |
| `data/macro/INDIA_macro.csv` | CSV | India: inflation, GDP, debt/GDP, energy imports, trade volume (1999–2023) |
| `data/macro/RUSSIA_macro.csv` | CSV | Russia: inflation, GDP, debt/GDP, energy imports, trade volume (1999–2023) |
| `data/macro/SOUTHAFRICA_macro.csv` | CSV | South Africa: inflation, GDP, debt/GDP, energy imports, trade volume (1999–2023) |
| `data/macro/BRICS_Vulnerability_Index.xlsx` | Excel | Multi-sheet workbook: composite vulnerability index (0–1), pole scores, z-scores, methodology |

### Results — Tables

| File | Description |
|---|---|
| `results/tables/egarch_volatility_metrics.csv` | Mean conditional variance, annualised vol, unconditional variance, persistence, half-life, stability rank, and IV weight per currency |
| `results/tables/egarch_weights.csv` | Clean export of stability ranks and IV weights (%) for use in basket optimiser |
| `results/tables/prefit_diagnostics.csv` | Jarque-Bera and ARCH-LM test statistics and p-values for each currency return series |
| `results/tables/postfit_diagnostics.csv` | Post-fit Ljung-Box test results on standardised residuals ($z_t$) and squared residuals ($z_t^2$) |

### Results — Figures

| File | Description |
|---|---|
| `results/figures/brics_conditional_vol.png` | Overlay of annualised conditional volatility for all 5 currencies (2017–2024) |
| `results/figures/brics_stability_ranking.png` | Bar chart: mean conditional vol per currency, labelled with IV basket weights |
| `results/figures/brics_unit_index.png` | Simulated UNIT index vs individual currency components and gold (normalised to 1.0) |
| `results/figures/brics_diag_BRL.png` | 2×2 diagnostic grid for BRL: return series, conditional vol, ACF($z_t$), ACF($z_t^2$) |
| `results/figures/brics_diag_CNY.png` | 2×2 diagnostic grid for CNY |
| `results/figures/brics_diag_INR.png` | 2×2 diagnostic grid for INR |
| `results/figures/brics_diag_RUB.png` | 2×2 diagnostic grid for RUB |
| `results/figures/brics_diag_ZAR.png` | 2×2 diagnostic grid for ZAR |

---

## 9. Instructions to Run

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/brics-egarch-volatility.git
cd brics-egarch-volatility
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3a. Run the Full Pipeline (Recommended)

```bash
python src/analysis.py
```

Runs all nine steps end-to-end. FX and gold data are downloaded live from Yahoo Finance. Plots are saved to `results/figures/` and tables to `results/tables/`.

### 3b. Run Step-by-Step (Notebook)

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

Imports from the three `src/` modules and runs each step in a separate cell with inline outputs and plots. Recommended for exploration and for verifying individual steps.

### 3c. Run Individual Modules

```bash
python src/data_cleaning.py   # Step 1 only — downloads and saves raw data and returns
python src/model.py           # Steps 1 + 3 — downloads data, fits EGARCH, prints parameters
```

### Notes

- The `data/raw/` and `data/processed/` files are generated at runtime by `data_cleaning.py`. They are included in the repository for reproducibility without requiring a live internet connection.
- The `data/macro/` files are **static** — they do not require re-downloading and are used only by the macroeconomic vulnerability analysis (documented in the paper).
- The `src/__pycache__/` directory contains compiled `.pyc` bytecode and can be safely ignored or deleted.

---

## 10. Dependencies

```
arch>=8.0.0
yfinance>=0.2.0
statsmodels>=0.13.0
pandas>=1.4.0
numpy>=1.22.3
matplotlib>=3.5.0
scipy>=1.8.0
```

Full version-pinned list is in `requirements.txt`. All packages are available via `pip`.

---

## Citation

If you use this code or derived results in your research, please cite:

```
Economics and Finance Association (2026).
Volatility Analysis of a Potential BRICS Currency.
EGARCH-Based Stability Ranking and Macroeconomic Vulnerability Assessment.
GitHub: https://github.com/TanX4503/brics-egarch-volatility
```

---
