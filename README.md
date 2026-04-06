# Stability and Weight Optimization of a Potential BRICS Currency: An EGARCH-Based Volatility Analysis

---

## Project Title

**Stability and Weight Optimization of a Potential BRICS Currency: An EGARCH-Based Volatility Analysis**

---

## Overview

This project examines the financial feasibility of a unified **BRICS Currency Unit (UNIT)** — a proposed composite monetary instrument for Brazil, Russia, India, China, and South Africa. Using an **EGARCH(1,1)** framework with Student-*t* distributed errors applied to daily exchange rate returns from 2017 to 2024, we estimate time-varying volatility, shock persistence, and asymmetry across BRICS currencies against the US dollar.

Volatility-based metrics are used to construct **inverse-variance weights**, producing a basket structure that assigns dominant roles to the Indian Rupee (INR) and Chinese Yuan (CNY). A complementary **Macroeconomic Vulnerability Index (MVI)** is constructed from World Bank panel data (1999–2023) to capture structural risk across five dimensions: inflation, output, fiscal position, energy dependence, and trade exposure.

The simulated UNIT index — comprising a **60% EGARCH-weighted currency basket** and a **40% gold anchor** — achieves an annualised volatility of **13.62%** and a total return of **+63.39%** over the sample period.

---

## Research Question

> *Can a financially defensible BRICS Currency Unit be analytically designed using EGARCH-derived volatility measures, and what basket weights emerge from this framework?*

---

## Methodology

### 1. EGARCH(1,1) Volatility Modelling
- Daily log returns computed for BRL, RUB, INR, CNY, ZAR vs USD (2017–2024)
- Pre-fit diagnostics: Jarque-Bera normality test, ARCH-LM test
- EGARCH(1,1) with Student-*t* errors estimated per currency (Nelson, 1991)
- Post-fit diagnostics: Ljung-Box tests on standardised residuals *z*ₜ and *z*²ₜ
- Key parameters extracted: ω, α (magnitude), γ (leverage/asymmetry), β (persistence)

### 2. Stability Ranking & Inverse-Variance Weights
- Mean conditional variance extracted per currency as the primary ranking criterion
- Inverse-variance weights derived: *wᵢ = (1/σᵢ²) / Σⱼ(1/σⱼ²)*
- INR and CNY emerge as the dominant basket anchors (~85% combined)

### 3. BRICS Currency Unit (UNIT) Simulation
- UNIT = 0.6 × [IV-weighted currency basket] + 0.4 × Gold (COMEX GC=F)
- All components normalised to base 1.0 in January 2017

### 4. Macroeconomic Vulnerability Index (MVI)
- Four poles: Inflation, GDP, Debt-to-GDP, Net Trade (1999–2023, World Bank WDI)
- Country-specific z-score standardisation → pooled percentile ranking → equal-weight averaging

---

## Dataset Description

| Dataset | Source | Period | Variables |
|---|---|---|---|
| FX daily prices (BRL, RUB, INR, CNY, ZAR) | Yahoo Finance (`yfinance`) | 2017–2024 | Close prices → log returns |
| Gold Futures (GC=F) | Yahoo Finance (`yfinance`) | 2017–2024 | Close prices |
| Macroeconomic panel data | World Bank WDI | 1999–2023 | Inflation (CPI), GDP, Debt/GDP, Net Trade |
| Vulnerability Index (processed) | Authors | 1999–2023 | MVI scores, pole-level scores, z-scores |

**Raw data files** in `data/raw/`:
- `BRICS_Vulnerability_Index.xlsx` — MVI summary and pole-level scores

> FX and gold data are downloaded live via `yfinance` at runtime. An internet connection is required.

---

## Instructions to Run the Code

### Prerequisites

Ensure Python 3.9+ is installed. Then install all dependencies:

```bash
pip install -r requirements.txt
```

### Option A — Run the full pipeline as a Python script

```bash
python src/egarch_analysis.py
```

This will:
1. Download FX and gold data from Yahoo Finance
2. Run pre-fit diagnostics
3. Fit EGARCH(1,1) models for all 5 currencies
4. Run post-fit diagnostics
5. Compute volatility metrics and stability ranking
6. Derive inverse-variance basket weights
7. Simulate the UNIT index
8. Save all plots to `results/figures/`
9. Export all tables to `results/tables/`

### Option B — Step-by-step Jupyter Notebook

```bash
jupyter notebook notebooks/BRICS_EGARCH_Analysis.ipynb
```

Run all cells top-to-bottom. All outputs (figures and tables) are rendered inline and also saved to `results/`.

### Option C — Run individual modules

```bash
# Data download only
python src/data_download.py

# Diagnostics only (requires processed data)
python src/diagnostics.py

# Fit EGARCH models (requires processed data)
python src/egarch_model.py

# Simulate UNIT index (requires EGARCH results)
python src/unit_simulation.py

```

---

## Results Summary

| Metric | Value |
|---|---|
| INR annualised conditional volatility | 6.31% |
| CNY annualised conditional volatility | 6.90% |
| ZAR annualised conditional volatility | 15.79% |
| BRL annualised conditional volatility | 16.42% |
| RUB annualised conditional volatility | 24,431%* |
| INR basket weight | 46.66% |
| CNY basket weight | 39.01% |
| ZAR basket weight | 7.45% |
| BRL basket weight | 6.88% |
| RUB basket weight | ≈ 0.00% |
| UNIT annualised volatility | 13.62% |
| UNIT total return (2017–2024) | +63.39% |

*Reflects the post-2022 sanctions structural break; not an estimation error.

---

## Repository Structure

```
brics-unit-egarch/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
│
├── data/
│   ├── raw/
│   │   ├── BRICS_Vulnerability_Index.xlsx
│   └── processed/                     ← Generated at runtime
│       ├── egarch_volatility_metrics.csv
│       ├── egarch_weights.csv
│       ├── prefit_diagnostics.csv
│       └── postfit_diagnostics.csv
│
├── src/
│   ├── data_download.py               ← FX & gold data via yfinance
│   ├── diagnostics.py                 ← Pre- and post-fit tests
│   ├── egarch_model.py                ← EGARCH(1,1) fitting & metrics
│   ├── unit_simulation.py             ← UNIT index construction
│   └── egarch_analysis.py             ← Full pipeline (entry point)
│
├── notebooks/
│   └── BRICS_EGARCH_Analysis.ipynb   ← End-to-end documented notebook
│
├── results/
│   ├── figures/                       ← All plots (saved at runtime)
│   │   ├── brics_conditional_vol.png
│   │   ├── brics_stability_ranking.png
│   │   ├── brics_unit_index.png
│   │   ├── brics_individual_fx.png
│   │   └── brics_diag_{CCY}.png       ← Per-currency diagnostic grids
│   └── tables/                        ← All CSV exports (saved at runtime)
│       ├── egarch_volatility_metrics.csv
│       ├── egarch_weights.csv
│       ├── prefit_diagnostics.csv
│       ├── postfit_diagnostics.csv
│       └── Vulnerability_Index_4_Parameters.csv
│
└── paper/
    └── final_report.pdf
```

---

## Dependencies

See `requirements.txt` for pinned versions. Key libraries:

| Library | Purpose |
|---|---|
| `arch` | EGARCH model estimation |
| `yfinance` | FX and commodity price download |
| `statsmodels` | Diagnostic tests (Ljung-Box, ARCH-LM, Jarque-Bera) |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Visualisation |
| `scipy` | Statistical functions |
| `openpyxl` | Reading `.xlsx` MVI data |
| `jupyter` | Notebook interface |

---

## Authors and Co-Authors

| Name | Role |
|---|---|
| **Naman Singhal** | Author |
| **Tapan C. Sanghani** | Co-Author |
| **Shubham Jain** | Co-Author |

**Semester II, AY 2025–26**

---

## Faculty Supervisor

Submitted to the **Ajay Upadhyaya Centre for Financial Markets Education and Research (CFM)**
Department of Economics & Finance, BITS Pilani

---

## References

- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347–370.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
- Mundell, R. A. (1961). A theory of optimum currency areas. *American Economic Review*, 51(4), 657–665.
- World Bank (2024). World Development Indicators. https://databank.worldbank.org/source/world-development-indicators
- Jain, A., & Tripathy, T. (2021). Estimating and forecasting BRICS currency exchange rate volatility. *Cogent Economics & Finance*, 9(1), 1985777.
