"""
data_download.py
================
Downloads FX daily prices for all five BRICS currencies and gold futures
from Yahoo Finance, computes log returns, and saves to data/processed/.

Usage:
    python src/data_download.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CURRENCIES = {
    "BRL": "BRL=X",   # Brazilian Real  / USD
    "RUB": "RUB=X",   # Russian Ruble   / USD
    "INR": "INR=X",   # Indian Rupee    / USD
    "CNY": "CNY=X",   # Chinese Yuan    / USD
    "ZAR": "ZAR=X",   # South African Rand / USD
}

GOLD_TICKER = "GC=F"   # COMEX Gold Futures

START_DATE = "2017-01-01"
END_DATE   = "2024-12-31"

# Known dates with data anomalies/gaps to drop before computing returns
ANOMALY_DATES = ["2015-01-16", "2021-06-22", "2022-03-07"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_fx(tickers: dict = CURRENCIES,
                start: str = START_DATE,
                end: str = END_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download FX close prices from Yahoo Finance and compute log returns.

    Parameters
    ----------
    tickers : dict
        Mapping of {currency_code: yahoo_ticker}, e.g. {"INR": "INR=X"}.
    start, end : str
        Date range in "YYYY-MM-DD" format.

    Returns
    -------
    prices : pd.DataFrame
        Daily close prices, one column per currency.
    returns : pd.DataFrame
        Log returns scaled to %, shape (T-1, N_currencies).
    """
    print("\n[DATA] Downloading FX prices from Yahoo Finance...")
    raw = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {name} ({ticker}).")
        raw[name] = df["Close"].squeeze()
        print(f"  {name:>3s}:  {len(df):5d} obs  "
              f"[{df.index[0].date()} → {df.index[-1].date()}]")

    prices = pd.DataFrame(raw).dropna(how="all")

    # Drop known anomaly dates
    prices = prices.drop(
        index=pd.to_datetime(ANOMALY_DATES, errors="ignore"), errors="ignore"
    )

    # Log returns, scaled to percent
    returns = np.log(prices / prices.shift(1)).dropna() * 100

    print(f"\n  Prices  shape : {prices.shape}")
    print(f"  Returns shape : {returns.shape}")
    return prices, returns


def download_gold(start: str = START_DATE,
                  end: str = END_DATE) -> pd.Series:
    """
    Download COMEX Gold Futures (GC=F) close prices.

    Parameters
    ----------
    start, end : str
        Date range in "YYYY-MM-DD" format.

    Returns
    -------
    gold : pd.Series
        Daily gold close prices indexed by date.
    """
    print("\n[DATA] Downloading Gold Futures (GC=F)...")
    df = yf.download(GOLD_TICKER, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No gold data returned for ticker {GOLD_TICKER}.")
    gold = df["Close"].squeeze()
    print(f"  Gold: {len(gold):5d} obs  "
          f"[{gold.index[0].date()} → {gold.index[-1].date()}]")
    return gold


def save_to_processed(prices: pd.DataFrame,
                       returns: pd.DataFrame,
                       gold: pd.Series,
                       out_dir: str = OUT_DIR) -> None:
    """Save downloaded data to data/processed/ as CSV files."""
    os.makedirs(out_dir, exist_ok=True)
    prices.to_csv(os.path.join(out_dir, "fx_prices.csv"))
    returns.to_csv(os.path.join(out_dir, "fx_returns.csv"))
    gold.to_csv(os.path.join(out_dir, "gold_prices.csv"), header=True)
    print(f"\n[DATA] Saved to '{out_dir}/'")
    for f in ["fx_prices.csv", "fx_returns.csv", "gold_prices.csv"]:
        print(f"  {f}")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prices, returns = download_fx()
    gold = download_gold()
    save_to_processed(prices, returns, gold)
    print("\n[DATA] Download complete.")
