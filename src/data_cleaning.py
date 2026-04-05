"""
data_cleaning.py

Step 1 — Data Download & Log Returns

Downloads daily FX prices for the five BRICS currencies and gold from
Yahoo Finance, computes log returns (scaled to %), and saves both the
raw prices and processed returns as CSVs under data/.

Usage (standalone):
    python src/data_cleaning.py

Imported by:
    notebooks/exploratory_analysis.ipynb
    src/analysis.py (via main pipeline)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import yfinance as yf

# CONFIGURATION

CURRENCIES = {
    "BRL": "BRL=X",   # Brazilian Real     / USD
    "RUB": "RUB=X",   # Russian Ruble      / USD
    "INR": "INR=X",   # Indian Rupee       / USD
    "CNY": "CNY=X",   # Chinese Yuan       / USD
    "ZAR": "ZAR=X",   # South African Rand / USD
}

GOLD_TICKER = "GC=F"   # COMEX Gold Futures

START_DATE = "2017-01-01"
END_DATE   = "2024-12-31"

RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


# STEP 1 — DATA DOWNLOAD & LOG RETURNS

def download_fx(tickers: dict, start: str, end: str):
    print("\n[1] Downloading FX data from Yahoo Finance...")
    raw = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data for {name} ({ticker}). Check ticker.")
        raw[name] = df["Close"].squeeze()
        print(f"    {name}: {len(df)} obs  "
              f"({df.index[0].date()} → {df.index[-1].date()})")

    prices = pd.DataFrame(raw).dropna(how="all")
    print(f"\n    Prices shape: {prices.shape}")
    return prices


def download_gold(ticker: str, start: str, end: str) -> pd.Series:
    print(f"\n    Downloading Gold ({ticker})...")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for Gold ({ticker}). Check ticker.")
    gold = df["Close"].squeeze()
    print(f"    Gold: {len(gold)} obs  "
          f"({gold.index[0].date()} → {gold.index[-1].date()})")
    return gold


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna() * 100
    print(f"    Returns shape: {returns.shape}")
    return returns


def save_raw(prices: pd.DataFrame, gold: pd.Series,
             out_dir: str = RAW_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    prices.to_csv(os.path.join(out_dir, "fx_prices.csv"))
    gold.to_csv(os.path.join(out_dir, "gold_prices.csv"), header=True)
    print(f"\n    Raw data saved → {out_dir}/")
    print(f"      fx_prices.csv   {prices.shape}")
    print(f"      gold_prices.csv {gold.shape}")


def save_processed(returns: pd.DataFrame,
                   out_dir: str = PROCESSED_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    returns.to_csv(os.path.join(out_dir, "fx_log_returns.csv"))
    print(f"\n    Processed data saved → {out_dir}/")
    print(f"      fx_log_returns.csv  {returns.shape}")


def load_raw(in_dir: str = RAW_DIR):
    prices = pd.read_csv(os.path.join(in_dir, "fx_prices.csv"),
                         index_col=0, parse_dates=True)
    gold   = pd.read_csv(os.path.join(in_dir, "gold_prices.csv"),
                         index_col=0, parse_dates=True).squeeze()
    return prices, gold


def load_processed(in_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    returns = pd.read_csv(os.path.join(in_dir, "fx_log_returns.csv"),
                          index_col=0, parse_dates=True)
    return returns


def run(tickers: dict = CURRENCIES,
        gold_ticker: str = GOLD_TICKER,
        start: str = START_DATE,
        end: str = END_DATE,
        save: bool = True):
    prices  = download_fx(tickers, start, end)
    gold    = download_gold(gold_ticker, start, end)
    returns = compute_log_returns(prices)

    if save:
        save_raw(prices, gold)
        save_processed(returns)

    return prices, gold, returns

# STANDALONE ENTRY POINT

if __name__ == "__main__":
    prices, gold, returns = run()
    print("\n[Done] data_cleaning.py complete.")
