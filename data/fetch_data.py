"""
data/fetch_data.py — Download and cache Indian ETF OHLCV data via yfinance.
"""
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

TICKERS = {
    "NIFTYBEES":        "NIFTYBEES.NS",
    "BANKBEES":         "BANKBEES.NS",
    "ITBEES":           "ITBEES.NS",
    "PHARMABEES":       "PHARMABEES.NS",
    "PSUBANKBEES":      "PSUBANKBEES.NS",
    "FMCGBEES":         "FMCGBEES.NS",
    "METALBEES":        "METALBEES.NS",
    "CONSUMPTIONBEES":  "KONSUMBEES.NS",
    "MIDCAPBEES":       "MIDCAPBEES.NS",
    "SMALLCAPBEES":     "SMALLCAPBEES.NS",
}


def _load_or_download(name: str, yf_symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Return cached CSV if fresh enough, else re-download."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.csv")

    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # If last row is within 2 days, use cache
        if (datetime.now() - df.index[-1]).days <= 2:
            print(f"  {name}: loaded from cache ({len(df)} rows)")
            return df

    print(f"  {name}: downloading …")
    try:
        df = yf.download(yf_symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  {name}: WARNING — empty response")
            return None
        # Flatten multi-level columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.to_csv(path)
        print(f"  {name}: saved {len(df)} rows")
        return df
    except Exception as exc:
        print(f"  {name}: ERROR — {exc}")
        return None


def fetch_all(start: str = "2013-01-01", end: str | None = None) -> dict[str, pd.DataFrame]:
    """Download (or load cached) data for every ticker. Returns only successful ones."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    print(f"fetch_all: {start} → {end}")
    raw = {}
    for name, symbol in TICKERS.items():
        df = _load_or_download(name, symbol, start, end)
        if df is not None and len(df) > 60:
            raw[name] = df
    print(f"fetch_all: {len(raw)} ETFs loaded\n")
    return raw


def align(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Intersect all date indices so every ETF covers the same trading days."""
    if not data:
        return {}
    common = set(data[next(iter(data))].index)
    for df in data.values():
        common &= set(df.index)
    common = sorted(common)
    print(f"align: {len(common)} common trading days  ({common[0].date()} → {common[-1].date()})")
    return {name: df.loc[common] for name, df in data.items()}
