"""
features/indicators.py - Technical indicator calculations using the ta library.
"""
import numpy as np
import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Receive a single-ETF OHLCV DataFrame (columns: Open High Low Close Volume).
    Return the same DataFrame with ~20 indicator columns appended.
    """
    df = df.copy()

    # -- daily returns ------------------------------------------------
    df["Returns"] = df["Close"].pct_change()

    # -- RSI ----------------------------------------------------------
    df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # -- MACD ---------------------------------------------------------
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # -- Bollinger Bands ----------------------------------------------
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_PercentB"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # -- ATR ----------------------------------------------------------
    df["ATR_14"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14
    ).average_true_range()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"] * 100

    # -- ADX ----------------------------------------------------------
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["ADX_14"] = adx.adx()
    df["ADX_Pos"] = adx.adx_pos()
    df["ADX_Neg"] = adx.adx_neg()

    # -- Volume ratio (today vs 20-day avg) ---------------------------
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Vol_SMA20"]

    # -- Stochastic ---------------------------------------------------
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    return df


def add_indicators_all(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply add_indicators to every ETF in the dict."""
    out = {}
    for name, df in data.items():
        print(f"  indicators: {name}")
        out[name] = add_indicators(df)
    return out
