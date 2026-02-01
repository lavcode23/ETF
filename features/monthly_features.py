"""
features/monthly_features.py
Aggregate daily data → one feature-row per (ETF, month-end).
Also creates the binary target label.
"""
import numpy as np
import pandas as pd


BENCHMARK = "NIFTYBEES"


def _month_features(df: pd.DataFrame, etf_name: str) -> pd.DataFrame:
    """
    Walk month-by-month through a single ETF's daily data and pull
    every feature we need at each month-end close.
    """
    df = df.copy()
    df["_ym"] = df.index.to_period("M")
    rows = []

    for ym, grp in df.groupby("_ym"):
        if len(grp) < 5:
            continue

        idx_end = grp.index[-1]           # last trading day of month
        loc_end = df.index.get_loc(idx_end)

        close_end = grp["Close"].iloc[-1]
        close_start = grp["Close"].iloc[0]

        def _lookback_ret(days):
            if loc_end < days:
                return np.nan
            return (close_end / df["Close"].iloc[loc_end - days] - 1) * 100

        def _lookback_vol(days):
            if loc_end < days:
                return np.nan
            return df["Returns"].iloc[loc_end - days: loc_end].std() * np.sqrt(252) * 100

        def _lookback_sharpe(days):
            if loc_end < days:
                return np.nan
            rets = df["Returns"].iloc[loc_end - days: loc_end]
            m = rets.mean() * 252
            s = rets.std() * np.sqrt(252)
            return m / s if s > 0 else 0.0

        def _avg_vol(days):
            if loc_end < days:
                return np.nan
            return df["Volume"].iloc[loc_end - days: loc_end].mean()

        avg_v_1m = grp["Volume"].mean()
        avg_v_3m = _avg_vol(63)

        rows.append(
            {
                "Date": idx_end,
                "ETF": etf_name,
                "Close": close_end,
                "MonthlyReturn": (close_end / close_start - 1) * 100,
                "Momentum_1M": _lookback_ret(21),
                "Momentum_3M": _lookback_ret(63),
                "Momentum_6M": _lookback_ret(126),
                "Momentum_12M": _lookback_ret(252),
                "Volatility_1M": _lookback_vol(21),
                "Volatility_3M": _lookback_vol(63),
                "SharpeRatio_1M": _lookback_sharpe(21),
                "SharpeRatio_3M": _lookback_sharpe(63),
                "RSI_14": grp["RSI_14"].iloc[-1],
                "MACD_Hist": grp["MACD_Hist"].iloc[-1],
                "ADX_14": grp["ADX_14"].iloc[-1],
                "BB_PercentB": grp["BB_PercentB"].iloc[-1],
                "ATR_Pct": grp["ATR_Pct"].iloc[-1],
                "Volume_Ratio": grp["Volume_Ratio"].iloc[-1],
                "ADX_Pos": grp["ADX_Pos"].iloc[-1],
                "ADX_Neg": grp["ADX_Neg"].iloc[-1],
                "VolumeSpike": avg_v_1m / avg_v_3m if (avg_v_3m and avg_v_3m > 0) else 1.0,
            }
        )

    return pd.DataFrame(rows).set_index("Date")


def build_monthly(data_with_ind: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build monthly features for every sector ETF (skips benchmark).
    Appends a RelativeStrength column (ETF monthly ret − benchmark monthly ret).
    """
    # Benchmark monthly returns first
    bench_rows = []
    if BENCHMARK in data_with_ind:
        bdf = data_with_ind[BENCHMARK].copy()
        bdf["_ym"] = bdf.index.to_period("M")
        for ym, grp in bdf.groupby("_ym"):
            if len(grp) < 5:
                continue
            bench_rows.append(
                {
                    "Date": grp.index[-1],
                    "Bench_Ret": (grp["Close"].iloc[-1] / grp["Close"].iloc[0] - 1) * 100,
                }
            )
    bench_df = pd.DataFrame(bench_rows).set_index("Date") if bench_rows else pd.DataFrame()

    # Sector ETFs
    parts = []
    for name, df in data_with_ind.items():
        if name == BENCHMARK:
            continue
        print(f"  monthly_features: {name}")
        parts.append(_month_features(df, name))

    monthly = pd.concat(parts, axis=0).sort_index()

    # Merge benchmark
    if not bench_df.empty:
        monthly = monthly.join(bench_df, how="left")
        monthly["RelativeStrength"] = monthly["MonthlyReturn"] - monthly["Bench_Ret"]
        monthly.drop(columns=["Bench_Ret"], inplace=True, errors="ignore")
    else:
        monthly["RelativeStrength"] = np.nan

    return monthly


def add_targets(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Target = 1 if next-month return > median of all ETFs that month, else 0.
    Rows where we cannot look one month ahead are dropped.
    """
    df = monthly.copy().sort_index()
    df["NextMonth_Return"] = df.groupby("ETF")["MonthlyReturn"].shift(-1)

    # Per-date median of next-month returns
    med = df.groupby(df.index)["NextMonth_Return"].transform("median")
    df["Target"] = (df["NextMonth_Return"] > med).astype(int)

    df.dropna(subset=["NextMonth_Return"], inplace=True)
    return df
