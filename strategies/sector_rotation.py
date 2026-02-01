"""
strategies/sector_rotation.py
Pick top-N ETFs each month after applying risk filters.
"""
import pandas as pd
import numpy as np

MAX_EXPOSURE = 0.90   # 90 % invested, 10 % cash
MIN_ADX = 15          # trend must be present
MAX_RSI = 80          # not extremely overbought


def allocate(scored: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    *scored* has one row per (ETF, month-end) with FinalScore, ADX_14, RSI_14.
    Returns a DataFrame with one row per selected ETF per month, plus a Weight column.
    """
    allocs = []
    for date, grp in scored.groupby(scored.index):
        # risk filters
        grp = grp.copy()
        if "ADX_14" in grp.columns:
            grp = grp[grp["ADX_14"] >= MIN_ADX]
        if "RSI_14" in grp.columns:
            grp = grp[grp["RSI_14"] <= MAX_RSI]

        if grp.empty:
            continue

        top = grp.nlargest(top_n, "FinalScore").copy()
        top["Weight"] = MAX_EXPOSURE / len(top)
        top["Selected_Date"] = date
        allocs.append(top)

    if not allocs:
        return pd.DataFrame()
    return pd.concat(allocs)
