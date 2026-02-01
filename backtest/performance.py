"""
backtest/performance.py
Comprehensive performance-metric calculation.
"""
import numpy as np
import pandas as pd
from backtest.portfolio import Portfolio

RF_ANNUAL = 0.06  # 6% Indian risk-free rate


def compute(port: Portfolio, benchmark: pd.DataFrame) -> dict:
    """Return a flat dict of every metric we report."""
    eq = port.get_equity()
    if eq.empty:
        return {}

    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.set_index("Date").sort_index()
    pv = eq["Portfolio_Value"]

    # -- basic returns ------------------------------------------------
    total_ret = (pv.iloc[-1] / port.initial_capital - 1) * 100
    days = (pv.index[-1] - pv.index[0]).days
    years = max(days / 365.25, 0.01)
    cagr = ((pv.iloc[-1] / port.initial_capital) ** (1 / years) - 1) * 100

    # daily returns (approximate from equity snapshots)
    daily_ret = pv.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252) * 100

    # -- risk-adjusted ------------------------------------------------
    excess = daily_ret.mean() * 252 - RF_ANNUAL
    sharpe = excess / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    down = daily_ret[daily_ret < 0]
    down_std = down.std() * np.sqrt(252) if len(down) > 0 else 1e-8
    sortino = excess / down_std

    # -- drawdown -----------------------------------------------------
    running_max = pv.expanding().max()
    dd = (pv - running_max) / running_max * 100
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # -- monthly stats ------------------------------------------------
    mr = port.monthly_returns()
    m_ret = mr["Monthly_Return"].dropna() if not mr.empty else pd.Series(dtype=float)
    win_rate = (m_ret > 0).mean() * 100 if len(m_ret) > 0 else 0.0
    best_m = float(m_ret.max()) if len(m_ret) > 0 else 0.0
    worst_m = float(m_ret.min()) if len(m_ret) > 0 else 0.0

    # -- benchmark comparison -----------------------------------------
    bench_slice = benchmark.loc[pv.index[0]:pv.index[-1]]
    bench_total = (bench_slice["Close"].iloc[-1] / bench_slice["Close"].iloc[0] - 1) * 100
    bench_cagr = ((bench_slice["Close"].iloc[-1] / bench_slice["Close"].iloc[0]) ** (1 / years) - 1) * 100
    alpha = cagr - bench_cagr

    # hit rate
    hit_rate = 0.0
    if not mr.empty:
        mr2 = mr.copy()
        mr2["Date"] = pd.to_datetime(mr2["Date"])
        mr2 = mr2.set_index("Date")
        bench_m = bench_slice["Close"].resample("ME").last().pct_change() * 100
        merged = mr2.join(bench_m.rename("Bench_M"), how="inner").dropna()
        if len(merged) > 0:
            hit_rate = (merged["Monthly_Return"] > merged["Bench_M"]).mean() * 100

    # -- trade stats --------------------------------------------------
    trades = port.get_trades()
    sells = trades[trades["Action"] == "SELL"] if not trades.empty else pd.DataFrame()
    n_trades = len(trades)
    n_completed = len(sells)
    trade_wr = (sells["PnL"] > 0).mean() * 100 if len(sells) > 0 else 0.0
    avg_trade_ret = sells["PnL_Pct"].mean() if len(sells) > 0 else 0.0
    best_trade = sells["PnL_Pct"].max() if len(sells) > 0 else 0.0
    worst_trade = sells["PnL_Pct"].min() if len(sells) > 0 else 0.0

    return {
        "Total_Return_%": round(total_ret, 2),
        "CAGR_%": round(cagr, 2),
        "Annual_Volatility_%": round(ann_vol, 2),
        "Sharpe_Ratio": round(sharpe, 3),
        "Sortino_Ratio": round(sortino, 3),
        "Max_Drawdown_%": round(max_dd, 2),
        "Calmar_Ratio": round(calmar, 3),
        "Monthly_Win_Rate_%": round(win_rate, 2),
        "Best_Month_%": round(best_m, 2),
        "Worst_Month_%": round(worst_m, 2),
        "Benchmark_Total_Return_%": round(bench_total, 2),
        "Benchmark_CAGR_%": round(bench_cagr, 2),
        "Alpha_%": round(alpha, 2),
        "Hit_Rate_%": round(hit_rate, 2),
        "Total_Trades": n_trades,
        "Completed_Trades": n_completed,
        "Trade_Win_Rate_%": round(trade_wr, 2),
        "Avg_Trade_Return_%": round(avg_trade_ret, 2),
        "Best_Trade_%": round(best_trade, 2),
        "Worst_Trade_%": round(worst_trade, 2),
    }
