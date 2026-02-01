"""
backtest/portfolio.py
Simple monthly-rebalance portfolio simulator.
"""
import pandas as pd
import numpy as np
from datetime import datetime

SLIPPAGE = 0.001  # 0.1 % round-trip


class Portfolio:
    def __init__(self, capital: float = 1_000_000):
        self.initial_capital = capital
        self.cash = capital
        self.positions: dict[str, dict] = {}   # etf → {shares, entry_price, entry_date}
        self.portfolio_value = capital

        self.equity_log: list[dict] = []       # one row per rebalance
        self.trade_log: list[dict] = []

    # ── public ───────────────────────────────────────────────────────
    def rebalance(self, date, allocs: pd.DataFrame, prices: dict[str, float]):
        """Close everything, then open new positions per *allocs*."""
        self._close_all(date, prices)
        for _, row in allocs.iterrows():
            etf = row["ETF"]
            if etf == "CASH" or etf not in prices:
                continue
            target_value = self.portfolio_value * row["Weight"]
            price = prices[etf]
            shares = int(target_value / price)
            if shares == 0:
                continue
            cost = shares * price * (1 + SLIPPAGE)
            if cost > self.cash:
                shares = int(self.cash / (price * (1 + SLIPPAGE)))
                cost = shares * price * (1 + SLIPPAGE)
            if shares <= 0:
                continue

            self.cash -= cost
            self.positions[etf] = {"shares": shares, "entry_price": price, "entry_date": date}
            self.trade_log.append({
                "Date": date, "ETF": etf, "Action": "BUY",
                "Shares": shares, "Price": price, "Value": shares * price,
            })
        self._update_value(date, prices)

    def get_equity(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_log)

    def get_trades(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)

    def monthly_returns(self) -> pd.DataFrame:
        eq = self.get_equity()
        if eq.empty:
            return pd.DataFrame()
        eq["Date"] = pd.to_datetime(eq["Date"])
        eq = eq.set_index("Date")
        m = eq["Portfolio_Value"].resample("ME").last()
        ret = m.pct_change() * 100
        return pd.DataFrame({"Date": m.index, "Portfolio_Value": m.values, "Monthly_Return": ret.values})

    # ── private ──────────────────────────────────────────────────────
    def _close_all(self, date, prices: dict[str, float]):
        for etf, pos in list(self.positions.items()):
            price = prices.get(etf, pos["entry_price"])
            proceeds = pos["shares"] * price * (1 - SLIPPAGE)
            pnl = proceeds - pos["shares"] * pos["entry_price"]
            pnl_pct = (price / pos["entry_price"] - 1) * 100
            self.cash += proceeds
            self.trade_log.append({
                "Date": date, "ETF": etf, "Action": "SELL",
                "Shares": pos["shares"], "Price": price,
                "Entry_Price": pos["entry_price"], "Entry_Date": pos["entry_date"],
                "PnL": round(pnl, 2), "PnL_Pct": round(pnl_pct, 2),
            })
        self.positions.clear()

    def _update_value(self, date, prices: dict[str, float]):
        pos_val = sum(
            p["shares"] * prices.get(etf, p["entry_price"])
            for etf, p in self.positions.items()
        )
        self.portfolio_value = self.cash + pos_val
        self.equity_log.append({
            "Date": date,
            "Portfolio_Value": round(self.portfolio_value, 2),
            "Cash": round(self.cash, 2),
            "Positions_Value": round(pos_val, 2),
        })


def simulate(data: dict[str, pd.DataFrame], allocs: pd.DataFrame, capital: float = 1_000_000) -> Portfolio:
    """
    Walk through every unique Selected_Date in *allocs*, look up month-end prices
    from *data*, and call portfolio.rebalance().
    """
    port = Portfolio(capital)
    dates = sorted(allocs["Selected_Date"].unique())
    print(f"  simulate: {len(dates)} rebalances")

    for date in dates:
        prices = {}
        for etf, df in data.items():
            if etf == "NIFTYBEES":
                continue
            # find closest date <= rebalance date
            candidates = df.index[df.index <= date]
            if len(candidates) > 0:
                prices[etf] = float(df.loc[candidates[-1], "Close"])

        if not prices:
            continue
        day_allocs = allocs[allocs["Selected_Date"] == date]
        port.rebalance(date, day_allocs, prices)

    print(f"  simulate: {len(port.trade_log)} trades, final value ₹{port.portfolio_value:,.0f}")
    return port
