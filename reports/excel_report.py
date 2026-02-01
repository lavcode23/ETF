"""
reports/excel_report.py
Multi-sheet Excel report from BacktestRunner results.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime


def generate(runner, output_dir: str = "reports") -> str:
    """Write one .xlsx with 7 sheets. Return filepath."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"Backtest_Report_{ts}.xlsx")

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        # 1 ── Summary
        rows = [["Metric", "Value"]]
        rows.append(["Start Date", runner.start_date])
        rows.append(["Initial Capital", f"₹{runner.initial_capital:,.0f}"])
        rows.append(["Top N ETFs", runner.top_n])
        rows.append(["", ""])
        for k, v in runner.performance.items():
            rows.append([k.replace("_", " "), v])
        pd.DataFrame(rows[1:], columns=rows[0]).to_excel(w, sheet_name="Summary", index=False)

        # 2 ── Performance
        pd.DataFrame([runner.performance]).T.rename_axis("Metric").rename(columns={0: "Value"}).to_excel(
            w, sheet_name="Performance"
        )

        # 3 ── Trade Log
        trades = runner.portfolio.get_trades()
        if not trades.empty:
            trades.to_excel(w, sheet_name="Trade Log", index=False)

        # 4 ── Monthly Returns
        mr = runner.portfolio.monthly_returns()
        if not mr.empty:
            mr.to_excel(w, sheet_name="Monthly Returns", index=False)

            # heatmap pivot
            mr2 = mr.copy()
            mr2["Date"] = pd.to_datetime(mr2["Date"])
            mr2["Year"] = mr2["Date"].dt.year
            mr2["Month"] = mr2["Date"].dt.month
            pivot = mr2.pivot_table(index="Year", columns="Month", values="Monthly_Return")
            pivot.to_excel(w, sheet_name="Returns Heatmap")

        # 5 ── Allocations
        if not runner.allocations.empty:
            cols = ["ETF", "Weight", "FinalScore", "ML_Probability", "TechScore",
                    "RSI_14", "MACD_Hist", "ADX_14", "Selected_Date"]
            runner.allocations[[c for c in cols if c in runner.allocations.columns]].to_excel(
                w, sheet_name="Allocations", index=False
            )

        # 6 ── Current Recommendation
        cur = runner.current_allocation()
        if cur is not None:
            cur.to_excel(w, sheet_name="Current Recommendation", index=False)

        # 7 ── Feature Importance
        if not runner.importance_df.empty:
            runner.importance_df.to_excel(w, sheet_name="Feature Importance", index=False)

    print(f"  Excel report → {path}")
    return path
