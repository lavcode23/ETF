"""
reports/charts.py
Generate static PNG charts and save to reports/charts/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
CHART_DIR = os.path.join(os.path.dirname(__file__), "charts")


def _ensure_dir():
    os.makedirs(CHART_DIR, exist_ok=True)


# -- helpers ----------------------------------------------------------
def _equity_and_bench(runner):
    eq = runner.portfolio.get_equity()
    eq["Date"] = pd.to_datetime(eq["Date"])
    bench = runner.data["NIFTYBEES"]
    bench_slice = bench.loc[eq["Date"].min():eq["Date"].max()]
    bench_norm = (bench_slice["Close"] / bench_slice["Close"].iloc[0]) * runner.initial_capital
    return eq, bench_slice, bench_norm


# -- chart functions --------------------------------------------------
def equity_curve(runner):
    _ensure_dir()
    eq, bench_slice, bench_norm = _equity_and_bench(runner)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq["Date"], eq["Portfolio_Value"], label="Strategy", color="#2E86AB", linewidth=2)
    ax.plot(bench_slice.index, bench_norm, label="NIFTYBEES", color="#A23B72", linewidth=2, linestyle="--")
    ax.set_title("Equity Curve vs Benchmark", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (INR)")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "equity_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def drawdown_chart(runner):
    _ensure_dir()
    eq, _, _ = _equity_and_bench(runner)
    eq = eq.set_index("Date")
    rm = eq["Portfolio_Value"].expanding().max()
    dd = (eq["Portfolio_Value"] - rm) / rm * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dd.index, dd, 0, color="#E63946", alpha=0.6)
    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown %")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "drawdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def monthly_heatmap(runner):
    _ensure_dir()
    mr = runner.portfolio.monthly_returns().dropna()
    mr["Date"] = pd.to_datetime(mr["Date"])
    mr["Year"] = mr["Date"].dt.year
    mr["Month"] = mr["Date"].dt.month
    pivot = mr.pivot_table(index="Year", columns="Month", values="Monthly_Return")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Monthly Returns Heatmap (%)", fontsize=14, fontweight="bold")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticklabels([month_labels[int(c)-1] for c in pivot.columns], rotation=0)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "monthly_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def sector_rotation_heatmap(runner):
    _ensure_dir()
    a = runner.allocations.copy()
    a["Date"] = pd.to_datetime(a["Selected_Date"]).dt.strftime("%Y-%m")
    pivot = a.pivot_table(index="ETF", columns="Date", values="Weight", fill_value=0) * 100

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Allocation %"})
    ax.set_title("Sector Rotation Over Time", fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "sector_rotation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def feature_importance_chart(runner):
    _ensure_dir()
    if runner.importance_df.empty:
        return None
    top = runner.importance_df.head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Feature"][::-1], top["Importance"][::-1], color="#457B9D")
    ax.set_title("Feature Importance (XGBoost)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all(runner) -> dict[str, str | None]:
    print("  Generating static charts ...")
    return {
        "equity_curve": equity_curve(runner),
        "drawdown": drawdown_chart(runner),
        "monthly_heatmap": monthly_heatmap(runner),
        "sector_rotation": sector_rotation_heatmap(runner),
        "feature_importance": feature_importance_chart(runner),
    }
