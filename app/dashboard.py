"""
app/dashboard.py
Full Streamlit dashboard for the Indian ETF Sector Rotation system.

sys.path is adjusted at the top so that every sibling package (data/, features/, ...)
is importable regardless of where Streamlit mounts the repo.
"""
# ----------------------------------------------------------------------
# PATH FIX  <- this must come before any project import
# ----------------------------------------------------------------------
import sys, os
_THIS = os.path.dirname(os.path.abspath(__file__))          # .../app/
_ROOT = os.path.dirname(_THIS)                              # repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ----------------------------------------------------------------------
# STANDARD LIB / THIRD-PARTY
# ----------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------
# PROJECT IMPORTS  (all safe now that ROOT is on path)
# ----------------------------------------------------------------------
from backtest.backtest_runner import BacktestRunner
from reports.excel_report import generate as gen_excel

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Indian ETF Sector Rotation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  h1 { color: #1f77b4; }
  .stMetric { background: #f0f4f8; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# CACHED BACKTEST
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Running backtest ... (first run can take a few minutes)")
def _run(start: str, end: str, top_n: int, capital: int) -> BacktestRunner:
    r = BacktestRunner(start_date=start, end_date=end, top_n=top_n, capital=capital)
    r.run()
    return r


# ----------------------------------------------------------------------
# PLOTLY HELPERS
# ----------------------------------------------------------------------
def _fig_equity(runner: BacktestRunner):
    eq = runner.portfolio.get_equity()
    eq["Date"] = pd.to_datetime(eq["Date"])

    bench = runner.data["NIFTYBEES"]
    b = bench.loc[eq["Date"].min():eq["Date"].max()]
    b_norm = (b["Close"] / b["Close"].iloc[0]) * runner.initial_capital

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["Date"], y=eq["Portfolio_Value"],
                             name="Strategy", line=dict(color="#2E86AB", width=2.5)))
    fig.add_trace(go.Scatter(x=b.index, y=b_norm,
                             name="NIFTYBEES", line=dict(color="#A23B72", width=2, dash="dash")))
    fig.update_layout(title="Equity Curve", xaxis_title="Date",
                      yaxis_title="Value", hovermode="x unified",
                      height=420, template="plotly_white")
    return fig


def _fig_drawdown(runner: BacktestRunner):
    eq = runner.portfolio.get_equity()
    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.set_index("Date")
    rm = eq["Portfolio_Value"].expanding().max()
    dd = (eq["Portfolio_Value"] - rm) / rm * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown",
                             line=dict(color="#E63946")))
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="DD %",
                      height=300, template="plotly_white")
    return fig


def _fig_monthly_bars(runner: BacktestRunner):
    mr = runner.portfolio.monthly_returns().dropna()
    colors = ["#06D6A0" if v > 0 else "#EF476F" for v in mr["Monthly_Return"]]
    fig = go.Figure(data=go.Bar(x=pd.to_datetime(mr["Date"]), y=mr["Monthly_Return"],
                                marker_color=colors))
    fig.update_layout(title="Monthly Returns", xaxis_title="Date", yaxis_title="Return %",
                      height=320, template="plotly_white")
    return fig


def _fig_sector_rotation(runner: BacktestRunner):
    a = runner.allocations.copy()
    a["Month"] = pd.to_datetime(a["Selected_Date"]).dt.strftime("%Y-%m")
    pivot = a.pivot_table(index="ETF", columns="Month", values="Weight", fill_value=0) * 100

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="YlOrRd", colorbar=dict(title="Alloc %")
    ))
    fig.update_layout(title="Sector Rotation", xaxis_title="Month", yaxis_title="ETF",
                      height=400, template="plotly_white",
                      xaxis=dict(tickangle=45))
    return fig


# ----------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------
def main():
    st.title("Indian ETF Sector Rotation System")
    st.caption("AI-powered monthly rotation across Indian sector ETFs")

    # -- sidebar ------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        start = st.date_input("Start Date", value=datetime(2015, 1, 1))
        end   = st.date_input("End Date",   value=datetime.now())
        top_n = st.slider("Top N ETFs", 2, 5, 3)
        capital = st.number_input("Initial Capital (INR)", value=1_000_000, step=100_000, min_value=100_000)
        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        **How it works**
        - XGBoost ML predicts outperformance
        - Technical score: RSI, MACD, ADX, Bollinger
        - FinalScore = 0.6 x ML + 0.4 x Tech
        - Top ETFs chosen monthly; 10% cash reserve
        """)

    # -- trigger / cache ----------------------------------------------
    cache_key = f"{start}|{end}|{top_n}|{capital}"
    if "cache_key" not in st.session_state or st.session_state["cache_key"] != cache_key or run_btn:
        st.session_state["cache_key"] = cache_key
        _run.clear()

    runner: BacktestRunner = _run(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        top_n,
        capital,
    )

    if not runner.performance:
        st.error("Backtest returned no results. Check data availability.")
        return

    p = runner.performance

    # ==================================================================
    # TABS
    # ==================================================================
    tab_over, tab_perf, tab_alloc, tab_trades, tab_dl = st.tabs([
        "Overview", "Performance", "Allocation", "Trades", "Download"
    ])

    # -- Overview -----------------------------------------------------
    with tab_over:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return",  f"{p['Total_Return_%']} %",
                  delta=f"vs Bench {p['Benchmark_Total_Return_%']} %")
        c2.metric("CAGR",          f"{p['CAGR_%']} %",
                  delta=f"Alpha = {p['Alpha_%']} %")
        c3.metric("Sharpe",        f"{p['Sharpe_Ratio']}")
        c4.metric("Max Drawdown",  f"{p['Max_Drawdown_%']} %")

        st.plotly_chart(_fig_equity(runner), use_container_width=True)

        col_dd, col_risk = st.columns([2, 1])
        with col_dd:
            st.plotly_chart(_fig_drawdown(runner), use_container_width=True)
        with col_risk:
            st.subheader("Risk Summary")
            for label, key in [("Annual Vol", "Annual_Volatility_%"),
                               ("Sortino", "Sortino_Ratio"),
                               ("Calmar", "Calmar_Ratio"),
                               ("Monthly Win %", "Monthly_Win_Rate_%"),
                               ("Hit Rate %", "Hit_Rate_%")]:
                st.markdown(f"**{label}:** {p.get(key, '-')}")

    # -- Performance --------------------------------------------------
    with tab_perf:
        col_bars, col_rot = st.columns(2)
        with col_bars:
            st.plotly_chart(_fig_monthly_bars(runner), use_container_width=True)
        with col_rot:
            st.plotly_chart(_fig_sector_rotation(runner), use_container_width=True)

        st.subheader("All Metrics")
        st.dataframe(
            pd.DataFrame(list(p.items()), columns=["Metric", "Value"]),
            use_container_width=True, hide_index=True
        )

        if not runner.importance_df.empty:
            st.subheader("Feature Importance")
            fig_imp = px.bar(runner.importance_df.head(12), x="Importance", y="Feature",
                             orientation="h", color="Importance",
                             color_continuous_scale="Blues")
            fig_imp.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_imp, use_container_width=True)

    # -- Allocation ---------------------------------------------------
    with tab_alloc:
        cur = runner.current_allocation()
        if cur is not None and not cur.empty:
            st.subheader("Current Month Allocation")

            # Pie chart
            cash_w = 1.0 - cur["Weight"].sum()
            pie_df = pd.concat([
                cur[["ETF", "Weight"]],
                pd.DataFrame([{"ETF": "CASH", "Weight": cash_w}])
            ], ignore_index=True)
            fig_pie = px.pie(pie_df, names="ETF", values="Weight",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Detail table
            display = cur.copy()
            display["Weight %"] = (display["Weight"] * 100).round(2)
            st.dataframe(
                display[["ETF", "Weight %", "FinalScore", "ML_Probability", "TechScore"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("No current allocation available.")

        # Next-month predictions
        st.subheader("Next-Month Top Predictions")
        preds = runner.next_month_predictions()
        if preds is not None and not preds.empty:
            st.dataframe(preds.head(9), use_container_width=True)
        else:
            st.info("No predictions available.")

    # -- Trades -------------------------------------------------------
    with tab_trades:
        trades = runner.portfolio.get_trades()
        if trades.empty:
            st.warning("No trades recorded.")
        else:
            sells = trades[trades["Action"] == "SELL"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Trades", len(trades))
            c2.metric("Avg Trade Return", f"{p['Avg_Trade_Return_%']} %")
            c3.metric("Trade Win Rate", f"{p['Trade_Win_Rate_%']} %")

            st.subheader("Filter")
            actions = st.multiselect("Action", trades["Action"].unique(), default=trades["Action"].unique())
            etfs    = st.multiselect("ETF",    sorted(trades["ETF"].unique()), default=sorted(trades["ETF"].unique()))
            filtered = trades[trades["Action"].isin(actions) & trades["ETF"].isin(etfs)]
            st.dataframe(filtered.sort_values("Date", ascending=False),
                         use_container_width=True, hide_index=True)

    # -- Download -----------------------------------------------------
    with tab_dl:
        st.subheader("Export Excel Report")
        st.markdown("Contains: Summary, Performance, Trade Log, Monthly Returns, Allocations, Feature Importance")

        if st.button("Generate & Download Report", type="primary"):
            with st.spinner("Generating ..."):
                path = gen_excel(runner, output_dir="reports")
            with open(path, "rb") as f:
                st.download_button(
                    "Download Excel Report",
                    data=f,
                    file_name=os.path.basename(path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


if __name__ == "__main__":
    main()
