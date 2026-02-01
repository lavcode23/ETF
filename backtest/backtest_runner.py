"""
backtest/backtest_runner.py
Orchestrates the full pipeline: fetch -> indicators -> features -> train -> predict -> score -> allocate -> simulate -> metrics.
"""
import pandas as pd
from data.fetch_data import fetch_all, align
from features.indicators import add_indicators_all
from features.monthly_features import build_monthly, add_targets
from models.train_model import walk_forward, feature_importance
from models.predict import predict_all
from strategies.technical_score import add_scores
from strategies.sector_rotation import allocate
from backtest.portfolio import simulate
from backtest.performance import compute


class BacktestRunner:
    """Hold every intermediate and final result so the dashboard can read them."""

    def __init__(self, start_date="2013-01-01", end_date=None, top_n=3, capital=1_000_000):
        self.start_date = start_date
        self.end_date = end_date
        self.top_n = top_n
        self.initial_capital = capital

        # filled by run()
        self.data: dict[str, pd.DataFrame] = {}
        self.monthly: pd.DataFrame = pd.DataFrame()
        self.scored: pd.DataFrame = pd.DataFrame()
        self.allocations: pd.DataFrame = pd.DataFrame()
        self.portfolio = None
        self.performance: dict = {}
        self.models_dict: dict = {}
        self.feat_cols: list = []
        self.importance_df: pd.DataFrame = pd.DataFrame()

    # -- public -------------------------------------------------------
    def run(self):
        print("=" * 70)
        print("BACKTEST PIPELINE")
        print("=" * 70)

        print("\n[1] Fetching data ...")
        raw = fetch_all(self.start_date, self.end_date)
        self.data = align(raw)

        print("\n[2] Technical indicators ...")
        indic = add_indicators_all(self.data)

        print("\n[3] Monthly features ...")
        monthly = build_monthly(indic)
        monthly_t = add_targets(monthly)

        print("\n[4] Walk-forward training ...")
        self.models_dict, history, self.feat_cols = walk_forward(monthly_t)
        self.importance_df = feature_importance(self.models_dict)

        print("\n[5] ML predictions ...")
        predicted = predict_all(monthly_t, self.models_dict, self.feat_cols)

        print("\n[6] Technical + final scores ...")
        self.scored = add_scores(predicted)
        self.monthly = monthly_t

        print("\n[7] Sector rotation allocations ...")
        self.allocations = allocate(self.scored, top_n=self.top_n)

        print("\n[8] Portfolio simulation ...")
        self.portfolio = simulate(self.data, self.allocations, self.initial_capital)

        print("\n[9] Performance metrics ...")
        self.performance = compute(self.portfolio, self.data["NIFTYBEES"])

        self._print_summary()

    # -- convenience accessors used by dashboard ----------------------
    def current_allocation(self) -> pd.DataFrame | None:
        if self.allocations.empty:
            return None
        last = self.allocations["Selected_Date"].max()
        return self.allocations[self.allocations["Selected_Date"] == last][
            ["ETF", "Weight", "FinalScore", "ML_Probability", "TechScore"]
        ]

    def next_month_predictions(self) -> pd.DataFrame | None:
        if self.scored.empty:
            return None
        last_date = self.scored.index.max()
        top = self.scored.loc[self.scored.index == last_date].sort_values("FinalScore", ascending=False)
        cols = ["ETF", "FinalScore", "ML_Probability", "TechScore", "RSI_14", "MACD_Hist", "ADX_14"]
        return top[[c for c in cols if c in top.columns]]

    # -- private ------------------------------------------------------
    def _print_summary(self):
        p = self.performance
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"  Total Return   : {p.get('Total_Return_%', 0):>9.2f} %")
        print(f"  CAGR           : {p.get('CAGR_%', 0):>9.2f} %   (Benchmark {p.get('Benchmark_CAGR_%', 0):.2f} %)")
        print(f"  Alpha          : {p.get('Alpha_%', 0):>9.2f} %")
        print(f"  Sharpe         : {p.get('Sharpe_Ratio', 0):>9.3f}")
        print(f"  Sortino        : {p.get('Sortino_Ratio', 0):>9.3f}")
        print(f"  Max Drawdown   : {p.get('Max_Drawdown_%', 0):>9.2f} %")
        print(f"  Monthly Win    : {p.get('Monthly_Win_Rate_%', 0):>9.2f} %")
        print(f"  Hit Rate       : {p.get('Hit_Rate_%', 0):>9.2f} %")
        print(f"  Trades         : {p.get('Total_Trades', 0):>9}")
        print("=" * 70)
