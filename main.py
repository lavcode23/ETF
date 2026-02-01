"""
main.py — Run the complete backtest pipeline and generate reports.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.backtest_runner import BacktestRunner
from reports.excel_report import generate as gen_excel
from reports.charts import generate_all as gen_charts


def main():
    runner = BacktestRunner(
        start_date="2015-01-01",
        top_n=3,
        capital=1_000_000,
    )
    runner.run()

    print("\n[Reports] Excel …")
    gen_excel(runner)

    print("[Reports] Charts …")
    gen_charts(runner)

    print("\n✅  Done.  Launch dashboard with:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
