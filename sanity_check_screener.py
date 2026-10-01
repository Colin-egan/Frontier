"""Quick manual check of analysis/screener.py's marginal Sharpe ranking.

Not a test file — run directly and eyeball the ranking. A low-correlation
asset (e.g. GLD, a gold ETF) should rank well despite modest standalone
returns, since it adds diversification; a high-correlation, no-better-return
asset should rank poorly.
"""

from analysis.data import get_daily_returns, get_price_history
from analysis.portfolio import annualized_return, annualized_volatility, sharpe_ratio
from analysis.screener import rank_candidates

PORTFOLIO_TICKERS = ["AAPL", "MSFT", "SPY"]
WEIGHTS = {"AAPL": 0.4, "MSFT": 0.4, "SPY": 0.2}
CANDIDATES = ["GOOGL", "AMZN", "NVDA", "TSLA", "JNJ", "XOM", "GLD"]
START = "2023-01-01"
END = "2024-01-01"

if __name__ == "__main__":
    all_tickers = sorted(set(PORTFOLIO_TICKERS) | set(CANDIDATES))
    prices = get_price_history(all_tickers, START, END)
    returns = get_daily_returns(prices)

    baseline_return = annualized_return(returns[PORTFOLIO_TICKERS], WEIGHTS)
    baseline_vol = annualized_volatility(returns[PORTFOLIO_TICKERS], WEIGHTS)
    baseline_sharpe = sharpe_ratio(baseline_return, baseline_vol)
    print(f"Baseline Sharpe: {baseline_sharpe:.3f} (return {baseline_return:.2%}, vol {baseline_vol:.2%})")
    print()

    ranked = rank_candidates(returns, WEIGHTS, CANDIDATES)
    print(ranked.to_string(index=False))
