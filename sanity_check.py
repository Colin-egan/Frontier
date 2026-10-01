"""Quick manual check of analysis/data.py and analysis/portfolio.py output.

Not a test file — run directly and eyeball the numbers against a known
source (e.g. Yahoo Finance / portfoliovisualizer.com) before trusting them.
"""

from analysis.data import get_daily_returns, get_price_history
from analysis.portfolio import portfolio_stats

TICKERS = ["AAPL", "MSFT", "SPY"]
WEIGHTS = {"AAPL": 0.4, "MSFT": 0.4, "SPY": 0.2}
START = "2023-01-01"
END = "2024-01-01"

if __name__ == "__main__":
    prices = get_price_history(TICKERS, START, END)
    print("Price history shape:", prices.shape)
    print(prices.head())
    print()

    returns = get_daily_returns(prices)
    stats = portfolio_stats(returns, WEIGHTS)

    print(f"Annualized return:     {stats['annualized_return']:.2%}")
    print(f"Annualized volatility: {stats['annualized_volatility']:.2%}")
    print(f"Sharpe ratio:          {stats['sharpe_ratio']:.2f}")
    print()
    print("Correlation matrix:")
    print(stats["correlation_matrix"])
