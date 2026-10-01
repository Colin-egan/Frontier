# Portfolio Analyzer

A tool that analyzes a given portfolio's stats and recommends stocks that would improve its risk-adjusted return (Sharpe ratio).

## What it does

Given a set of holdings (tickers + weights), it:

1. Pulls historical daily prices (via [yfinance](https://github.com/ranaroussi/yfinance)) and computes the portfolio's annualized return, annualized volatility, Sharpe ratio, and pairwise correlation matrix.
2. Screens a candidate universe (the S&P 500 by default) and ranks each candidate by how much it would improve the portfolio's Sharpe ratio if added.

## Methodology

### Portfolio stats

- **Annualized return**: the portfolio's daily returns are `returns @ weights`, averaged, then compounded to an annual figure: `(1 + mean_daily_return)^252 - 1`.
- **Annualized volatility**: `sqrt(wᵀ Σ w)` where `Σ` is the covariance matrix of daily returns, annualized by `sqrt(252)`.
- **Sharpe ratio**: `(annualized_return - risk_free_rate) / annualized_volatility`.
- **Correlation matrix**: pairwise Pearson correlation of daily returns, used to visualize diversification (or the lack of it) across holdings.

### Marginal Sharpe contribution screener

For each candidate ticker not already held, the screener asks: *what happens to this portfolio's Sharpe ratio if I add a little of this stock, funded from my existing holdings?*

1. **Baseline**: compute the current portfolio's Sharpe ratio from its actual weights.
2. **Perturb**: build a new weight vector by shrinking every existing holding by `(1 - ε)` (ε defaults to 5%) and assigning that freed-up ε to the candidate. The portfolio stays fully invested.
3. **Re-evaluate**: recompute the Sharpe ratio for the new weight vector using the full covariance matrix (existing holdings + candidate), which captures the candidate's diversification effect, not just its standalone risk/return.
4. **Rank**: sort candidates by `Δ Sharpe = new_sharpe - baseline_sharpe`, descending.

This is a direct simulation rather than a first-order analytical approximation (e.g. closed-form marginal risk contribution formulas) — it's exact for the given ε, and easy to defend: *"I tested what actually happens to the portfolio if I added 5% of this stock, funded from the existing holdings."* It's also cheap to compute at scale, since each candidate only requires matrix operations on (portfolio size + 1) assets, not the full universe.

The candidate universe defaults to current S&P 500 constituents (scraped from Wikipedia), filtered to exclude tickers with substantial missing price history in the lookback window — a proxy for illiquid or newly-listed names.

### Limitations

- Uses trailing historical returns/covariance as a stand-in for expected future risk/return — a standard simplification, but it means the tool is backward-looking, not predictive.
- Ignores transaction costs, taxes, and position size constraints.
- The ε perturbation size affects the ranking at the margin; a larger ε moves further from a true marginal (infinitesimal) analysis.

## Project layout

```
analysis/
  data.py          # yfinance pulls, caching, S&P 500 universe, liquidity filter
  portfolio.py      # return, volatility, Sharpe ratio, correlation matrix
  screener.py       # marginal Sharpe contribution ranking
app/
  streamlit_app.py  # UI only, imports from analysis/
tests/
```

`analysis/` has no dependency on `app/` or Streamlit, so the UI layer can be swapped (e.g. for a FastAPI + Next.js frontend) without touching the analysis logic.

## Running locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
