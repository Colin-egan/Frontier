"""Candidate ranking via marginal Sharpe contribution.

For each candidate ticker, simulate adding it to the existing portfolio at a
small weight (funded pro-rata from current holdings) and measure the change
in Sharpe ratio. See the module docstring in portfolio.py for the underlying
stats; this module only adds the perturb-and-rank logic.
"""

from __future__ import annotations

import pandas as pd

from analysis.portfolio import annualized_return, annualized_volatility, sharpe_ratio

DEFAULT_EPSILON = 0.05


def _shrink_and_add(current_weights: dict[str, float], candidate: str, epsilon: float) -> dict[str, float]:
    new_weights = {t: w * (1 - epsilon) for t, w in current_weights.items()}
    new_weights[candidate] = new_weights.get(candidate, 0.0) + epsilon
    return new_weights


def rank_candidates(
    returns: pd.DataFrame,
    current_weights: dict[str, float],
    candidates: list[str],
    epsilon: float = DEFAULT_EPSILON,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Rank candidate tickers by their marginal effect on portfolio Sharpe ratio.

    `returns` must contain daily return columns for every ticker in
    `current_weights` plus every ticker in `candidates`. Candidates missing
    from `returns` (no data) or already held are skipped.

    Returns a DataFrame sorted by `delta_sharpe` descending, with columns:
    ticker, delta_sharpe, new_sharpe, delta_volatility, new_volatility, new_return.
    """
    held = set(current_weights)
    portfolio_returns = returns[[t for t in current_weights if t in returns.columns]]

    baseline_return = annualized_return(portfolio_returns, current_weights)
    baseline_volatility = annualized_volatility(portfolio_returns, current_weights)
    baseline_sharpe = sharpe_ratio(baseline_return, baseline_volatility, risk_free_rate)

    rows = []
    for ticker in candidates:
        if ticker in held or ticker not in returns.columns:
            continue

        new_weights = _shrink_and_add(current_weights, ticker, epsilon)
        subset = returns[list(new_weights)]

        new_return = annualized_return(subset, new_weights)
        new_volatility = annualized_volatility(subset, new_weights)
        new_sharpe = sharpe_ratio(new_return, new_volatility, risk_free_rate)

        rows.append(
            {
                "ticker": ticker,
                "delta_sharpe": new_sharpe - baseline_sharpe,
                "new_sharpe": new_sharpe,
                "delta_volatility": new_volatility - baseline_volatility,
                "new_volatility": new_volatility,
                "new_return": new_return,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("delta_sharpe", ascending=False).reset_index(drop=True)
