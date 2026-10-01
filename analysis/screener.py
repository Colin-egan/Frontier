"""Candidate ranking via marginal Sharpe contribution.

For each candidate ticker, simulate adding it to the existing portfolio at a
small weight (funded pro-rata from current holdings) and measure the change
in Sharpe ratio. See the module docstring in portfolio.py for the underlying
stats; this module only adds the perturb-and-rank logic.
"""

from __future__ import annotations

import pandas as pd

from analysis.portfolio import annualized_return, annualized_volatility, portfolio_stats, sharpe_ratio

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


def simulate_trade(
    current_weights: dict[str, float],
    candidate: str,
    amount: float,
    total_value: float,
    funding: str,
    returns: pd.DataFrame,
    source_ticker: str | None = None,
    risk_free_rate: float = 0.02,
) -> dict:
    """Simulate funding `candidate` with `amount` dollars.

    `funding` is one of:
    - "new_money": portfolio grows by `amount`; existing holdings dilute
      proportionally, candidate gets `amount / new_total`.
    - "auto_prorata": every holding shrinks proportionally to fund the
      candidate (the same math `_shrink_and_add` uses for ranking, sized to
      a real dollar amount instead of a fixed epsilon).
    - "specific_holding" (requires `source_ticker`): only that holding is
      trimmed to fund the candidate, capped at its current dollar value.

    Returns before/after portfolio_stats, the resulting weights, the
    resulting total portfolio value, and whether/how much the requested
    amount was capped.
    """
    if funding == "new_money":
        new_total_value = total_value + amount
        new_weights = {t: w * total_value / new_total_value for t, w in current_weights.items()}
        new_weights[candidate] = new_weights.get(candidate, 0.0) + amount / new_total_value
        capped, capped_amount = False, amount

    elif funding == "auto_prorata":
        epsilon = amount / total_value
        new_weights = _shrink_and_add(current_weights, candidate, epsilon)
        new_total_value = total_value
        capped, capped_amount = False, amount

    elif funding == "specific_holding":
        if source_ticker is None or source_ticker not in current_weights:
            raise ValueError("source_ticker must be a currently held ticker")
        source_value = current_weights[source_ticker] * total_value
        capped_amount = min(amount, source_value)
        capped = capped_amount < amount
        delta_weight = capped_amount / total_value
        new_weights = dict(current_weights)
        new_weights[source_ticker] -= delta_weight
        new_weights[candidate] = new_weights.get(candidate, 0.0) + delta_weight
        new_total_value = total_value

    else:
        raise ValueError(f"Unknown funding mode: {funding}")

    before_cols = [t for t in current_weights if t in returns.columns]
    before = portfolio_stats(returns[before_cols], current_weights, risk_free_rate)

    after_cols = [t for t in new_weights if t in returns.columns]
    after = portfolio_stats(returns[after_cols], new_weights, risk_free_rate)

    return {
        "before": before,
        "after": after,
        "new_weights": new_weights,
        "new_total_value": new_total_value,
        "capped": capped,
        "capped_amount": capped_amount,
    }
