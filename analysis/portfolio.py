"""Portfolio statistics: annualized return, volatility, Sharpe ratio, correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(returns: pd.DataFrame, weights: dict[str, float]) -> float:
    """Annualized portfolio return from daily returns and ticker weights."""
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    daily_portfolio_returns = returns @ w
    mean_daily = daily_portfolio_returns.mean()
    return float((1 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1)


def annualized_volatility(returns: pd.DataFrame, weights: dict[str, float]) -> float:
    """Annualized portfolio volatility (std dev) from daily returns and weights."""
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    cov_daily = returns.cov()
    variance_daily = w @ cov_daily @ w
    return float(np.sqrt(variance_daily) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(ann_return: float, ann_volatility: float, risk_free_rate: float = 0.02) -> float:
    """Annualized Sharpe ratio given annualized return and volatility."""
    if ann_volatility == 0:
        return float("nan")
    return (ann_return - risk_free_rate) / ann_volatility


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation matrix of daily returns."""
    return returns.corr()


def portfolio_stats(
    returns: pd.DataFrame,
    weights: dict[str, float],
    risk_free_rate: float = 0.02,
) -> dict:
    """Compute the full stats bundle for a portfolio: return, vol, Sharpe, correlation."""
    ann_return = annualized_return(returns, weights)
    ann_volatility = annualized_volatility(returns, weights)
    sharpe = sharpe_ratio(ann_return, ann_volatility, risk_free_rate)
    corr = correlation_matrix(returns)
    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_volatility,
        "sharpe_ratio": sharpe,
        "correlation_matrix": corr,
    }
