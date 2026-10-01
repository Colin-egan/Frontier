import numpy as np
import pandas as pd
import pytest

from analysis.portfolio import (
    annualized_return,
    annualized_volatility,
    correlation_matrix,
    portfolio_stats,
    sharpe_ratio,
)


def test_annualized_return_constant_daily_return():
    daily = 0.001
    returns = pd.DataFrame({"A": [daily] * 252})
    result = annualized_return(returns, {"A": 1.0})
    assert result == pytest.approx((1 + daily) ** 252 - 1)


def test_annualized_volatility_zero_for_constant_returns():
    returns = pd.DataFrame({"A": [0.001] * 100, "B": [0.002] * 100})
    result = annualized_volatility(returns, {"A": 0.5, "B": 0.5})
    assert result == pytest.approx(0.0, abs=1e-10)


def test_annualized_volatility_single_asset_matches_std():
    rng = np.random.RandomState(42)
    daily_returns = rng.normal(loc=0.0005, scale=0.01, size=500)
    returns = pd.DataFrame({"A": daily_returns})
    expected = daily_returns.std() * np.sqrt(252)
    result = annualized_volatility(returns, {"A": 1.0})
    assert result == pytest.approx(expected, rel=1e-6)


def test_sharpe_ratio_known_values():
    assert sharpe_ratio(ann_return=0.10, ann_volatility=0.20, risk_free_rate=0.02) == pytest.approx(0.4)


def test_sharpe_ratio_zero_volatility_is_nan():
    assert np.isnan(sharpe_ratio(ann_return=0.10, ann_volatility=0.0))


def test_correlation_matrix_perfectly_correlated():
    series = pd.Series(range(1, 101), dtype=float)
    returns = pd.DataFrame({"A": series, "B": series * 2})
    corr = correlation_matrix(returns)
    assert corr.loc["A", "B"] == pytest.approx(1.0)


def test_correlation_matrix_perfectly_anti_correlated():
    series = pd.Series(range(1, 101), dtype=float)
    returns = pd.DataFrame({"A": series, "B": -series})
    corr = correlation_matrix(returns)
    assert corr.loc["A", "B"] == pytest.approx(-1.0)


def test_portfolio_stats_known_portfolio_reasonable_sharpe():
    rng = np.random.RandomState(7)
    n_days = 756
    returns = pd.DataFrame(
        {
            "A": rng.normal(0.0006, 0.012, n_days),
            "B": rng.normal(0.0004, 0.009, n_days),
        }
    )
    stats = portfolio_stats(returns, {"A": 0.6, "B": 0.4})

    assert -1.0 < stats["sharpe_ratio"] < 5.0
    assert stats["annualized_volatility"] > 0
    assert stats["correlation_matrix"].shape == (2, 2)
