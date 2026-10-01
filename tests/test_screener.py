import numpy as np
import pandas as pd

from analysis.screener import rank_candidates


def _sample_returns(n: int = 500) -> pd.DataFrame:
    rng = np.random.RandomState(1)
    market = rng.normal(0.0004, 0.01, n)

    port = market + rng.normal(0.0, 0.002, n)
    good = rng.normal(0.0015, 0.012, n)  # independent of market, higher mean return
    bad = rng.normal(-0.0010, 0.01, n)  # negative mean return

    return pd.DataFrame({"PORT": port, "GOOD": good, "BAD": bad})


def test_rank_candidates_orders_by_sharpe_improvement():
    returns = _sample_returns()
    ranked = rank_candidates(returns, {"PORT": 1.0}, ["GOOD", "BAD"])

    assert list(ranked["ticker"]) == ["GOOD", "BAD"]
    assert ranked.iloc[0]["delta_sharpe"] > 0
    assert ranked.iloc[-1]["delta_sharpe"] < ranked.iloc[0]["delta_sharpe"]


def test_rank_candidates_skips_already_held_tickers():
    returns = _sample_returns()
    ranked = rank_candidates(returns, {"PORT": 1.0}, ["PORT", "GOOD"])

    assert "PORT" not in set(ranked["ticker"])
    assert "GOOD" in set(ranked["ticker"])


def test_rank_candidates_skips_tickers_missing_from_returns():
    returns = _sample_returns()
    ranked = rank_candidates(returns, {"PORT": 1.0}, ["GOOD", "MISSING_TICKER"])

    assert "MISSING_TICKER" not in set(ranked["ticker"])
    assert "GOOD" in set(ranked["ticker"])


def test_rank_candidates_empty_when_no_valid_candidates():
    returns = _sample_returns()
    ranked = rank_candidates(returns, {"PORT": 1.0}, ["PORT", "MISSING_TICKER"])

    assert ranked.empty


def test_rank_candidates_result_columns():
    returns = _sample_returns()
    ranked = rank_candidates(returns, {"PORT": 1.0}, ["GOOD", "BAD"])

    expected_columns = {"ticker", "delta_sharpe", "new_sharpe", "delta_volatility", "new_volatility", "new_return"}
    assert expected_columns.issubset(ranked.columns)
