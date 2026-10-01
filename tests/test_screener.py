import numpy as np
import pandas as pd
import pytest

from analysis.screener import rank_candidates, simulate_trade


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


def test_simulate_trade_new_money_dilutes_existing_and_adds_candidate():
    returns = _sample_returns()
    result = simulate_trade({"PORT": 1.0}, "GOOD", amount=1000.0, total_value=4000.0, funding="new_money", returns=returns)

    assert result["new_weights"] == pytest.approx({"PORT": 0.8, "GOOD": 0.2})
    assert result["new_total_value"] == pytest.approx(5000.0)
    assert result["capped"] is False
    assert set(result["before"]) == {"annualized_return", "annualized_volatility", "sharpe_ratio", "correlation_matrix"}


def test_simulate_trade_auto_prorata_matches_shrink_and_add():
    returns = _sample_returns()
    result = simulate_trade({"PORT": 1.0}, "GOOD", amount=400.0, total_value=4000.0, funding="auto_prorata", returns=returns)

    assert result["new_weights"] == pytest.approx({"PORT": 0.9, "GOOD": 0.1})
    assert result["new_total_value"] == pytest.approx(4000.0)


def test_simulate_trade_specific_holding_moves_only_source():
    returns = _sample_returns()
    result = simulate_trade(
        {"PORT": 0.6, "BAD": 0.4}, "GOOD", amount=800.0, total_value=4000.0,
        funding="specific_holding", returns=returns, source_ticker="BAD",
    )

    assert result["new_weights"] == pytest.approx({"PORT": 0.6, "BAD": 0.2, "GOOD": 0.2})
    assert result["capped"] is False


def test_simulate_trade_specific_holding_caps_when_amount_exceeds_source_value():
    returns = _sample_returns()
    result = simulate_trade(
        {"PORT": 0.6, "BAD": 0.4}, "GOOD", amount=5000.0, total_value=4000.0,
        funding="specific_holding", returns=returns, source_ticker="BAD",
    )

    assert result["capped"] is True
    assert result["capped_amount"] == pytest.approx(1600.0)
    assert result["new_weights"]["BAD"] == pytest.approx(0.0)
    assert result["new_weights"]["GOOD"] == pytest.approx(0.4)


def test_simulate_trade_unknown_funding_raises():
    returns = _sample_returns()
    with pytest.raises(ValueError):
        simulate_trade({"PORT": 1.0}, "GOOD", 100.0, 1000.0, "bogus_mode", returns)


def test_simulate_trade_specific_holding_requires_valid_source_ticker():
    returns = _sample_returns()
    with pytest.raises(ValueError):
        simulate_trade({"PORT": 1.0}, "GOOD", 100.0, 1000.0, "specific_holding", returns, source_ticker="NOT_HELD")
