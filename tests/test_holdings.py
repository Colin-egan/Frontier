from datetime import date
from io import StringIO
from unittest.mock import patch

import pandas as pd
import pytest

from analysis.holdings import (
    Holding,
    compute_gain_loss,
    compute_value_history,
    holdings_from_dataframe,
    holdings_to_dataframe,
    holdings_to_weights,
    holdings_total_value,
    parse_holdings_csv,
)


def test_holdings_from_dataframe_parses_required_and_optional_fields():
    df = pd.DataFrame(
        [
            {"ticker": "aapl", "shares": 10, "cost_basis": 1200.0, "purchase_date": "2024-01-15"},
            {"ticker": "msft", "shares": 5, "cost_basis": None, "purchase_date": None},
        ]
    )
    holdings = holdings_from_dataframe(df)

    assert holdings[0] == Holding(ticker="AAPL", shares=10.0, cost_basis=1200.0, purchase_date=date(2024, 1, 15))
    assert holdings[1] == Holding(ticker="MSFT", shares=5.0, cost_basis=None, purchase_date=None)


def test_holdings_from_dataframe_missing_ticker_raises():
    df = pd.DataFrame([{"ticker": None, "shares": 10}])
    with pytest.raises(ValueError, match="ticker"):
        holdings_from_dataframe(df)


def test_holdings_from_dataframe_invalid_shares_raises():
    df = pd.DataFrame([{"ticker": "AAPL", "shares": "not-a-number"}])
    with pytest.raises(ValueError, match="shares"):
        holdings_from_dataframe(df)


def test_holdings_from_dataframe_missing_required_columns_raises():
    df = pd.DataFrame([{"ticker": "AAPL"}])
    with pytest.raises(ValueError, match="shares"):
        holdings_from_dataframe(df)


def test_holdings_to_dataframe_round_trip():
    holdings = [
        Holding(ticker="AAPL", shares=10.0, cost_basis=1200.0, purchase_date=date(2024, 1, 15)),
        Holding(ticker="MSFT", shares=5.0, cost_basis=None, purchase_date=None),
    ]
    round_tripped = holdings_from_dataframe(holdings_to_dataframe(holdings))
    assert round_tripped == holdings


def test_parse_holdings_csv_reads_file():
    csv_text = "ticker,shares,cost_basis,purchase_date\nAAPL,10,1200.0,2024-01-15\n"
    holdings = parse_holdings_csv(StringIO(csv_text))
    assert holdings == [Holding(ticker="AAPL", shares=10.0, cost_basis=1200.0, purchase_date=date(2024, 1, 15))]


def test_holdings_to_weights_normalizes_by_dollar_value():
    holdings = [Holding(ticker="AAPL", shares=10.0), Holding(ticker="MSFT", shares=5.0)]
    weights = holdings_to_weights(holdings, {"AAPL": 100.0, "MSFT": 200.0})
    assert weights == pytest.approx({"AAPL": 0.5, "MSFT": 0.5})


def test_holdings_to_weights_skips_missing_prices():
    holdings = [Holding(ticker="AAPL", shares=10.0), Holding(ticker="MSFT", shares=5.0)]
    weights = holdings_to_weights(holdings, {"AAPL": 100.0})
    assert weights == pytest.approx({"AAPL": 1.0})


def test_holdings_to_weights_raises_when_no_usable_prices():
    holdings = [Holding(ticker="AAPL", shares=10.0)]
    with pytest.raises(ValueError):
        holdings_to_weights(holdings, {})


def test_holdings_total_value_sums_only_priced_holdings():
    holdings = [Holding(ticker="AAPL", shares=10.0), Holding(ticker="MSFT", shares=5.0)]
    total = holdings_total_value(holdings, {"AAPL": 100.0})
    assert total == pytest.approx(1000.0)


def test_compute_gain_loss_known_values():
    holdings = [Holding(ticker="AAPL", shares=10.0, cost_basis=1000.0)]
    row = compute_gain_loss(holdings, {"AAPL": 120.0}).iloc[0]
    assert row["current_value"] == pytest.approx(1200.0)
    assert row["dollar_gain"] == pytest.approx(200.0)
    assert row["pct_gain"] == pytest.approx(0.2)


def test_compute_gain_loss_missing_cost_basis_is_nan():
    holdings = [Holding(ticker="AAPL", shares=10.0, cost_basis=None)]
    row = compute_gain_loss(holdings, {"AAPL": 120.0}).iloc[0]
    assert row["current_value"] == pytest.approx(1200.0)
    assert pd.isna(row["dollar_gain"])
    assert pd.isna(row["pct_gain"])


def test_compute_value_history_sums_across_holdings_from_purchase_date():
    holdings = [
        Holding(ticker="AAPL", shares=10.0, purchase_date=date(2024, 1, 1)),
        Holding(ticker="MSFT", shares=2.0, purchase_date=date(2024, 1, 3)),
    ]

    def fake_get_price_history(tickers, start, end, use_cache=True):
        if tickers == ["AAPL"]:
            idx = pd.date_range("2024-01-01", "2024-01-04")
            return pd.DataFrame({"AAPL": [100.0, 101.0, 102.0, 103.0]}, index=idx)
        if tickers == ["MSFT"]:
            idx = pd.date_range("2024-01-03", "2024-01-04")
            return pd.DataFrame({"MSFT": [300.0, 305.0]}, index=idx)
        raise AssertionError(f"Unexpected tickers: {tickers}")

    with patch("analysis.holdings.get_price_history", side_effect=fake_get_price_history):
        history = compute_value_history(holdings, today=date(2024, 1, 4))

    assert history[pd.Timestamp("2024-01-01")] == pytest.approx(1000.0)
    assert history[pd.Timestamp("2024-01-03")] == pytest.approx(10 * 102.0 + 2 * 300.0)
    assert history[pd.Timestamp("2024-01-04")] == pytest.approx(10 * 103.0 + 2 * 305.0)


def test_compute_value_history_empty_when_no_purchase_dates():
    holdings = [Holding(ticker="AAPL", shares=10.0)]
    assert compute_value_history(holdings).empty
