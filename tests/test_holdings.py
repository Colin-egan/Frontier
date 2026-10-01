from datetime import date
from io import StringIO

import pandas as pd
import pytest

from analysis.holdings import (
    Holding,
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
