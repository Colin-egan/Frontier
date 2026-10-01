# Dollar-Based Holdings & Trade Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the typed `ticker:weight` portfolio input with dollar-based holdings (CSV upload + editable table, including cost basis and purchase date), add a gain/loss table and a performance-over-time chart, and let the user simulate acting on a recommended candidate — choosing how to fund it and previewing the before/after impact plus a 5-year forward projection.

**Architecture:** A new `analysis/holdings.py` module owns the `Holding` data model, CSV/dataframe parsing, and dollar↔weight conversion, bridging into the existing weight-based `analysis/portfolio.py` and `analysis/screener.py` unchanged. `screener.py` gains one new function, `simulate_trade()`, reusing the existing `_shrink_and_add` perturbation logic for the pro-rata funding mode. `portfolio.py` gains one new function, `project_value()`, for the deterministic 5-year compounding. `app/streamlit_app.py` is rewritten to wire these together; no other UI logic changes.

**Tech Stack:** Python 3.13, pandas, Streamlit (`st.file_uploader`, `st.data_editor`), matplotlib. No new dependencies.

## Global Constraints

- No new third-party dependencies — everything needed (pandas, matplotlib, streamlit) is already in `requirements.txt`.
- CSV/table columns are exactly: `ticker`, `shares` (required), `cost_basis`, `purchase_date` (optional per row — missing/blank is valid, not an error).
- `cost_basis` is the total dollars paid for the position, not a per-share price.
- Funding mode string literals are exactly: `"new_money"`, `"auto_prorata"`, `"specific_holding"`.
- The 5-year projection is deterministic compounding only (`total_value * (1 + annualized_return) ** years`) — no Monte Carlo, no confidence interval.
- Trade simulation is preview-only — it never writes back to the holdings table, session state, or any file.
- Existing functions in `analysis/portfolio.py` and `analysis/screener.py` (`annualized_return`, `annualized_volatility`, `sharpe_ratio`, `correlation_matrix`, `portfolio_stats`, `rank_candidates`, `_shrink_and_add`) keep their current signatures — do not modify them, only add alongside.
- Follow the existing code style: no comments except where a non-obvious constraint/invariant needs explaining (see existing files for the bar).

---

### Task 1: Holdings model, parsing, and weight conversion

**Files:**
- Create: `analysis/holdings.py`
- Test: `tests/test_holdings.py`

**Interfaces:**
- Produces: `Holding` (dataclass: `ticker: str`, `shares: float`, `cost_basis: float | None = None`, `purchase_date: date | None = None`), `holdings_from_dataframe(df: pd.DataFrame) -> list[Holding]`, `holdings_to_dataframe(holdings: list[Holding]) -> pd.DataFrame`, `parse_holdings_csv(file) -> list[Holding]`, `holdings_to_weights(holdings: list[Holding], current_prices: dict[str, float]) -> dict[str, float]`, `holdings_total_value(holdings: list[Holding], current_prices: dict[str, float]) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_holdings.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_holdings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.holdings'`

- [ ] **Step 3: Write the implementation**

Create `analysis/holdings.py`:

```python
"""Dollar-based holdings: parsing and dollar-to-weight conversion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class Holding:
    ticker: str
    shares: float
    cost_basis: float | None = None
    purchase_date: date | None = None


def holdings_from_dataframe(df: pd.DataFrame) -> list[Holding]:
    """Parse a DataFrame with columns ticker, shares, cost_basis, purchase_date.

    `cost_basis` and `purchase_date` are optional per row (NaN/None allowed)
    and may be missing as columns entirely. `ticker` and `shares` are
    required for every row.
    """
    required = {"ticker", "shares"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    holdings = []
    for i, row in df.iterrows():
        if pd.isna(row["ticker"]) or not str(row["ticker"]).strip():
            raise ValueError(f"Row {i}: ticker is required")
        ticker = str(row["ticker"]).strip().upper()

        try:
            shares = float(row["shares"])
        except (TypeError, ValueError):
            raise ValueError(f"Row {i}: shares must be a number") from None

        cost_basis = None
        if "cost_basis" in df.columns and pd.notna(row["cost_basis"]):
            cost_basis = float(row["cost_basis"])

        purchase_date = None
        if "purchase_date" in df.columns and pd.notna(row["purchase_date"]):
            purchase_date = pd.to_datetime(row["purchase_date"]).date()

        holdings.append(Holding(ticker=ticker, shares=shares, cost_basis=cost_basis, purchase_date=purchase_date))

    return holdings


def holdings_to_dataframe(holdings: list[Holding]) -> pd.DataFrame:
    """Inverse of holdings_from_dataframe, for seeding an editable table."""
    return pd.DataFrame(
        [
            {
                "ticker": h.ticker,
                "shares": h.shares,
                "cost_basis": h.cost_basis,
                "purchase_date": h.purchase_date,
            }
            for h in holdings
        ]
    )


def parse_holdings_csv(file) -> list[Holding]:
    """Parse an uploaded CSV with columns ticker, shares, cost_basis, purchase_date."""
    return holdings_from_dataframe(pd.read_csv(file))


def holdings_to_weights(holdings: list[Holding], current_prices: dict[str, float]) -> dict[str, float]:
    """Convert dollar holdings into a normalized weight dict using current prices.

    Holdings whose ticker is missing from `current_prices` are skipped.
    """
    values = {h.ticker: h.shares * current_prices[h.ticker] for h in holdings if h.ticker in current_prices}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("No holdings have usable current prices")
    return {t: v / total for t, v in values.items()}


def holdings_total_value(holdings: list[Holding], current_prices: dict[str, float]) -> float:
    """Total current dollar value of all holdings with a usable current price."""
    return sum(h.shares * current_prices[h.ticker] for h in holdings if h.ticker in current_prices)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_holdings.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add analysis/holdings.py tests/test_holdings.py
git commit -m "Add Holding model, CSV/dataframe parsing, and dollar-to-weight conversion"
```

---

### Task 2: Gain/loss and performance-over-time

**Files:**
- Modify: `analysis/holdings.py` (append)
- Test: `tests/test_holdings.py` (append)

**Interfaces:**
- Consumes: `Holding` from Task 1; `get_price_history(tickers: list[str], start: str, end: str) -> pd.DataFrame` from `analysis/data.py` (already exists, unchanged).
- Produces: `compute_gain_loss(holdings: list[Holding], current_prices: dict[str, float]) -> pd.DataFrame` (columns: `ticker, shares, cost_basis, current_value, dollar_gain, pct_gain`), `compute_value_history(holdings: list[Holding], today: date | None = None) -> pd.Series` (date-indexed).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_holdings.py`:

```python
from unittest.mock import patch

from analysis.holdings import compute_gain_loss, compute_value_history


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_holdings.py -v -k "gain_loss or value_history"`
Expected: FAIL with `ImportError: cannot import name 'compute_gain_loss'`

- [ ] **Step 3: Write the implementation**

Append to `analysis/holdings.py` (add `from analysis.data import get_price_history` to the imports at the top first):

```python
def compute_gain_loss(holdings: list[Holding], current_prices: dict[str, float]) -> pd.DataFrame:
    """Per-holding cost basis, current value, and $/% gain-loss.

    Holdings without `cost_basis` get NaN gain/loss columns. Holdings
    missing from `current_prices` get NaN current value too.
    """
    rows = []
    for h in holdings:
        current_value = h.shares * current_prices[h.ticker] if h.ticker in current_prices else float("nan")
        if h.cost_basis is not None:
            dollar_gain = current_value - h.cost_basis
            pct_gain = dollar_gain / h.cost_basis if h.cost_basis != 0 else float("nan")
        else:
            dollar_gain = float("nan")
            pct_gain = float("nan")
        rows.append(
            {
                "ticker": h.ticker,
                "shares": h.shares,
                "cost_basis": h.cost_basis if h.cost_basis is not None else float("nan"),
                "current_value": current_value,
                "dollar_gain": dollar_gain,
                "pct_gain": pct_gain,
            }
        )
    return pd.DataFrame(rows)


def compute_value_history(holdings: list[Holding], today: date | None = None) -> pd.Series:
    """Combined daily dollar value of holdings that have a purchase_date.

    Each holding contributes shares * price from its own purchase_date
    onward, so the total starts low and grows as positions "come online".
    Holdings without a purchase_date are skipped (can't be placed on a
    timeline). Returns an empty Series if no holding has a purchase_date.
    """
    dated = [h for h in holdings if h.purchase_date is not None]
    if not dated:
        return pd.Series(dtype=float)

    end = (today or date.today()).isoformat()
    series_list = []
    for h in dated:
        prices = get_price_history([h.ticker], h.purchase_date.isoformat(), end)
        if h.ticker not in prices.columns:
            continue
        series_list.append(prices[h.ticker] * h.shares)

    if not series_list:
        return pd.Series(dtype=float)

    combined = pd.concat(series_list, axis=1).sum(axis=1, min_count=1)
    return combined.dropna()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_holdings.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add analysis/holdings.py tests/test_holdings.py
git commit -m "Add gain/loss and performance-over-time calculations to holdings"
```

---

### Task 3: 5-year projection

**Files:**
- Modify: `analysis/portfolio.py` (append)
- Test: `tests/test_portfolio.py` (append)

**Interfaces:**
- Produces: `project_value(total_value: float, annualized_return: float, years: float = 5) -> float`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_portfolio.py`:

```python
from analysis.portfolio import project_value


def test_project_value_compounds_correctly():
    result = project_value(total_value=1000.0, annualized_return=0.10, years=5)
    assert result == pytest.approx(1000.0 * 1.10**5)


def test_project_value_zero_return_stays_flat():
    result = project_value(total_value=1000.0, annualized_return=0.0, years=5)
    assert result == pytest.approx(1000.0)


def test_project_value_default_years_is_five():
    result = project_value(total_value=1000.0, annualized_return=0.10)
    assert result == pytest.approx(1000.0 * 1.10**5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_portfolio.py -v -k project_value`
Expected: FAIL with `ImportError: cannot import name 'project_value'`

- [ ] **Step 3: Write the implementation**

Append to `analysis/portfolio.py`:

```python
def project_value(total_value: float, annualized_return: float, years: float = 5) -> float:
    """Compound `total_value` forward `years` at `annualized_return` (deterministic point estimate)."""
    return total_value * (1 + annualized_return) ** years
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_portfolio.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add analysis/portfolio.py tests/test_portfolio.py
git commit -m "Add project_value for deterministic 5-year forward projection"
```

---

### Task 4: Trade simulation

**Files:**
- Modify: `analysis/screener.py`
- Test: `tests/test_screener.py` (append)

**Interfaces:**
- Consumes: `_shrink_and_add` (existing, same file), `portfolio_stats` from `analysis/portfolio.py`.
- Produces: `simulate_trade(current_weights: dict[str, float], candidate: str, amount: float, total_value: float, funding: str, returns: pd.DataFrame, source_ticker: str | None = None, risk_free_rate: float = 0.02) -> dict` returning `{"before": dict, "after": dict, "new_weights": dict[str, float], "new_total_value": float, "capped": bool, "capped_amount": float}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_screener.py`:

```python
from analysis.screener import simulate_trade


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_screener.py -v -k simulate_trade`
Expected: FAIL with `ImportError: cannot import name 'simulate_trade'`

- [ ] **Step 3: Write the implementation**

In `analysis/screener.py`, change the import line:

```python
from analysis.portfolio import annualized_return, annualized_volatility, portfolio_stats, sharpe_ratio
```

Then append:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_screener.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add analysis/screener.py tests/test_screener.py
git commit -m "Add simulate_trade: new money, auto pro-rata, and specific-holding funding"
```

---

### Task 5: Streamlit UI — dollar holdings, gain/loss, performance chart, trade simulation

**Files:**
- Modify: `app/streamlit_app.py` (replace entire contents)

**Interfaces:**
- Consumes: everything produced in Tasks 1–4 (`holdings_from_dataframe`, `holdings_to_dataframe`, `parse_holdings_csv`, `holdings_to_weights`, `holdings_total_value`, `compute_gain_loss`, `compute_value_history`, `project_value`, `simulate_trade`), plus existing `filter_liquid`, `get_daily_returns`, `get_price_history`, `sp500_tickers`, `portfolio_stats`, `rank_candidates`.

- [ ] **Step 1: Replace `app/streamlit_app.py`**

```python
"""Thin Streamlit UI over analysis/. No analysis logic lives here."""

import sys
from pathlib import Path

# Streamlit puts this script's own directory (app/) on sys.path, not the
# repo root, so the analysis/ package next door isn't importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis.data import filter_liquid, get_daily_returns, get_price_history, sp500_tickers
from analysis.holdings import (
    compute_gain_loss,
    compute_value_history,
    holdings_from_dataframe,
    holdings_to_dataframe,
    holdings_to_weights,
    holdings_total_value,
    parse_holdings_csv,
)
from analysis.portfolio import portfolio_stats, project_value
from analysis.screener import rank_candidates, simulate_trade

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("Portfolio Analyzer")

DEFAULT_HOLDINGS_DF = pd.DataFrame(
    [
        {"ticker": "AAPL", "shares": 10.0, "cost_basis": 1200.0, "purchase_date": pd.Timestamp("2024-01-15")},
        {"ticker": "MSFT", "shares": 5.0, "cost_basis": 1550.0, "purchase_date": pd.Timestamp("2024-01-15")},
        {"ticker": "SPY", "shares": 8.0, "cost_basis": 3200.0, "purchase_date": pd.Timestamp("2024-01-15")},
    ]
)

st.sidebar.header("Current Holdings")

if "_last_upload_name" not in st.session_state:
    st.session_state._last_upload_name = "default"
    st.session_state.holdings_df = DEFAULT_HOLDINGS_DF.copy()

uploaded = st.sidebar.file_uploader("Upload holdings CSV (ticker, shares, cost_basis, purchase_date)", type="csv")
if uploaded is not None and uploaded.name != st.session_state._last_upload_name:
    try:
        st.session_state.holdings_df = holdings_to_dataframe(parse_holdings_csv(uploaded))
        st.session_state._last_upload_name = uploaded.name
    except ValueError as e:
        st.sidebar.error(f"Could not parse CSV: {e}")

edited_df = st.sidebar.data_editor(
    st.session_state.holdings_df,
    num_rows="dynamic",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", required=True),
        "shares": st.column_config.NumberColumn("Shares", required=True, min_value=0.0),
        "cost_basis": st.column_config.NumberColumn("Cost Basis ($)", min_value=0.0),
        "purchase_date": st.column_config.DateColumn("Purchase Date"),
    },
    key=f"holdings_editor_{st.session_state._last_upload_name}",
)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start date", value=pd.Timestamp.today() - pd.DateOffset(years=1))
end_date = col2.date_input("End date", value=pd.Timestamp.today())

risk_free_rate = st.sidebar.number_input("Risk-free rate", value=0.02, step=0.005, format="%.3f")
epsilon = st.sidebar.slider("Candidate weight (epsilon) for screening", 0.01, 0.20, 0.05, 0.01)
universe_choice = st.sidebar.selectbox("Candidate universe", ["Full S&P 500", "First 100 (faster)"])
top_n = st.sidebar.slider("Show top N recommendations", 5, 50, 15)

run = st.sidebar.button("Analyze")


@st.cache_data(ttl=3600)
def load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    return get_price_history(list(tickers), start, end)


@st.cache_data(ttl=86400)
def load_sp500() -> list:
    return sp500_tickers()


def render_correlation_heatmap(corr: pd.DataFrame) -> None:
    size = max(4, 0.5 * len(corr.columns))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)


def render_value_history_chart(value_history: pd.Series, cost_basis_total: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(value_history.index, value_history.values, label="Portfolio value")
    if cost_basis_total > 0:
        ax.axhline(cost_basis_total, color="gray", linestyle="--", label="Total cost basis")
    ax.set_ylabel("Value ($)")
    ax.legend()
    st.pyplot(fig)


if not run:
    st.info("Enter your holdings in the sidebar and click **Analyze** to get started.")
    st.stop()

try:
    holdings = holdings_from_dataframe(edited_df)
except ValueError as e:
    st.error(f"Could not parse holdings: {e}")
    st.stop()

if not holdings:
    st.error("Add at least one holding.")
    st.stop()

with st.spinner("Fetching candidate universe..."):
    universe = load_sp500()
    if universe_choice == "First 100 (faster)":
        universe = universe[:100]

holding_tickers = [h.ticker for h in holdings]
all_tickers = sorted(set(holding_tickers) | set(universe))

with st.spinner(f"Downloading price history for {len(all_tickers)} tickers..."):
    prices = load_prices(tuple(all_tickers), str(start_date), str(end_date))

prices = filter_liquid(prices)

held_missing = [t for t in holding_tickers if t not in prices.columns]
if held_missing:
    st.warning(f"No sufficient price data for held tickers: {held_missing}. Excluding from analysis.")
    holdings = [h for h in holdings if h.ticker not in held_missing]

if not holdings:
    st.error("No held tickers have usable data.")
    st.stop()

current_prices = prices.iloc[-1].to_dict()

weights = holdings_to_weights(holdings, current_prices)
total_value = holdings_total_value(holdings, current_prices)

returns = get_daily_returns(prices)
portfolio_returns = returns[list(weights)]
stats = portfolio_stats(portfolio_returns, weights, risk_free_rate)

st.subheader("Current Portfolio Stats")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Value", f"${total_value:,.0f}")
m2.metric("Annualized Return", f"{stats['annualized_return']:.2%}")
m3.metric("Annualized Volatility", f"{stats['annualized_volatility']:.2%}")
m4.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")

st.subheader("Gain / Loss")
gain_loss = compute_gain_loss(holdings, current_prices)
totals_row = pd.DataFrame(
    [
        {
            "ticker": "TOTAL",
            "shares": float("nan"),
            "cost_basis": gain_loss["cost_basis"].sum(),
            "current_value": gain_loss["current_value"].sum(),
            "dollar_gain": gain_loss["dollar_gain"].sum(),
            "pct_gain": (gain_loss["dollar_gain"].sum() / gain_loss["cost_basis"].sum())
            if gain_loss["cost_basis"].sum()
            else float("nan"),
        }
    ]
)
gain_loss_display = pd.concat([gain_loss, totals_row], ignore_index=True)
st.dataframe(
    gain_loss_display.style.format(
        {"cost_basis": "${:,.0f}", "current_value": "${:,.0f}", "dollar_gain": "${:,.0f}", "pct_gain": "{:.1%}"},
        na_rep="—",
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("Performance Over Time")
value_history = compute_value_history(holdings)
if value_history.empty:
    st.info("Add a purchase date to at least one holding to see a performance chart.")
else:
    cost_basis_total = sum(h.cost_basis for h in holdings if h.purchase_date is not None and h.cost_basis is not None)
    render_value_history_chart(value_history, cost_basis_total)

st.subheader("Correlation Matrix")
render_correlation_heatmap(stats["correlation_matrix"])

st.subheader(f"Top {top_n} Recommendations (Marginal Sharpe Contribution)")
candidates = [t for t in universe if t in returns.columns and t not in weights]
with st.spinner("Screening candidates..."):
    ranked = rank_candidates(returns, weights, candidates, epsilon=epsilon, risk_free_rate=risk_free_rate)

if ranked.empty:
    st.info("No candidates could be scored.")
else:
    display = ranked.head(top_n).copy()
    display["delta_sharpe"] = display["delta_sharpe"].map("{:+.3f}".format)
    display["new_sharpe"] = display["new_sharpe"].map("{:.3f}".format)
    display["delta_volatility"] = display["delta_volatility"].map("{:+.2%}".format)
    display["new_volatility"] = display["new_volatility"].map("{:.2%}".format)
    display["new_return"] = display["new_return"].map("{:.2%}".format)
    st.dataframe(display, width="stretch", hide_index=True)

    st.subheader("Simulate a Trade")
    candidate_choice = st.selectbox("Candidate", options=ranked["ticker"].tolist())
    funding_label = st.radio(
        "Fund this by",
        ["Invest new money", "Move money, auto (pro-rata)", "Move money from a specific holding"],
    )
    source_ticker = None
    if funding_label == "Move money from a specific holding":
        source_ticker = st.selectbox("Sell from", list(weights.keys()))
    amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0, step=100.0)
    simulate = st.button("Preview Trade")

    if simulate:
        funding_key = {
            "Invest new money": "new_money",
            "Move money, auto (pro-rata)": "auto_prorata",
            "Move money from a specific holding": "specific_holding",
        }[funding_label]

        result = simulate_trade(
            weights,
            candidate_choice,
            amount,
            total_value,
            funding_key,
            returns,
            source_ticker=source_ticker,
            risk_free_rate=risk_free_rate,
        )

        if result["capped"]:
            st.warning(
                f"Requested ${amount:,.0f} exceeds {source_ticker}'s current value; "
                f"capped at ${result['capped_amount']:,.0f}."
            )

        before, after = result["before"], result["after"]
        new_total_value = result["new_total_value"]

        st.write("**Before vs. After**")
        comparison = pd.DataFrame(
            {
                "Current portfolio": [
                    f"${total_value:,.0f}",
                    f"{before['annualized_return']:.2%}",
                    f"{before['annualized_volatility']:.2%}",
                    f"{before['sharpe_ratio']:.2f}",
                ],
                "What-if portfolio": [
                    f"${new_total_value:,.0f}",
                    f"{after['annualized_return']:.2%}",
                    f"{after['annualized_volatility']:.2%}",
                    f"{after['sharpe_ratio']:.2f}",
                ],
            },
            index=["Total value", "Annualized return", "Annualized volatility", "Sharpe ratio"],
        )
        st.table(comparison)

        st.write("**5-Year Projection** (compounding today's annualized return forward, point estimate)")
        current_future = project_value(total_value, before["annualized_return"])
        whatif_future = project_value(new_total_value, after["annualized_return"])
        projection = pd.DataFrame(
            {
                "Current portfolio": [f"${total_value:,.0f}", f"${current_future:,.0f}"],
                "What-if portfolio": [f"${new_total_value:,.0f}", f"${whatif_future:,.0f}"],
            },
            index=["Total today", "In 5 years"],
        )
        st.table(projection)

        st.write("**Per-Holding Breakdown**")
        all_holding_tickers = sorted(set(weights) | set(result["new_weights"]))
        breakdown = pd.DataFrame(
            {
                "ticker": all_holding_tickers,
                "before_$": [weights.get(t, 0.0) * total_value for t in all_holding_tickers],
                "after_$": [result["new_weights"].get(t, 0.0) * new_total_value for t in all_holding_tickers],
            }
        )
        st.dataframe(
            breakdown.style.format({"before_$": "${:,.0f}", "after_$": "${:,.0f}"}),
            width="stretch",
            hide_index=True,
        )
```

- [ ] **Step 2: Verify the app runs end-to-end with no exceptions**

Create a throwaway script at the repo root (do not commit it):

```python
# _verify_ui.py
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/streamlit_app.py", default_timeout=600)
at.run()
at.sidebar.button[0].click().run()

if at.exception:
    for exc in at.exception:
        print("EXCEPTION after Analyze:", exc)
    raise SystemExit(1)

if at.button:
    at.button[-1].click().run()

if at.exception:
    for exc in at.exception:
        print("EXCEPTION after Preview Trade:", exc)
    raise SystemExit(1)

print("UI ran end-to-end with no exceptions.")
```

Run: `venv/bin/python _verify_ui.py`
Expected: `UI ran end-to-end with no exceptions.` (this will take a while — it downloads the full S&P 500 price history). If it fails, read the exception, fix `app/streamlit_app.py`, and re-run — do not proceed to Step 3 until this passes.

Then delete the script: `rm _verify_ui.py`

- [ ] **Step 3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "Rewrite UI for dollar-based holdings, gain/loss, performance chart, and trade simulation"
```

---

### Task 6: README documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Add holdings import and trade simulation sections**

In `README.md`, insert the following new sections immediately before `## Project layout`:

```markdown
## Holdings import

Holdings are dollar-based: upload a CSV or edit the table directly in the sidebar. Columns:

| column | required | meaning |
|---|---|---|
| `ticker` | yes | stock ticker |
| `shares` | yes | number of shares held |
| `cost_basis` | no | total dollars paid for the position (not per-share) — omit to skip gain/loss for that holding |
| `purchase_date` | no | when the position was opened — omit to skip that holding from the performance-over-time chart |

## Trade simulation

Below the recommendations table, pick a candidate and a dollar amount, then choose how to fund it:

- **Invest new money** — adds the amount on top; existing holdings dilute proportionally.
- **Move money, auto (pro-rata)** — trims every current holding proportionally to raise the amount (the same math the screener uses to rank candidates, parameterized by a real dollar figure instead of a fixed 5%).
- **Move money from a specific holding** — trims only that one position, capped at its current value.

The preview shows before/after portfolio stats, a per-holding dollar breakdown, and a 5-year projection: today's annualized return compounded forward as a deterministic point estimate (not a confidence interval — see Limitations).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document holdings CSV format and trade simulation funding modes"
```
