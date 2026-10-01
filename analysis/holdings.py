"""Dollar-based holdings: parsing and dollar-to-weight conversion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from analysis.data import get_price_history


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
