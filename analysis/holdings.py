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
