"""Historical price data pulls via yfinance, with on-disk caching."""

from __future__ import annotations

import hashlib
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _cache_key(tickers: list[str], start: str, end: str) -> Path:
    raw = f"{','.join(sorted(tickers))}|{start}|{end}"
    digest = hashlib.sha1(raw.encode()).hexdigest()
    return CACHE_DIR / f"{digest}.parquet"


def get_price_history(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted close prices for `tickers` between `start` and `end`.

    Returns a DataFrame indexed by date, one column per ticker. Tickers with
    no data in the requested range are dropped with a printed warning.
    """
    cache_path = _cache_key(tickers, start, end)
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if len(tickers) == 1:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw})

    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        print(f"Warning: no data returned for {missing}, dropping from result")
        prices = prices.drop(columns=missing, errors="ignore")

    prices = prices.dropna(how="all")

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        prices.to_parquet(cache_path)

    return prices


def get_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns from a price DataFrame."""
    return prices.pct_change().dropna(how="all")


def sp500_tickers() -> list[str]:
    """Current S&P 500 constituent tickers, scraped from Wikipedia.

    Tickers are normalized for yfinance (e.g. "BRK.B" -> "BRK-B").
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    symbols = tables[0]["Symbol"].tolist()
    return [s.replace(".", "-") for s in symbols]


def filter_liquid(prices: pd.DataFrame, min_coverage: float = 0.95) -> pd.DataFrame:
    """Drop columns with too much missing data over the window (illiquid/newly-listed proxy).

    `min_coverage` is the minimum fraction of non-NaN trading days required.
    """
    coverage = prices.notna().mean()
    keep = coverage[coverage >= min_coverage].index
    return prices[keep]
