"""Thin Streamlit UI over analysis/. No analysis logic lives here."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis.data import filter_liquid, get_daily_returns, get_price_history, sp500_tickers
from analysis.portfolio import portfolio_stats
from analysis.screener import rank_candidates

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("Portfolio Analyzer")

st.sidebar.header("Current Holdings")
holdings_input = st.sidebar.text_area(
    "Ticker:weight pairs (comma-separated)",
    value="AAPL:0.4, MSFT:0.4, SPY:0.2",
    height=80,
)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start date", value=pd.Timestamp.today() - pd.DateOffset(years=1))
end_date = col2.date_input("End date", value=pd.Timestamp.today())

risk_free_rate = st.sidebar.number_input("Risk-free rate", value=0.02, step=0.005, format="%.3f")
epsilon = st.sidebar.slider("Candidate weight (epsilon) for screening", 0.01, 0.20, 0.05, 0.01)
universe_choice = st.sidebar.selectbox("Candidate universe", ["Full S&P 500", "First 100 (faster)"])
top_n = st.sidebar.slider("Show top N recommendations", 5, 50, 15)

run = st.sidebar.button("Analyze")


def parse_holdings(text: str) -> dict[str, float]:
    weights = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        ticker, weight = pair.split(":")
        weights[ticker.strip().upper()] = float(weight)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return {t: w / total for t, w in weights.items()}


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


if not run:
    st.info("Enter your holdings in the sidebar and click **Analyze** to get started.")
    st.stop()

try:
    weights = parse_holdings(holdings_input)
except ValueError as e:
    st.error(f"Could not parse holdings: {e}")
    st.stop()

with st.spinner("Fetching candidate universe..."):
    universe = load_sp500()
    if universe_choice == "First 100 (faster)":
        universe = universe[:100]

all_tickers = sorted(set(weights) | set(universe))

with st.spinner(f"Downloading price history for {len(all_tickers)} tickers..."):
    prices = load_prices(tuple(all_tickers), str(start_date), str(end_date))

prices = filter_liquid(prices)

held_missing = [t for t in weights if t not in prices.columns]
if held_missing:
    st.warning(f"No sufficient price data for held tickers: {held_missing}. Excluding from analysis.")
    weights = {t: w for t, w in weights.items() if t not in held_missing}

if not weights:
    st.error("No held tickers have usable data.")
    st.stop()

returns = get_daily_returns(prices)
portfolio_returns = returns[list(weights)]
stats = portfolio_stats(portfolio_returns, weights, risk_free_rate)

st.subheader("Current Portfolio Stats")
m1, m2, m3 = st.columns(3)
m1.metric("Annualized Return", f"{stats['annualized_return']:.2%}")
m2.metric("Annualized Volatility", f"{stats['annualized_volatility']:.2%}")
m3.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")

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
    st.dataframe(display, use_container_width=True, hide_index=True)
