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
