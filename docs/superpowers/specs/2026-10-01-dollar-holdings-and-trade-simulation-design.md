# Dollar-based holdings, performance history, and trade simulation

## Problem

The app currently takes a portfolio as typed-in `ticker:weight` pairs with no
connection to real dollar amounts, cost basis, or purchase history. There's
no way to see how a portfolio has actually performed since it was bought, and
the recommendations table only ranks candidates in the abstract — it doesn't
let you act on one with a real dollar amount.

There is no public API for pulling a user's actual brokerage holdings from
Yahoo Finance or elsewhere without an authenticated account connection, so
"import" means CSV upload (the common brokerage-export shape) plus manual
entry/editing, not a live account link.

## Scope

1. Dollar-based holdings data model, with CSV import and manual editing.
2. Portfolio display: dollar stats, gain/loss table, performance-over-time chart.
3. Trade simulation: pick a recommended candidate, choose how to fund it,
   preview the before/after impact, including a 5-year forward projection.

Out of scope: persisting/saving a modified portfolio, live account
connections, Monte Carlo or any probabilistic projection (deterministic
compounding only, matching the rest of the app's historical-return-as-proxy
approach).

## 1. Holdings data model & import

New module `analysis/holdings.py`:

- `Holding`: `ticker: str`, `shares: float`, `cost_basis: float | None` (total
  dollars paid, not per-share), `purchase_date: date | None`.
- `parse_holdings_csv(file) -> list[Holding]`: reads a CSV with columns
  `ticker, shares, cost_basis, purchase_date`. `ticker` and `shares` are
  required; `cost_basis` and `purchase_date` are optional per row (NaN/blank
  allowed) — a holding missing them just won't get a gain/loss figure or a
  "since purchase" chart line, it isn't rejected.
- `holdings_to_weights(holdings, current_prices: dict[str, float]) -> dict[str, float]`:
  computes `value_i = shares_i * current_prices[ticker_i]`, normalizes by
  total value. This is the bridge into the existing `portfolio.py` and
  `screener.py`, which keep operating on weight dicts unchanged.

UI (`app/streamlit_app.py`): replaces the `ticker:weight` text box with:

- `st.file_uploader` for the CSV.
- `st.data_editor` pre-populated from the parsed CSV (or empty if none
  uploaded), showing `ticker, shares, cost_basis, purchase_date` as editable
  rows, so holdings can be tweaked or added without re-uploading.

## 2. Portfolio display

- **Current stats** (annualized return/volatility/Sharpe/correlation): same
  functions as today, fed by `holdings_to_weights()` instead of typed
  weights.
- **Gain/loss table**: one row per holding with `cost_basis`, current value
  (`shares × latest price`), $ gain/loss, % gain/loss, plus a totals row.
  Holdings without `cost_basis` are shown with gain/loss columns blank, not
  excluded from the table.
- **Performance-over-time chart**: for each holding with a `purchase_date`,
  fetch its own price history from that date to today
  (`get_price_history([ticker], purchase_date, today)`), multiply by shares
  to get a daily dollar value series, and sum across holdings day-by-day — a
  holding only contributes from its own purchase date onward, so the total
  line starts low and grows as positions "come online". A flat dashed
  reference line shows total cost basis. Holdings without a `purchase_date`
  are excluded from this chart only (they still count in the current
  stats/gain-loss table, using the existing global Start/End date inputs for
  the return/vol/Sharpe numbers).

## 3. Trade simulation

New function in `analysis/screener.py` (or a new `analysis/trade.py` if
`screener.py` gets crowded — implementer's call):

- `simulate_trade(holdings, candidate, amount, funding, returns) -> dict`
  with `funding` one of:
  - `"new_money"`: total portfolio value grows by `amount`; every existing
    holding's weight dilutes proportionally
    (`new_weight_i = old_weight_i * old_total / (old_total + amount)`);
    candidate's weight = `amount / (old_total + amount)`.
  - `"auto_prorata"`: reuses the existing `_shrink_and_add` logic from the
    screener, with `epsilon = amount / old_total` instead of the fixed
    screening default — every holding shrinks proportionally to fund the
    candidate.
  - `"specific_holding"` (+ `source_ticker`): `source_ticker`'s weight is
    reduced by `amount / old_total`, floored at 0 — if `amount` exceeds that
    holding's current value, cap at its full value and surface a warning
    rather than going negative; candidate's weight increases by the same
    (capped) amount.
- Returns before/after `portfolio_stats()` for both the current and
  resulting weight sets, plus a per-holding old-$ / new-$ breakdown.

UI: below the ranked recommendations table —

1. Candidate dropdown (defaults to the top-ranked row, any ticker from the
   table selectable).
2. Funding mode: `new_money` / `auto_prorata` / `specific_holding` (with a
   second dropdown for the source ticker when applicable).
3. Dollar amount (`st.number_input`).
4. On submit: before/after stats table, per-holding $ breakdown, and the
   5-year projection (below). Pure what-if — never writes back to the
   holdings table/CSV; changing inputs just re-renders.

## 4. 5-year forward projection

Attached to the trade simulation preview (not a separate standalone feature):
for both the current portfolio and the what-if portfolio, compound today's
total dollar value forward 5 years using each one's own annualized return
(already computed for the stats row):

```
future_value = total_value * (1 + annualized_return) ** 5
```

Shown side by side with the before/after stats:

```
                Current portfolio    What-if portfolio
Total today         $10,000               $10,000
Annualized return     12.9%                 16.4%
In 5 years          $18,400               $21,200
```

Deterministic point estimate only — no confidence interval or simulation —
consistent with the rest of the app's historical-return-as-proxy approach
(already flagged as a limitation in the README).

## Testing

- `analysis/holdings.py`: CSV parsing (required vs. optional columns,
  malformed rows), `holdings_to_weights` normalization.
- `simulate_trade`: each funding mode's weight math, the `specific_holding`
  cap-and-warn path when `amount` exceeds the source holding's value, and
  the 5-year compounding formula.
- No new UI tests beyond the existing manual Streamlit verification pattern
  used for the rest of the app.

## README

Add a short section documenting the CSV format and the three funding modes,
alongside the existing methodology writeup.
