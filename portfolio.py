"""
portfolio.py

Simulates a portfolio for each firm using only their confirmed order fills.
Rules:
  - Portfolio starts at 0 shares and $0 PnL on day 1
  - Bid fill  → buy  shares: position += size,  cash -= size * price
  - Ask fill  → sell shares: position -= size,  cash += size * price
  - Position can never go negative — sells skipped if shares not held
  - Portfolio value at any moment = position * current_mid_price
  - Total PnL = cash + unrealised_value  (realised + unrealised combined)

Only COST data is available now; the script is structured to extend to
more tickers by dropping additional firm_*_orders.csv files into output/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = Path(r"C:\Users\komal\Project285J\output")
DATA_DIR   = Path(r"C:\Users\komal\Downloads\_data_dwn_67_491__COST_2025-10-01_2025-10-14_0")
PRICE_SCALE = 0.0001
TICKER      = "COST"

# ── 1. Load mid-price series (1-minute sampled) for portfolio valuation ────────
print("Building 1-minute mid-price series from message files...")
mid_series_parts = []

for fpath in sorted(DATA_DIR.glob(f"{TICKER}_*_message_0.csv")):
    date_str = fpath.name.split("_")[1]
    df = pd.read_csv(fpath, header=None,
                     names=["time","type","oid","size","price","dir","firm"],
                     low_memory=False)
    execs = df[df["type"].isin([4, 5])].copy()
    if execs.empty:
        continue
    execs["mid_usd"] = execs["price"].astype(float) * PRICE_SCALE
    execs["minute"]  = (execs["time"] // 60).astype(int)
    minute_mid = execs.groupby("minute")["mid_usd"].last().reset_index()
    minute_mid["date"] = date_str
    minute_mid["t_sec"] = minute_mid["minute"] * 60
    mid_series_parts.append(minute_mid)

mid_df = pd.concat(mid_series_parts, ignore_index=True)
mid_df = mid_df.sort_values(["date","t_sec"]).reset_index(drop=True)
# forward-fill any gaps
mid_df["mid_usd"] = mid_df["mid_usd"].ffill()
print(f"  {len(mid_df):,} 1-minute price points across {mid_df.date.nunique()} days")

# helper: get mid-price at or just before a given (date, timestamp)
def get_mid(date: str, t: float) -> float:
    sub = mid_df[(mid_df.date == date) & (mid_df.t_sec <= t)]
    return float(sub.iloc[-1]["mid_usd"]) if len(sub) else np.nan


# ── 2. Load fills from all firm CSVs ──────────────────────────────────────────
print("\nLoading fills from per-firm CSVs...")
firm_csvs = sorted(OUTPUT_DIR.glob("firm_*_orders.csv"))
all_fills = []

for csv_path in firm_csvs:
    firm_name = csv_path.stem.replace("firm_", "").replace("_orders", "")
    df = pd.read_csv(csv_path, low_memory=False)
    fills = df[df["filled"] == True].copy()
    fills["firm"] = firm_name
    fills["ticker"] = TICKER       # tag with ticker; extend here for multi-ticker
    all_fills.append(fills)
    print(f"  {firm_name:6s}: {len(fills):>4} fills")

fills_df = pd.concat(all_fills, ignore_index=True) if all_fills else pd.DataFrame()
fills_df = fills_df.sort_values(["date","t_submit"]).reset_index(drop=True)

firms = sorted(fills_df["firm"].unique()) if len(fills_df) else []
print(f"\n  {len(fills_df)} total fills across {len(firms)} firms with at least 1 fill")

# ── 3. Simulate portfolio per firm ────────────────────────────────────────────
print("\nSimulating portfolios...")

portfolio_records = {}   # firm -> DataFrame of (date, t_sec, position, cash, pnl, mid)

for firm in firms:
    firm_fills = fills_df[fills_df.firm == firm].sort_values(["date","t_submit"])
    position = 0       # shares held
    cash     = 0.0     # realised cash flow (positive = received from sells)
    records  = []

    for row in firm_fills.itertuples(index=False):
        size  = int(row.size)
        price = float(row.sub_price)
        date  = row.date
        t     = float(row.t_submit)
        mid   = get_mid(date, t)

        if row.direction == "bid":          # buy
            position += size
            cash     -= size * price
        elif row.direction == "ask":        # sell
            if position >= size:            # only sell what we hold
                position -= size
                cash     += size * price
            else:
                continue                   # skip — can't go negative

        unrealised = position * mid if not np.isnan(mid) else 0.0
        total_pnl  = cash + unrealised

        records.append({
            "date":        date,
            "t_submit":    t,
            "direction":   row.direction,
            "size":        size,
            "price":       price,
            "mid_at_fill": mid,
            "position":    position,
            "cash":        round(cash, 2),
            "unrealised":  round(unrealised, 2),
            "total_pnl":   round(total_pnl, 2),
        })

    portfolio_records[firm] = pd.DataFrame(records)
    df_p = portfolio_records[firm]
    final_pnl = df_p["total_pnl"].iloc[-1] if len(df_p) else 0
    print(f"  {firm:6s}: {len(df_p):>3} trades | final position={df_p['position'].iloc[-1] if len(df_p) else 0} shares | total PnL=${final_pnl:,.2f}")

# ── 4. Build full time-series of portfolio value (using 1-min mid-price) ──────
# Between fills, position doesn't change, so we project value onto the mid grid
print("\nProjecting portfolio value onto minute grid...")

ts_records = {}  # firm -> DataFrame(date, t_sec, portfolio_value, total_pnl)

for firm, df_p in portfolio_records.items():
    if df_p.empty:
        continue

    rows = []
    for _, date_mid in mid_df.groupby("date"):
        date = date_mid.iloc[0]["date"]
        # find position and cash at end of this date (from fills up to this date)
        prior = df_p[df_p.date <= date]
        if prior.empty:
            pos, cash_so_far = 0, 0.0
        else:
            last = prior.iloc[-1]
            pos, cash_so_far = int(last["position"]), float(last["cash"])

        for row in date_mid.itertuples(index=False):
            val = pos * row.mid_usd
            rows.append({"date": row.date, "t_sec": row.t_sec,
                         "position": pos, "portfolio_value": round(val, 2),
                         "total_pnl": round(cash_so_far + val, 2)})

    ts_records[firm] = pd.DataFrame(rows)

# ── 5. Charts ─────────────────────────────────────────────────────────────────
print("\nGenerating charts...")

COLOR_MAP = {
    "WBPX":"#1f77b4","MAXM":"#ff7f0e","WCHV":"#2ca02c",
    "JPMS":"#d62728","UBSS":"#9467bd","ETMM":"#8c564b","IMCC":"#e377c2"
}

# ── Chart A: Portfolio value over time ─────────────────────────────────────────
fig_val = go.Figure()

for firm, ts_df in ts_records.items():
    if ts_df.empty:
        continue
    label = f"{firm} ({mid_df[mid_df.date.isin(ts_df.date.unique())].shape[0]} pts)"
    ts_df["datetime_approx"] = ts_df["date"] + " " + ts_df["t_sec"].apply(
        lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}")
    fig_val.add_trace(go.Scatter(
        x=ts_df["datetime_approx"],
        y=ts_df["portfolio_value"],
        mode="lines",
        name=firm,
        line=dict(color=COLOR_MAP.get(firm)),
    ))

fig_val.update_layout(
    title=f"{TICKER} Portfolio Value Over Time — Per Firm",
    xaxis_title="Date / Time (ET)",
    yaxis_title="Portfolio Value (USD)",
    template="plotly_dark",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
fig_val.write_html(str(OUTPUT_DIR / "chart_portfolio_value.html"))
print("  Saved chart_portfolio_value.html")

# ── Chart B: Total PnL over time ───────────────────────────────────────────────
fig_pnl = go.Figure()

for firm, ts_df in ts_records.items():
    if ts_df.empty:
        continue
    ts_df["datetime_approx"] = ts_df["date"] + " " + ts_df["t_sec"].apply(
        lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}")
    fig_pnl.add_trace(go.Scatter(
        x=ts_df["datetime_approx"],
        y=ts_df["total_pnl"],
        mode="lines",
        name=firm,
        line=dict(color=COLOR_MAP.get(firm)),
    ))

fig_pnl.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
fig_pnl.update_layout(
    title=f"{TICKER} Cumulative PnL Over Time — Per Firm",
    xaxis_title="Date / Time (ET)",
    yaxis_title="Total PnL (USD)  [realised + unrealised]",
    template="plotly_dark",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
fig_pnl.write_html(str(OUTPUT_DIR / "chart_portfolio_pnl.html"))
print("  Saved chart_portfolio_pnl.html")

# ── Chart C: Per-firm position trajectory ─────────────────────────────────────
fig_pos = go.Figure()
for firm, ts_df in ts_records.items():
    if ts_df.empty:
        continue
    ts_df["datetime_approx"] = ts_df["date"] + " " + ts_df["t_sec"].apply(
        lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}")
    fig_pos.add_trace(go.Scatter(
        x=ts_df["datetime_approx"],
        y=ts_df["position"],
        mode="lines",
        name=firm,
        line=dict(color=COLOR_MAP.get(firm)),
        fill="tozeroy",
        fillcolor=COLOR_MAP.get(firm,"grey"),
        opacity=0.3,
    ))

fig_pos.update_layout(
    title=f"{TICKER} Share Position Over Time — Per Firm",
    xaxis_title="Date / Time (ET)",
    yaxis_title="Shares Held",
    template="plotly_dark",
    height=500,
)
fig_pos.write_html(str(OUTPUT_DIR / "chart_position.html"))
print("  Saved chart_position.html")

# ── Save portfolio trade log ───────────────────────────────────────────────────
all_trades = pd.concat(
    [df.assign(firm=f) for f, df in portfolio_records.items()],
    ignore_index=True
) if portfolio_records else pd.DataFrame()
all_trades.to_csv(OUTPUT_DIR / "portfolio_trades.csv", index=False)
print(f"  Saved portfolio_trades.csv ({len(all_trades)} trade events)")

print("\nDone.")
