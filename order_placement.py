"""
order_placement.py

Visualises where each firm's limit orders are placed relative to the
COST mid-price across Oct 1–14, 2025.

Charts produced (all saved as interactive HTML):
  1. price_hist_{FIRM}.html       — distribution of price-to-mid in bps, bid vs ask
  2. placement_scatter_{FIRM}.html — each order plotted: time vs bps from mid
  3. placement_heatmap_{FIRM}.html — heatmap: time-of-day × price-bucket × order count
  4. ladder_snapshot_{FIRM}.html  — single-day order ladder vs actual COST price range
  5. bid_ask_violin.html          — violin: bid-side vs ask-side placement per firm
  6. activity_by_hour.html        — order count per hour per firm
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

OUTPUT_DIR  = Path(r"C:\Users\komal\Project285J\output")
DATE_RANGE  = "Oct 1–14 2025"
TICKER_FULL = "COST (Costco Wholesale, NASDAQ)"
DATA_DIR    = Path(r"C:\Users\komal\Downloads\_data_dwn_67_491__COST_2025-10-01_2025-10-14_0")

# Best firm name per MPID (de-duped from firm_identities.json)
FIRM_NAMES = {
    "WBPX": "Wedbush Securities Inc.",
    "MAXM": "Maxim Group LLC",
    "WCHV": "Wells Fargo Securities LLC",
    "JPMS": "J.P. Morgan Securities LLC",
    "UBSS": "UBS Securities LLC",
    "ETMM": "G1 Execution Services LLC (fmr. E*TRADE Capital Markets)",
    "IMCC": "IM Cannabis Corp. [ticker anomaly — not a broker-dealer]",
}

COLOR_BID = "#00ccff"
COLOR_ASK = "#ff6666"

FOCUS_FIRMS = ["WBPX", "MAXM", "WCHV", "JPMS", "UBSS", "ETMM"]

def firm_label(mpid: str) -> str:
    return f"{mpid} — {FIRM_NAMES.get(mpid, 'Unknown')}"

def subtitle(mpid: str, extra: str = "") -> str:
    base = f"{TICKER_FULL} | {DATE_RANGE} | {FIRM_NAMES.get(mpid,'Unknown')} ({mpid})"
    return base + (f"<br><sup>{extra}</sup>" if extra else "")

# ── Load all firm order data ───────────────────────────────────────────────────
print("Loading per-firm order CSVs...")
firm_dfs: dict[str, pd.DataFrame] = {}
for mpid in FOCUS_FIRMS:
    path = OUTPUT_DIR / f"firm_{mpid}_orders.csv"
    if not path.exists():
        print(f"  {mpid}: file not found, skipping")
        continue
    df = pd.read_csv(path, low_memory=False)
    df["time_et_h"] = df["t_submit"].apply(lambda s: float(s) / 3600)   # hours since midnight
    df["hour_et"]   = df["t_submit"].apply(lambda s: int(float(s) // 3600))
    firm_dfs[mpid]  = df
    print(f"  {mpid}: {len(df):,} orders  fills={df.filled.sum()}")

# ── Build actual COST mid-price envelope (min/max/mean by minute) ──────────────
print("\nBuilding COST mid-price envelope from tape...")
tape_parts = []
for fpath in sorted(DATA_DIR.glob("COST_*_message_0.csv")):
    date_str = fpath.name.split("_")[1]
    raw = pd.read_csv(fpath, header=None,
                      names=["time","type","oid","size","price","dir","firm"],
                      low_memory=False)
    execs = raw[raw["type"].isin([4, 5])].copy()
    execs["mid"]    = execs["price"].astype(float) * 0.0001
    execs["minute"] = (execs["time"] // 60).astype(int)
    execs["date"]   = date_str
    tape_parts.append(execs[["date","minute","mid"]])

tape = pd.concat(tape_parts, ignore_index=True)
tape_env = tape.groupby("minute")["mid"].agg(["min","max","mean"]).reset_index()
tape_env["time_h"] = tape_env["minute"] / 60

print(f"  {len(tape_env):,} 1-minute price points  mid range "
      f"${tape_env['min'].min():.2f}–${tape_env['max'].max():.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Price-to-mid histogram, per firm, bid vs ask stacked
# ═══════════════════════════════════════════════════════════════════════════════
print("\nChart 1: Price-to-mid histograms...")
for mpid, df in firm_dfs.items():
    p2m = df["price_to_mid_bps"].dropna()
    bids = df[df.direction == "bid"]["price_to_mid_bps"].dropna()
    asks = df[df.direction == "ask"]["price_to_mid_bps"].dropna()

    # clip extreme outliers for readability
    lo, hi = p2m.quantile(0.01), p2m.quantile(0.99)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=bids.clip(lo, hi), name="Bid orders (buy side)",
        marker_color=COLOR_BID, opacity=0.7,
        xbins=dict(size=5),
    ))
    fig.add_trace(go.Histogram(
        x=asks.clip(lo, hi), name="Ask orders (sell side)",
        marker_color=COLOR_ASK, opacity=0.7,
        xbins=dict(size=5),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="white",
                  annotation_text="mid-price", annotation_position="top")

    fig.update_layout(
        title=dict(
            text=f"Limit Order Placement Distribution — {firm_label(mpid)}<br>"
                 f"<sup>{TICKER_FULL} | {DATE_RANGE} | "
                 f"x-axis = price of order minus mid-price at submission (basis points, 1bp = $0.09) | "
                 f"negative = placed below mid (bid side), positive = placed above mid (ask side)</sup>",
            font_size=13,
        ),
        xaxis_title="Order Price vs Mid-Price at Submission (basis points)",
        yaxis_title="Number of Orders",
        barmode="overlay",
        template="plotly_dark",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        xaxis=dict(range=[lo, hi]),
    )
    out = OUTPUT_DIR / f"price_hist_{mpid}.html"
    fig.write_html(str(out))
    print(f"  Saved {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Scatter: time vs price-to-mid, bid/ask coloured, per firm
# ═══════════════════════════════════════════════════════════════════════════════
print("\nChart 2: Placement scatter (time vs bps from mid)...")
for mpid, df in firm_dfs.items():
    sub = df[["date","time_et_h","direction","price_to_mid_bps","sub_price",
              "hold_s","filled"]].dropna(subset=["price_to_mid_bps"]).copy()
    # sample for performance if huge
    if len(sub) > 30_000:
        sub = sub.sample(30_000, random_state=42)

    lo, hi = sub["price_to_mid_bps"].quantile(0.01), sub["price_to_mid_bps"].quantile(0.99)
    sub = sub[(sub.price_to_mid_bps >= lo) & (sub.price_to_mid_bps <= hi)]

    fig = go.Figure()
    for direction, color, label in [("bid", COLOR_BID, "Bid (buy)"),
                                     ("ask", COLOR_ASK, "Ask (sell)")]:
        side = sub[sub.direction == direction]
        fig.add_trace(go.Scatter(
            x=side["time_et_h"],
            y=side["price_to_mid_bps"],
            mode="markers",
            name=label,
            marker=dict(color=color, size=2, opacity=0.4),
            text=[f"{d} {h:.2f}h | hold={hs:.2f}s"
                  for d, h, hs in zip(side.date, side.time_et_h, side.hold_s)],
            hovertemplate="%{text}<br>bps from mid: %{y:.1f}",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5,
                  annotation_text="mid-price", annotation_position="right")

    # annotate session phases
    for label, t in [("Pre-mkt\n6:55AM", 6.916), ("Open\n9:30AM", 9.5),
                      ("Mid-day\n12pm", 12.0), ("Close\n4pm", 16.0)]:
        fig.add_vline(x=t, line_dash="dash", line_color="grey", opacity=0.4,
                      annotation_text=label, annotation_position="top")

    fig.update_layout(
        title=dict(
            text=f"When and Where Orders Are Placed — {firm_label(mpid)}<br>"
                 f"<sup>{TICKER_FULL} | {DATE_RANGE} | "
                 f"x-axis = time of day (hours ET) | "
                 f"y-axis = order price vs mid-price at submission (bps) | "
                 f"each dot = one order submission</sup>",
            font_size=13,
        ),
        xaxis_title="Time of Day (hours ET, e.g. 9.5 = 9:30 AM)",
        yaxis_title="Order Price vs Mid-Price (bps, 0 = at mid)",
        template="plotly_dark",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    out = OUTPUT_DIR / f"placement_scatter_{mpid}.html"
    fig.write_html(str(out))
    print(f"  Saved {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Order ladder: actual order prices vs COST trading range, per firm
#            Picks the day with most orders for each firm
# ═══════════════════════════════════════════════════════════════════════════════
print("\nChart 3: Price ladder (order prices vs COST trading range)...")
for mpid, df in firm_dfs.items():
    # pick busiest day
    busiest_day = df.groupby("date").size().idxmax()
    day_df = df[df.date == busiest_day].copy()
    day_tape = tape[tape.date == busiest_day]

    # COST actual price range on that day
    cost_min = day_tape["mid"].min()
    cost_max = day_tape["mid"].max()
    cost_mean = day_tape["mid"].mean()

    bids = day_df[day_df.direction == "bid"]["sub_price"].dropna()
    asks = day_df[day_df.direction == "ask"]["sub_price"].dropna()

    fig = go.Figure()

    # COST trading band
    fig.add_hrect(y0=cost_min, y1=cost_max,
                  fillcolor="yellow", opacity=0.08,
                  annotation_text=f"COST actual trading range {busiest_day}",
                  annotation_position="top right")
    fig.add_hline(y=cost_mean, line_color="yellow", line_dash="dot",
                  annotation_text=f"COST mean mid ${cost_mean:.2f}",
                  annotation_position="right")

    # bid order prices as histogram on its side
    if len(bids):
        counts, edges = np.histogram(bids, bins=60)
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(
            x=-counts,
            y=centers,
            orientation="h",
            name="Bid orders (buys)",
            marker_color=COLOR_BID,
            opacity=0.7,
        ))
    if len(asks):
        counts, edges = np.histogram(asks, bins=60)
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(
            x=counts,
            y=centers,
            orientation="h",
            name="Ask orders (sells)",
            marker_color=COLOR_ASK,
            opacity=0.7,
        ))

    fig.update_layout(
        title=dict(
            text=f"Order Price Ladder vs COST Trading Range — {firm_label(mpid)}<br>"
                 f"<sup>{TICKER_FULL} | Busiest day shown: {busiest_day} | "
                 f"y-axis = order submission price (USD) | "
                 f"x-axis = order count (left=bids, right=asks) | "
                 f"yellow band = actual COST price range that day</sup>",
            font_size=13,
        ),
        xaxis_title="Order Count (negative = bids, positive = asks)",
        yaxis_title="Order Submission Price (USD)",
        barmode="overlay",
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    out = OUTPUT_DIR / f"ladder_snapshot_{mpid}.html"
    fig.write_html(str(out))
    print(f"  Saved {out.name}  (day={busiest_day}  COST range ${cost_min:.2f}–${cost_max:.2f})")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Bid-vs-ask placement violin, all firms on one plot
# ═══════════════════════════════════════════════════════════════════════════════
print("\nChart 4: Bid vs Ask placement violin (all firms)...")
fig_vio = go.Figure()

for mpid, df in firm_dfs.items():
    p2m = df["price_to_mid_bps"].dropna()
    lo, hi = p2m.quantile(0.02), p2m.quantile(0.98)
    name = FIRM_NAMES.get(mpid, mpid)[:28]

    for direction, color, offset in [("bid", COLOR_BID, -0.2), ("ask", COLOR_ASK, 0.2)]:
        vals = df[df.direction == direction]["price_to_mid_bps"].dropna()
        vals = vals.clip(lo, hi)
        if vals.empty:
            continue
        fig_vio.add_trace(go.Violin(
            y=vals,
            name=f"{mpid} {direction}",
            legendgroup=direction,
            scalegroup=direction,
            side="negative" if direction == "bid" else "positive",
            line_color=color,
            fillcolor=color,
            opacity=0.5,
            meanline_visible=True,
            x0=mpid,
            points=False,
            showlegend=(mpid == FOCUS_FIRMS[0]),
        ))

fig_vio.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4,
                  annotation_text="mid-price", annotation_position="right")

fig_vio.update_layout(
    title=dict(
        text=f"Bid vs Ask Order Placement Distance — All Firms<br>"
             f"<sup>{TICKER_FULL} | {DATE_RANGE} | "
             f"y-axis = order price minus mid at submission (bps) | "
             f"left half of each violin = bid orders, right half = ask orders | "
             f"width shows density of orders at each price distance</sup>",
        font_size=13,
    ),
    xaxis_title="Firm (MPID)",
    yaxis_title="Order Price vs Mid at Submission (basis points)",
    violingap=0.05,
    violinmode="overlay",
    template="plotly_dark",
    height=560,
    legend=dict(title="Side", orientation="h", yanchor="bottom", y=1.01),
)
out = OUTPUT_DIR / "bid_ask_violin.html"
fig_vio.write_html(str(out))
print(f"  Saved {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5 — Hourly order submission activity per firm
# ═══════════════════════════════════════════════════════════════════════════════
print("\nChart 5: Hourly activity per firm...")
fig_act = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f"{m} — {FIRM_NAMES.get(m,'?')[:30]}" for m in FOCUS_FIRMS],
    shared_yaxes=False,
)

for i, mpid in enumerate(FOCUS_FIRMS):
    if mpid not in firm_dfs:
        continue
    df  = firm_dfs[mpid]
    row = i // 3 + 1
    col = i % 3 + 1

    hourly = df.groupby(["hour_et","direction"]).size().unstack(fill_value=0).reset_index()
    for direction, color in [("bid", COLOR_BID), ("ask", COLOR_ASK)]:
        if direction not in hourly.columns:
            continue
        fig_act.add_trace(
            go.Bar(
                x=hourly["hour_et"],
                y=hourly[direction],
                name=f"{direction}",
                marker_color=color,
                opacity=0.8,
                showlegend=(i == 0),
                legendgroup=direction,
            ),
            row=row, col=col,
        )

    # shade regular session
    for r in range(row, row+1):
        fig_act.add_vrect(x0=9.5, x1=16.0, fillcolor="white", opacity=0.04,
                          row=r, col=col)

fig_act.update_xaxes(title_text="Hour (ET)", tickvals=list(range(6,17)),
                     ticktext=[f"{h}:00" for h in range(6,17)])
fig_act.update_yaxes(title_text="Order Count")
fig_act.update_layout(
    title=dict(
        text=f"Hourly Order Submission Activity — All Firms<br>"
             f"<sup>{TICKER_FULL} | {DATE_RANGE} | "
             f"Aggregated across all 10 trading days | "
             f"White band = regular session 9:30AM–4:00PM ET | "
             f"Blue = bid (buy) orders, Red = ask (sell) orders</sup>",
        font_size=13,
    ),
    barmode="stack",
    template="plotly_dark",
    height=560,
    legend=dict(orientation="h", yanchor="bottom", y=1.01),
)
out = OUTPUT_DIR / "activity_by_hour.html"
fig_act.write_html(str(out))
print(f"  Saved {out.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6 — Holding time vs price distance (do far orders live longer?)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nChart 6: Hold time vs price distance...")
fig_hold = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f"{m} — {FIRM_NAMES.get(m,'?')[:30]}" for m in FOCUS_FIRMS],
)
for i, mpid in enumerate(FOCUS_FIRMS):
    if mpid not in firm_dfs:
        continue
    df  = firm_dfs[mpid]
    row = i // 3 + 1
    col = i % 3 + 1
    sub = df[["price_to_mid_bps","hold_s","direction"]].dropna()
    p2m = sub["price_to_mid_bps"]
    lo, hi = p2m.quantile(0.02), p2m.quantile(0.98)
    sub = sub[(sub.price_to_mid_bps >= lo) & (sub.price_to_mid_bps <= hi)]
    if len(sub) > 8000:
        sub = sub.sample(8000, random_state=0)

    for direction, color in [("bid", COLOR_BID), ("ask", COLOR_ASK)]:
        s = sub[sub.direction == direction]
        fig_hold.add_trace(
            go.Scatter(
                x=s["price_to_mid_bps"],
                y=s["hold_s"].clip(upper=s["hold_s"].quantile(0.97)),
                mode="markers",
                marker=dict(color=color, size=2, opacity=0.3),
                name=direction,
                showlegend=(i == 0),
                legendgroup=direction,
            ),
            row=row, col=col,
        )

fig_hold.update_xaxes(title_text="bps from mid")
fig_hold.update_yaxes(title_text="Hold time (s)")
fig_hold.update_layout(
    title=dict(
        text=f"Order Holding Time vs Price Distance from Mid — All Firms<br>"
             f"<sup>{TICKER_FULL} | {DATE_RANGE} | "
             f"x-axis = how far order was placed from mid (bps) | "
             f"y-axis = seconds before order was deleted or filled | "
             f"shows whether far-away orders live longer</sup>",
        font_size=13,
    ),
    template="plotly_dark",
    height=560,
    legend=dict(orientation="h", yanchor="bottom", y=1.01),
)
out = OUTPUT_DIR / "hold_vs_distance.html"
fig_hold.write_html(str(out))
print(f"  Saved {out.name}")

print("\nAll order placement charts saved to output/")
print("Open .html files in browser for interactive exploration.")
