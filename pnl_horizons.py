"""
pnl_horizons.py

For each firm, takes all confirmed order fills across all available tickers
and computes/charts PnL at four horizons: 0.1s, 1s, 30s, EOD.

Charts produced:
  1. Cumulative PnL trajectory over time — one subplot per firm, 4 lines per horizon
  2. Total PnL bar chart — firms on x-axis, grouped bars per horizon
  3. PnL distribution box plots — spread of per-order PnL at each horizon per firm
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR  = Path(r"C:\Users\komal\Project285J\output")
DATE_RANGE  = "Oct 1–14 2025"
TICKER_FULL = "COST (Costco Wholesale, NASDAQ)"

FIRM_NAMES = {
    "WBPX": "Wedbush Securities Inc.",
    "MAXM": "Maxim Group LLC",
    "WCHV": "Wells Fargo Securities LLC",
    "JPMS": "J.P. Morgan Securities LLC",
    "UBSS": "UBS Securities LLC",
    "ETMM": "G1 Execution Services LLC",
    "IMCC": "IM Cannabis Corp. [ticker anomaly]",
}

HORIZONS   = ["pnl_0.1s", "pnl_1s", "pnl_30s", "pnl_EOD"]
H_LABELS   = {
    "pnl_0.1s": "T+0.1s (100ms after fill)",
    "pnl_1s":   "T+1s (1 second after fill)",
    "pnl_30s":  "T+30s (30 seconds after fill)",
    "pnl_EOD":  "T=EOD (end-of-day mark-to-market)",
}
H_COLORS   = {"pnl_0.1s": "#00d4ff", "pnl_1s": "#00ff99",
               "pnl_30s": "#ffaa00", "pnl_EOD": "#ff4466"}

def firm_full(mpid: str) -> str:
    return f"{mpid} — {FIRM_NAMES.get(mpid, 'Unknown')}"

def chart_sub(extra: str = "") -> str:
    base = f"{TICKER_FULL} | {DATE_RANGE} | Only confirmed fills (vis_exec / hid_exec)"
    return base + (f" | {extra}" if extra else "")

COLOR_MAP = {
    "WBPX":"#1f77b4","MAXM":"#ff7f0e","WCHV":"#2ca02c",
    "JPMS":"#d62728","UBSS":"#9467bd","ETMM":"#8c564b","IMCC":"#e377c2"
}

# ── Load fills from all firm CSVs ─────────────────────────────────────────────
print("Loading fills from per-firm CSVs...")
firm_csvs = sorted(OUTPUT_DIR.glob("firm_*_orders.csv"))
all_fills = []

for csv_path in firm_csvs:
    firm = csv_path.stem.replace("firm_", "").replace("_orders", "")
    df   = pd.read_csv(csv_path, low_memory=False)
    fills = df[df["filled"] == True].copy()
    fills["firm"] = firm
    all_fills.append(fills)
    print(f"  {firm:6s}: {len(fills):>4} fills")

if not all_fills:
    print("No fills found — nothing to chart.")
    raise SystemExit

fills = pd.concat(all_fills, ignore_index=True)

# ensure horizon columns exist
for h in HORIZONS:
    if h not in fills.columns:
        fills[h] = np.nan

# combine date + submit time into a sortable label
fills["dt_label"] = fills["date"] + " " + fills["t_submit"].apply(
    lambda s: f"{int(float(s))//3600:02d}:{(int(float(s))%3600)//60:02d}"
)
fills = fills.sort_values(["date","t_submit"]).reset_index(drop=True)

firms_with_fills = fills[fills[HORIZONS].notna().any(axis=1)]["firm"].unique()
print(f"\n  Firms with at least 1 fill: {list(firms_with_fills)}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\nPnL summary by firm and horizon:")
summary_rows = []
for firm in fills["firm"].unique():
    sub = fills[fills.firm == firm]
    row = {"firm": firm, "n_fills": len(sub)}
    for h in HORIZONS:
        s = sub[h].dropna()
        row[f"{H_LABELS[h]}_total"]  = round(s.sum(), 2)
        row[f"{H_LABELS[h]}_mean"]   = round(s.mean(), 4) if len(s) else 0
        row[f"{H_LABELS[h]}_n"]      = len(s)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "pnl_horizons_summary.csv", index=False)
print(summary_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Cumulative PnL trajectory, one subplot per firm, 4 horizon lines
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Chart 1: Cumulative PnL trajectories...")

all_firms = sorted(fills["firm"].unique())
n_firms   = len(all_firms)
cols      = min(n_firms, 3)
rows      = (n_firms + cols - 1) // cols

fig1 = make_subplots(
    rows=rows, cols=cols,
    subplot_titles=[firm_full(f) for f in all_firms],
    shared_xaxes=False,
)

for i, firm in enumerate(all_firms):
    row_i = i // cols + 1
    col_i = i % cols + 1
    sub   = fills[fills.firm == firm].sort_values(["date","t_submit"])
    n_fills = len(sub)

    for h in HORIZONS:
        vals = sub[h].fillna(0)
        cum  = vals.cumsum()
        fig1.add_trace(
            go.Scatter(
                x=sub["dt_label"],
                y=cum,
                mode="lines",
                name=H_LABELS[h],
                line=dict(color=H_COLORS[h], width=1.5),
                showlegend=(i == 0),
                legendgroup=h,
                hovertemplate=f"{H_LABELS[h]}<br>Date/Time: %{{x}}<br>Cumulative PnL: $%{{y:,.2f}}<extra></extra>",
            ),
            row=row_i, col=col_i,
        )
    fig1.add_hline(y=0, line_dash="dot", line_color="white",
                   opacity=0.3, row=row_i, col=col_i)

fig1.update_layout(
    title=dict(
        text=f"Cumulative PnL by Holding Horizon — Per Firm<br>"
             f"<sup>{TICKER_FULL} | {DATE_RANGE} | "
             f"Each line = total PnL summed across all fills up to that point in time | "
             f"PnL measured T seconds after each fill executes</sup>",
        font_size=13,
    ),
    template="plotly_dark",
    height=max(400, 320 * rows),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, title="Horizon (time after fill)"),
)
fig1.write_html(str(OUTPUT_DIR / "chart_pnl_trajectories.html"))
print("  Saved chart_pnl_trajectories.html")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Total PnL bar chart: firms × horizon
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 2: Total PnL bar chart...")

fig2 = go.Figure()
for h in HORIZONS:
    totals = [
        fills[fills.firm == f][h].sum()
        for f in all_firms
    ]
    fig2.add_trace(go.Bar(
        name=H_LABELS[h],
        x=all_firms,
        y=totals,
        marker_color=H_COLORS[h],
        text=[f"${v:,.0f}" for v in totals],
        textposition="outside",
    ))

fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
fig2.update_layout(
    title=dict(
        text=f"Total PnL per Firm — Grouped by Holding Horizon<br>"
             f"<sup>{TICKER_FULL} | {DATE_RANGE} | "
             f"Each bar = sum of all fills' PnL measured T seconds after execution | "
             f"Firms with 0 fills show $0 for all horizons</sup>",
        font_size=13,
    ),
    xaxis=dict(title="Firm (MPID — hover for full name)",
               ticktext=[firm_full(f) for f in all_firms], tickvals=all_firms),
    yaxis_title="Total PnL USD (all fills summed)",
    barmode="group",
    template="plotly_dark",
    height=540,
    legend=dict(title="Horizon (time after fill)"),
)
fig2.write_html(str(OUTPUT_DIR / "chart_pnl_totals.html"))
print("  Saved chart_pnl_totals.html")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Per-order PnL box plots by firm × horizon
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 3: PnL distribution box plots...")

fig3 = go.Figure()
for h in HORIZONS:
    for firm in all_firms:
        vals = fills[fills.firm == firm][h].dropna()
        if vals.empty:
            continue
        fig3.add_trace(go.Box(
            y=vals,
            name=f"{firm} — {H_LABELS[h]}",
            marker_color=H_COLORS[h],
            boxmean=True,
            showlegend=False,
        ))

fig3.update_layout(
    title=dict(
        text=f"Per-Order PnL Distribution — Each Fill Individually, by Firm and Horizon<br>"
             f"<sup>{TICKER_FULL} | {DATE_RANGE} | "
             f"Box = 25th–75th percentile | Line = median | X = mean | "
             f"Each data point is one confirmed fill's PnL T seconds after execution</sup>",
        font_size=13,
    ),
    yaxis_title="PnL per Individual Fill (USD)",
    template="plotly_dark",
    height=550,
)
fig3.write_html(str(OUTPUT_DIR / "chart_pnl_distributions.html"))
print("  Saved chart_pnl_distributions.html")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — PnL decay: how does mean PnL change from 0.1s → EOD?
#           Shows whether fills are profitable short-term but decay
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 4: PnL decay across horizons...")

all_h_cols = ["pnl_0.001s","pnl_0.01s","pnl_0.05s","pnl_0.1s","pnl_0.25s",
              "pnl_0.5s","pnl_1s","pnl_2s","pnl_5s","pnl_10s",
              "pnl_30s","pnl_60s","pnl_300s","pnl_1800s","pnl_EOD"]
all_h_labels = ["1ms","10ms","50ms","100ms","250ms","500ms",
                "1s","2s","5s","10s","30s","1min","5min","30min","EOD"]

fig4 = go.Figure()
for firm in all_firms:
    sub   = fills[fills.firm == firm]
    means = [sub[h].mean() if h in sub.columns else np.nan for h in all_h_cols]
    if all(np.isnan(m) for m in means):
        continue
    fig4.add_trace(go.Scatter(
        x=all_h_labels,
        y=means,
        mode="lines+markers",
        name=firm_full(firm),
        line=dict(color=COLOR_MAP.get(firm), width=2),
        marker=dict(size=7),
        hovertemplate="Horizon: %{x}<br>Mean PnL per fill: $%{y:,.2f}<extra></extra>",
    ))

fig4.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4,
               annotation_text="Break-even", annotation_position="right")
fig4.update_layout(
    title=dict(
        text=f"PnL Decay Curve — Mean PnL per Fill from 1ms to EOD<br>"
             f"<sup>{TICKER_FULL} | {DATE_RANGE} | "
             f"x-axis = time elapsed after fill execution | "
             f"y-axis = average dollar PnL per fill at that horizon | "
             f"crossing zero = point at which fills become net losers on average</sup>",
        font_size=13,
    ),
    xaxis_title="Time Elapsed After Fill Execution",
    yaxis_title="Mean PnL per Fill (USD)",
    template="plotly_dark",
    height=500,
    legend=dict(title="Firm"),
)
fig4.write_html(str(OUTPUT_DIR / "chart_pnl_decay.html"))
print("  Saved chart_pnl_decay.html")

print("\nAll charts written to output/")
print("Open the .html files in a browser for interactive charts.")
print("\nDone.")
