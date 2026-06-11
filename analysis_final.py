"""
analysis_final.py

Processes AVAV, KTOS, TALO LOBSTER data (Feb 23 – Mar 27, 2026).
Produces per-firm PNG charts with:
  - Order placement (histogram of distance from mid, price ladder, hourly activity)
  - Same-side PnL benchmark (compares fill price to best same-side price T seconds later,
    NOT to mid-price — this correctly shows adverse selection)
  - Portfolio simulation (position + PnL) for firms with fills
  - Hold time distribution

Deletes all existing HTML files in output/ before saving PNGs.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
BASE       = Path(r"C:\Users\komal\Project285J")
DATA_BASE  = BASE / "data"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TICKERS      = ["AVAV", "KTOS", "TALO"]
DATE_RANGE   = "Feb 23 – Mar 27, 2026"
PRICE_SCALE  = 0.0001

PNL_HORIZONS_S = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 300, 1800]
H_LABELS       = ["1ms","10ms","50ms","100ms","250ms","500ms",
                   "1s","2s","5s","10s","30s","1min","5min","30min"]

FIRM_NAMES = {
    "WBPX": "Wedbush Securities", "JPMS": "JPMorgan Securities",
    "WCHV": "Wells Fargo Securities", "UBSS": "UBS Securities",
    "SGAS": "Susquehanna Fin. Group", "GSCO": "Goldman Sachs",
    "ETMM": "G1 Execution Services", "STFL": "Stifel Nicolaus",
    "FLTG": "Fidelity / FLTG", "IMCC": "IM Cannabis [anomaly]",
    "VIRT": "Virtu Financial", "MLCO": "Merrill Lynch / BofA",
    "TSSM": "Two Sigma Securities", "GTSM": "GTS Securities",
    "SUFI": "SuFi / Unknown", "SSUS": "SS&C / Unknown",
    "BTBS": "BTBS / Unknown", "NEED": "NEED / Unknown",
    "MSCO": "Morgan Stanley", "SPHN": "SPHN / Unknown",
    "KING": "KCG / Virtu",
}

# dominant firms to chart (skip tiny/anomaly ones)
FOCUS_FIRMS = ["WBPX","JPMS","WCHV","UBSS","SGAS","GSCO","ETMM","STFL","VIRT","MLCO"]

TICKER_COLORS = {"AVAV": "#4C9BE8", "KTOS": "#E8834C", "TALO": "#4CE896"}
STYLE = {"bid": "#5BC8F5", "ask": "#F55B5B"}

# ── 1. Delete all existing HTML files ─────────────────────────────────────────
deleted = list(OUTPUT_DIR.glob("*.html"))
for f in deleted:
    f.unlink()
print(f"Deleted {len(deleted)} HTML files")

# ── 2. Load all labeled orders for each ticker ─────────────────────────────────
print("Loading labeled orders...")

ticker_data = {}   # ticker -> DataFrame of all labeled orders (all files)
for ticker in TICKERS:
    parts = []
    for fpath in sorted((DATA_BASE / ticker).glob(f"{ticker}_*_message_0.csv")):
        date_str = fpath.name.split("_")[1]
        df = pd.read_csv(str(fpath), header=None,
                         names=["time","type","oid","size","price","dir","firm"],
                         low_memory=False)
        df["date"]   = date_str
        df["ticker"] = ticker
        df["type_n"] = pd.to_numeric(df["type"], errors="coerce")
        df["dir_n"]  = pd.to_numeric(df["dir"],  errors="coerce")
        df["price_usd"] = pd.to_numeric(df["price"], errors="coerce") * PRICE_SCALE
        df["time"]   = pd.to_numeric(df["time"], errors="coerce")
        labeled = df[df["firm"].notna() & ~df["firm"].isin(["null",""])]
        parts.append(labeled)
    full = pd.concat(parts, ignore_index=True)
    ticker_data[ticker] = full
    subs  = (full.type_n == 1).sum()
    fills = full.type_n.isin([4,5]).sum()
    print(f"  {ticker}: {len(full):,} labeled rows | {subs:,} submits | {fills:,} fills")

# ── 3. Build execution price series per ticker per day (for same-side PnL) ────
print("\nBuilding same-side execution price series...")

exec_series = {}  # (ticker, date, 'bid'|'ask') -> (times_arr, prices_arr)

for ticker, df in ticker_data.items():
    execs = df[df.type_n.isin([4, 5])].copy()
    execs["dir_str"] = execs["dir_n"].map({1: "bid", -1: "ask"})
    for (date_str, side), grp in execs.groupby(["date","dir_str"]):
        grp_s = grp.sort_values("time")
        exec_series[(ticker, date_str, side)] = (
            grp_s["time"].values.astype(np.float64),
            grp_s["price_usd"].values.astype(np.float64)
        )

def best_same_side_price(ticker: str, date: str, t: float, side: str) -> float:
    """Return the nearest same-side execution price at or after time t."""
    key = (ticker, date, side)
    if key not in exec_series:
        return np.nan
    times, prices = exec_series[key]
    idx = int(np.searchsorted(times, t, side="left"))
    if idx >= len(times):
        idx = len(times) - 1
    return float(prices[idx])

def mid_at(ticker: str, date: str, t: float) -> float:
    """Return mid-price (average of nearest bid and ask exec) at time t."""
    b = best_same_side_price(ticker, date, t, "bid")
    a = best_same_side_price(ticker, date, t, "ask")
    if np.isnan(b) or np.isnan(a):
        return (b if not np.isnan(b) else a)
    return (b + a) / 2

# ── 4. Build order lifecycles per ticker ──────────────────────────────────────
print("Building order lifecycles...")

all_lifecycles = []

for ticker, df in ticker_data.items():
    subs = (df[df.type_n == 1]
              [["date","ticker","firm","oid","time","price_usd","dir_n","size"]]
              .rename(columns={"time":"t_sub","price_usd":"sub_price",
                               "size":"sub_size"})
              .drop_duplicates(subset=["date","oid"]))
    subs["dir_str"] = subs["dir_n"].map({1:"bid",-1:"ask"})

    terms = (df[df.type_n.isin([2,3,4,5])]
               [["date","oid","time","type_n","price_usd"]]
               .rename(columns={"time":"t_term","type_n":"term_type",
                                "price_usd":"term_price"})
               .sort_values("t_term")
               .drop_duplicates(subset=["date","oid"], keep="first"))

    lc = subs.merge(terms, on=["date","oid"], how="left")
    lc["hold_s"] = lc["t_term"] - lc["t_sub"]
    lc["filled"] = lc["term_type"].isin([4.0, 5.0])
    lc = lc[lc.hold_s >= 0].copy()
    all_lifecycles.append(lc)

lifecycle = pd.concat(all_lifecycles, ignore_index=True)
fills_df  = lifecycle[lifecycle.filled].copy()

print(f"  {len(lifecycle):,} lifecycles | {len(fills_df):,} fills total")
print("  Fills by firm+ticker:")
if len(fills_df):
    print(fills_df.groupby(["firm","ticker"]).size().to_string())

# ── 5. Compute same-side PnL for all fills ─────────────────────────────────────
print("\nComputing same-side PnL...")

pnl_rows = []
for rec in fills_df.itertuples(index=False):
    t    = rec.t_sub
    p    = rec.sub_price
    side = rec.dir_str
    size = rec.sub_size
    d    = rec.date
    tk   = rec.ticker
    d_i  = 1 if side == "bid" else -1

    row = {"firm": rec.firm, "ticker": tk, "date": d,
           "t_sub": t, "direction": side, "size": size, "fill_price": p}

    for T, label in zip(PNL_HORIZONS_S, H_LABELS):
        best = best_same_side_price(tk, d, t + T, side)
        if np.isnan(best):
            row[f"pnl_{label}"] = np.nan
        else:
            # bid:  (best_bid_T - fill_price) * size  [positive = stock went up = good buy]
            # ask:  (fill_price - best_ask_T) * size  [positive = stock went down = good sell]
            if side == "bid":
                row[f"pnl_{label}"] = round((best - p) * size, 4)
            else:
                row[f"pnl_{label}"] = round((p - best) * size, 4)

    pnl_rows.append(row)

pnl_df = pd.DataFrame(pnl_rows) if pnl_rows else pd.DataFrame()

# ── 6. Portfolio simulation per firm ──────────────────────────────────────────
print("Simulating portfolios...")

portfolio_by_firm = {}  # firm -> {ticker -> [(t, position, cash, pnl_usd)]}

for firm in fills_df["firm"].unique() if len(fills_df) else []:
    firm_fills = fills_df[fills_df.firm == firm].sort_values(["date","t_sub"])
    port = {}
    for ticker, grp in firm_fills.groupby("ticker"):
        pos, cash = 0, 0.0
        events = []
        for r in grp.itertuples(index=False):
            sz  = int(r.sub_size)
            p   = float(r.sub_price)
            t   = float(r.t_sub)
            mid = mid_at(r.ticker, r.date, t)
            if r.dir_str == "bid":
                pos += sz; cash -= sz * p
            elif r.dir_str == "ask" and pos >= sz:
                pos -= sz; cash += sz * p
            else:
                continue
            unreal = pos * mid if not np.isnan(mid) else 0.0
            events.append({"date": r.date, "t": t, "pos": pos,
                           "cash": cash, "pnl": cash + unreal})
        if events:
            port[ticker] = pd.DataFrame(events)
    if port:
        portfolio_by_firm[firm] = port

# ── 7. Create PNG charts per firm ─────────────────────────────────────────────
print("\nGenerating per-firm PNG charts...")

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
    "axes.edgecolor": "#4a4a7a", "text.color": "white",
    "axes.labelcolor": "white", "xtick.color": "white",
    "ytick.color": "white", "grid.color": "#2a2a4a",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.size": 8,
})

def fmt_name(mpid: str) -> str:
    return f"{mpid} ({FIRM_NAMES.get(mpid, 'Unknown')})"

def make_placement_figure(mpid: str, lc_firm: pd.DataFrame) -> plt.Figure:
    """Per-firm order placement figure — works for all firms."""
    tickers_present = sorted(lc_firm["ticker"].unique())
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle(
        f"{fmt_name(mpid)}\nLimit Order Placement Analysis   |   {DATE_RANGE}   |   "
        f"Tickers: {', '.join(tickers_present)}\n"
        f"Total orders: {len(lc_firm):,}   Fills: {lc_firm.filled.sum():,}   "
        f"Fill rate: {lc_firm.filled.mean()*100:.2f}%",
        fontsize=10, y=1.0, color="white", wrap=True
    )

    # ── Panel A: Histogram of price-to-mid bps per ticker ─────────────────────
    ax_hist = fig.add_subplot(gs[0, 0])
    for ticker in tickers_present:
        sub = lc_firm[lc_firm.ticker == ticker]
        mid_s = sub.apply(lambda r: mid_at(r.ticker, r.date, r.t_sub), axis=1)
        p2m = ((sub["sub_price"] - mid_s) / (mid_s + 1e-9) * 10000).dropna()
        lo, hi = p2m.quantile(0.02), p2m.quantile(0.98)
        p2m_clip = p2m.clip(lo, hi)
        ax_hist.hist(p2m_clip, bins=60, alpha=0.55, color=TICKER_COLORS[ticker],
                     label=ticker, density=True)
    ax_hist.axvline(0, color="white", lw=1, ls="--", alpha=0.6, label="Mid-price")
    ax_hist.set_xlabel("Order price vs mid-price (bps)\nneg=below mid, pos=above mid")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("A. Placement Distribution vs Mid-Price", pad=4)
    ax_hist.legend(fontsize=7, loc="upper right")
    ax_hist.grid(True, alpha=0.3)

    # ── Panel B: Price ladder (order prices vs actual stock range) ─────────────
    ax_ladder = fig.add_subplot(gs[0, 1])
    for ticker in tickers_present:
        sub = lc_firm[lc_firm.ticker == ticker]
        bids = sub[sub.dir_str == "bid"]["sub_price"].dropna()
        asks = sub[sub.dir_str == "ask"]["sub_price"].dropna()
        # actual trading range from exec series
        all_exec = []
        for key, (ts, ps) in exec_series.items():
            if key[0] == ticker:
                all_exec.extend(ps)
        if all_exec:
            p_min, p_max = np.percentile(all_exec, [1, 99])
            ax_ladder.axhspan(p_min, p_max, alpha=0.08,
                              color=TICKER_COLORS[ticker],
                              label=f"{ticker} trading range")
        if len(bids):
            cnt_b, edg_b = np.histogram(bids, bins=40)
            ax_ladder.barh((edg_b[:-1]+edg_b[1:])/2, -cnt_b/cnt_b.max()*0.4,
                           height=(edg_b[1]-edg_b[0])*0.85,
                           color=STYLE["bid"], alpha=0.6, left=-0.42*(len(tickers_present)>1))
        if len(asks):
            cnt_a, edg_a = np.histogram(asks, bins=40)
            ax_ladder.barh((edg_a[:-1]+edg_a[1:])/2, cnt_a/cnt_a.max()*0.4,
                           height=(edg_a[1]-edg_a[0])*0.85,
                           color=STYLE["ask"], alpha=0.6)
    ax_ladder.set_xlabel("Normalised order count\n(left=bid, right=ask)")
    ax_ladder.set_ylabel("Submission price (USD)")
    ax_ladder.set_title("B. Order Price Ladder\nvs Actual Trading Range (shaded)", pad=4)
    ax_ladder.legend(fontsize=6, loc="lower right")
    ax_ladder.grid(True, alpha=0.3)

    # ── Panel C: Hourly activity by ticker ─────────────────────────────────────
    ax_hour = fig.add_subplot(gs[0, 2])
    hours = list(range(6, 17))
    bottom = np.zeros(len(hours))
    for ticker in tickers_present:
        sub = lc_firm[lc_firm.ticker == ticker]
        sub = sub[sub.type_n == 1] if "type_n" in sub.columns else sub
        hr_counts = sub["t_sub"].apply(lambda t: int(t // 3600)).value_counts()
        counts = [hr_counts.get(h, 0) for h in hours]
        ax_hour.bar(hours, counts, bottom=bottom, color=TICKER_COLORS[ticker],
                    alpha=0.8, label=ticker, width=0.7)
        bottom += np.array(counts)
    ax_hour.axvspan(9.5, 16.0, alpha=0.07, color="white")
    ax_hour.set_xticks(hours)
    ax_hour.set_xticklabels([f"{h}:00" for h in hours], rotation=45, fontsize=7)
    ax_hour.set_xlabel("Hour of Day (ET)")
    ax_hour.set_ylabel("Order submissions")
    ax_hour.set_title("C. Hourly Order Activity\n(shaded = regular session 9:30–16:00)", pad=4)
    ax_hour.legend(fontsize=7)
    ax_hour.grid(True, alpha=0.3, axis="y")

    # ── Panel D: Hold-time distribution ───────────────────────────────────────
    ax_hold = fig.add_subplot(gs[1, 0])
    for ticker in tickers_present:
        sub = lc_firm[lc_firm.ticker == ticker]["hold_s"].dropna()
        sub = sub[sub > 0]
        if sub.empty:
            continue
        p99 = sub.quantile(0.99)
        ax_hold.hist(sub.clip(upper=p99), bins=60, alpha=0.55,
                     color=TICKER_COLORS[ticker], label=ticker, density=True)
    ax_hold.set_xlabel("Hold time (seconds)\nfrom submit to delete/execute")
    ax_hold.set_ylabel("Density")
    ax_hold.set_title("D. Order Holding Time Distribution\n(clipped at 99th pct)", pad=4)
    ax_hold.legend(fontsize=7)
    ax_hold.grid(True, alpha=0.3)

    # ── Panel E: Bid vs ask price-to-mid box ──────────────────────────────────
    ax_box = fig.add_subplot(gs[1, 1])
    all_mid = []
    for ticker in tickers_present:
        sub = lc_firm[lc_firm.ticker == ticker]
        mid_s = sub.apply(lambda r: mid_at(r.ticker, r.date, r.t_sub), axis=1)
        p2m = ((sub["sub_price"] - mid_s) / (mid_s + 1e-9) * 10000).dropna()
        lo, hi = p2m.quantile(0.02), p2m.quantile(0.98)
        bids_p2m = p2m[sub.dir_str == "bid"].clip(lo, hi).values
        asks_p2m = p2m[sub.dir_str == "ask"].clip(lo, hi).values
        all_mid.append((ticker, bids_p2m, asks_p2m))

    positions = []
    labels_box = []
    data_box = []
    for i, (ticker, bids_p2m, asks_p2m) in enumerate(all_mid):
        base = i * 3
        if len(bids_p2m) > 10:
            data_box.append(bids_p2m)
            positions.append(base)
            labels_box.append(f"{ticker}\nbid")
        if len(asks_p2m) > 10:
            data_box.append(asks_p2m)
            positions.append(base + 1.2)
            labels_box.append(f"{ticker}\nask")

    if data_box:
        bp = ax_box.boxplot(data_box, positions=positions, widths=0.9,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="white", lw=1.5))
        colors_box = []
        for lb in labels_box:
            tk = lb.split("\n")[0]
            side = lb.split("\n")[1]
            colors_box.append(TICKER_COLORS.get(tk, "#aaaaaa") if side == "bid"
                               else STYLE["ask"])
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax_box.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(labels_box, fontsize=7)
    ax_box.set_ylabel("bps from mid-price")
    ax_box.set_title("E. Bid vs Ask Placement Distance\n(per ticker, IQR box, no outliers)", pad=4)
    ax_box.grid(True, alpha=0.3, axis="y")

    # ── Panel F: Fill rate vs hold time scatter (if enough fills) ─────────────
    ax_fill = fig.add_subplot(gs[1, 2])
    for ticker in tickers_present:
        sub = lc_firm[lc_firm.ticker == ticker]
        # bin by hold time and show fill rate
        sub2 = sub[sub.hold_s < sub.hold_s.quantile(0.98)].copy()
        sub2["hold_bin"] = pd.qcut(sub2["hold_s"], q=10, duplicates="drop")
        grp = sub2.groupby("hold_bin")["filled"].mean()
        mids_bin = [(float(str(b).strip("(]").split(",")[0]) +
                     float(str(b).strip("(]").split(",")[1])) / 2
                    for b in grp.index]
        ax_fill.plot(mids_bin, grp.values * 100,
                     "o-", color=TICKER_COLORS[ticker], label=ticker,
                     markersize=4, lw=1.5)
    ax_fill.set_xlabel("Hold time (seconds)")
    ax_fill.set_ylabel("Fill rate (%)")
    ax_fill.set_title("F. Fill Rate vs Order Holding Time\n(decile bins)", pad=4)
    ax_fill.legend(fontsize=7)
    ax_fill.grid(True, alpha=0.3)
    ax_fill.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))

    return fig


def make_pnl_figure(mpid: str, firm_pnl: pd.DataFrame,
                    firm_port: dict) -> plt.Figure:
    """PnL figure — only for firms with fills."""
    tickers_present = sorted(firm_pnl["ticker"].unique())
    n_fills = len(firm_pnl)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    fill_info = "  |  ".join(
        f"{tk}: {(firm_pnl.ticker==tk).sum()} fills"
        for tk in tickers_present
    )
    fig.suptitle(
        f"{fmt_name(mpid)}\nSame-Side PnL Analysis   |   {DATE_RANGE}   |   "
        f"Tickers: {', '.join(tickers_present)}\n"
        f"{fill_info}\n"
        f"PnL benchmark: best same-side price T seconds after fill "
        f"(bid fill vs. best bid at T, ask fill vs. best ask at T)",
        fontsize=9, y=1.0, color="white"
    )

    pnl_cols = [f"pnl_{l}" for l in H_LABELS]

    # ── Panel A: PnL decay curve (mean per fill) ───────────────────────────────
    ax_decay = fig.add_subplot(gs[0, :2])
    for ticker in tickers_present:
        sub = firm_pnl[firm_pnl.ticker == ticker]
        means = [sub[c].mean() if c in sub.columns and sub[c].notna().any()
                 else np.nan for c in pnl_cols]
        ax_decay.plot(range(len(H_LABELS)), means, "o-",
                      color=TICKER_COLORS[ticker], label=ticker,
                      markersize=5, lw=2)
    ax_decay.axhline(0, color="white", lw=1, ls="--", alpha=0.6,
                     label="Break-even")
    ax_decay.set_xticks(range(len(H_LABELS)))
    ax_decay.set_xticklabels(H_LABELS, rotation=40, fontsize=7, ha="right")
    ax_decay.set_xlabel("Time elapsed after fill execution")
    ax_decay.set_ylabel("Mean PnL per fill (USD)\nvs best same-side price at T")
    ax_decay.set_title(
        "A. PnL Decay Curve — Mean PnL per Fill vs Time\n"
        "Positive = fill was at a better price than the market offered T seconds later",
        pad=4
    )
    ax_decay.legend(fontsize=8)
    ax_decay.grid(True, alpha=0.3)

    # ── Panel B: Per-fill PnL box at 4 key horizons ───────────────────────────
    ax_box2 = fig.add_subplot(gs[0, 2])
    key_h = ["100ms", "1s", "30s", "30min"]
    key_c = [f"pnl_{h}" for h in key_h]
    positions2, labels2, data2 = [], [], []
    for i, (h_label, col) in enumerate(zip(key_h, key_c)):
        for j, ticker in enumerate(tickers_present):
            sub = firm_pnl[firm_pnl.ticker == ticker]
            vals = sub[col].dropna().values if col in sub.columns else np.array([])
            if len(vals) < 3:
                continue
            data2.append(vals)
            positions2.append(i * (len(tickers_present) + 0.5) + j)
            labels2.append(f"{h_label}\n{ticker}")

    if data2:
        bp2 = ax_box2.boxplot(data2, positions=positions2, widths=0.7,
                              patch_artist=True, showfliers=False,
                              medianprops=dict(color="white", lw=1.5))
        for patch, lbl in zip(bp2["boxes"], labels2):
            tk = lbl.split("\n")[1]
            patch.set_facecolor(TICKER_COLORS.get(tk, "#aaaaaa"))
            patch.set_alpha(0.7)
    ax_box2.axhline(0, color="white", lw=1, ls="--", alpha=0.6)
    ax_box2.set_xticks(positions2)
    ax_box2.set_xticklabels(labels2, fontsize=6, rotation=30, ha="right")
    ax_box2.set_ylabel("PnL per fill (USD)")
    ax_box2.set_title("B. PnL Distribution at\n4 Key Horizons (IQR)", pad=4)
    ax_box2.grid(True, alpha=0.3, axis="y")

    # ── Panel C: Cumulative PnL over time (4 horizons, best ticker) ───────────
    ax_cum = fig.add_subplot(gs[1, :2])
    show_cols = [("100ms","#00d4ff"), ("1s","#00ff99"),
                 ("30s","#ffaa00"), ("30min","#ff4466")]
    for ticker in tickers_present:
        sub = firm_pnl[firm_pnl.ticker == ticker].sort_values(["date","t_sub"])
        # x-axis: fill index (ordered by time)
        for h_label, color in show_cols:
            col = f"pnl_{h_label}"
            if col not in sub.columns:
                continue
            vals = sub[col].fillna(0).cumsum()
            x = np.arange(len(vals))
            ax_cum.plot(x, vals, "-", color=color, alpha=0.75 if ticker==tickers_present[0] else 0.4,
                        label=f"T={h_label} ({ticker})", lw=1.5)
    ax_cum.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
    ax_cum.set_xlabel("Fill number (ordered by time, Feb 23 → Mar 27 2026)")
    ax_cum.set_ylabel("Cumulative PnL (USD)\nvs best same-side price at T")
    ax_cum.set_title("C. Cumulative Same-Side PnL Over Time\nby Horizon", pad=4)
    ax_cum.legend(fontsize=6, ncol=2)
    ax_cum.grid(True, alpha=0.3)

    # ── Panel D: Portfolio PnL ─────────────────────────────────────────────────
    ax_port = fig.add_subplot(gs[1, 2])
    for ticker, port_df in firm_port.items():
        if port_df.empty:
            continue
        ax_port.plot(range(len(port_df)), port_df["pnl"],
                     "-", color=TICKER_COLORS.get(ticker, "white"),
                     label=f"{ticker} (pos×mid + cash)",
                     lw=1.5)
    ax_port.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
    ax_port.set_xlabel("Trade event number\n(ordered by time)")
    ax_port.set_ylabel("Total PnL (USD)\n= realised cash + unrealised position value")
    ax_port.set_title("D. Portfolio PnL Trajectory\n(realised + unrealised)", pad=4)
    ax_port.legend(fontsize=7)
    ax_port.grid(True, alpha=0.3)

    return fig


# ── Generate and save all figures ─────────────────────────────────────────────
saved_files = []

for mpid in FOCUS_FIRMS:
    lc_firm = lifecycle[lifecycle.firm == mpid]
    if lc_firm.empty:
        continue

    print(f"  {mpid} ({FIRM_NAMES.get(mpid,'?')}): {len(lc_firm):,} orders")

    # Placement figure (all firms)
    try:
        fig_p = make_placement_figure(mpid, lc_firm)
        fname = OUTPUT_DIR / f"firm_{mpid}_placement.png"
        fig_p.savefig(str(fname), dpi=150, bbox_inches="tight",
                      facecolor=fig_p.get_facecolor())
        plt.close(fig_p)
        saved_files.append(fname.name)
        print(f"    Saved {fname.name}")
    except Exception as e:
        print(f"    Placement chart failed for {mpid}: {e}")

    # PnL + portfolio figure (fills only)
    firm_pnl = pnl_df[pnl_df.firm == mpid] if len(pnl_df) else pd.DataFrame()
    firm_port = portfolio_by_firm.get(mpid, {})

    if len(firm_pnl) >= 5:
        try:
            fig_q = make_pnl_figure(mpid, firm_pnl, firm_port)
            fname = OUTPUT_DIR / f"firm_{mpid}_pnl.png"
            fig_q.savefig(str(fname), dpi=150, bbox_inches="tight",
                          facecolor=fig_q.get_facecolor())
            plt.close(fig_q)
            saved_files.append(fname.name)
            print(f"    Saved {fname.name}")
        except Exception as e:
            print(f"    PnL chart failed for {mpid}: {e}")

print(f"\nSaved {len(saved_files)} PNG files:")
for f in saved_files:
    print(f"  {f}")

print("\nDone.")
