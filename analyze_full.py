"""
analyze_full.py  —  Complete per-firm analysis of MPID-attributed LOBSTER orders.

For each of the 7 focus firms (WBPX, MAXM, WCHV, JPMS, UBSS, ETMM, IMCC):
  1. Build order lifecycles (submit -> delete/execute)
  2. Compute PnL at 15 horizons for actual fills (173 total)
     plus mark-to-market PnL at deletion time for ALL lifecycle orders
  3. Deep Claude web research on the firm — what do we know about their
     trading style, strategies, regulatory history, market-making focus areas
  4. Use that research to annotate each order (does this order fit the firm's
     known patterns? event context at submission time?)
  5. Write per-firm CSV  output/firm_{MPID}_orders.csv
  6. Write combined summary statistics  output/firm_summary_stats.csv

Model choices (cost-optimised, target < $50 total):
  - Firm research  : claude-sonnet-4-6  + web_search  (~$1-2 per firm, 7 firms = ~$10-14)
  - Order annotation: claude-haiku-4-5-20251001  (no search, ~$0.001 per firm)
  Total estimated spend: ~$12-16  well under the $50 ceiling.
"""

import os, json, re, time, sys
import pandas as pd
import numpy as np
from pathlib import Path
import anthropic

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\komal\Downloads\_data_dwn_67_491__COST_2025-10-01_2025-10-14_0")
OUTPUT_DIR = Path(r"C:\Users\komal\Project285J\output")
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER      = "COST"
PRICE_SCALE = 0.0001
TICK_USD    = 0.01

# PnL horizons for filled orders (seconds)
PNL_HORIZONS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 300, 1800]
# "EOD" is appended separately

FOCUS_THRESH = 500   # min submits to be a focus firm

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("Set $env:ANTHROPIC_API_KEY before running")
client = anthropic.Anthropic(api_key=api_key)

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_messages(path: Path) -> pd.DataFrame:
    cols  = ["time","type","oid","size","price","dir","firm"]
    df    = pd.read_csv(path, header=None, names=cols, low_memory=False)
    df["type"] = df["type"].map({1:"submit",2:"cancel",3:"delete",
                                  4:"vis_exec",5:"hid_exec",6:"cross",7:"halt"})
    df["dir"]  = df["dir"].map({1:"bid", -1:"ask"})
    return df.dropna(subset=["time","type"])

def build_mid_series(df: pd.DataFrame):
    """Build (times, mids) numpy arrays from execution events as mid-price proxy."""
    execs = df[df.type.isin(["vis_exec","hid_exec"])].copy()
    execs = execs.sort_values("time")
    times  = execs["time"].values.astype(np.float64)
    prices = execs["price"].values.astype(np.float64) * PRICE_SCALE
    return times, prices

def lookup_mid(mp_times, mp_prices, t: float) -> float:
    idx = int(np.searchsorted(mp_times, t, side="right")) - 1
    return float(mp_prices[max(idx, 0)]) if len(mp_prices) else np.nan

def safe_json(raw: str) -> dict | list | None:
    """Extract first valid JSON object or array from raw text."""
    for pat in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        for m in re.finditer(pat, raw):
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load all labeled orders + build lifecycles
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading labeled orders and building lifecycles...")
all_labeled = []
for fpath in sorted(DATA_DIR.glob(f"{TICKER}_*_message_0.csv")):
    date_str = fpath.name.split("_")[1]
    df       = load_messages(fpath)
    labeled  = df[df["firm"].notna() & ~df["firm"].isin(["null",""])].copy()
    labeled["date"]      = date_str
    labeled["price_usd"] = labeled["price"].astype(float) * PRICE_SCALE
    all_labeled.append(labeled)

ldf = pd.concat(all_labeled, ignore_index=True)
print(f"  {len(ldf):,} labeled rows | {ldf['firm'].nunique()} firms")

# build lifecycles
submits   = (ldf[ldf.type=="submit"]
               [["date","firm","oid","time","price_usd","dir","size"]]
               .rename(columns={"time":"t_sub","price_usd":"sub_price","size":"sub_size"})
               .drop_duplicates(subset=["date","oid"]))
terminals = (ldf[ldf.type.isin(["cancel","delete","vis_exec","hid_exec"])]
               [["date","oid","time","type","price_usd"]]
               .rename(columns={"time":"t_term","type":"term_type","price_usd":"term_price"})
               .sort_values("t_term")
               .drop_duplicates(subset=["date","oid"], keep="first"))
life = submits.merge(terminals, on=["date","oid"], how="left")
life["hold_s"]  = life["t_term"] - life["t_sub"]
life["filled"]  = life["term_type"].isin(["vis_exec","hid_exec"])
life = life[life.hold_s >= 0].copy()
print(f"  {len(life):,} lifecycles | {life.filled.sum()} fills")

# identify focus firms
firm_counts = life.groupby("firm").size().sort_values(ascending=False)
focus_firms = firm_counts[firm_counts >= FOCUS_THRESH].index.tolist()
print(f"  Focus firms (>={FOCUS_THRESH} orders): {focus_firms}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Build mid-price series per day, compute PnL for all lifecycle orders
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing PnL across all lifecycle orders...")

events_df = pd.read_csv(OUTPUT_DIR / "events.csv") if (OUTPUT_DIR / "events.csv").exists() else pd.DataFrame()
high_events = events_df[events_df.get("importance","low") == "high"] if len(events_df) else pd.DataFrame()

pnl_parts = []
for date_str, day_life in life[life.firm.isin(focus_firms)].groupby("date"):
    fpath = sorted(DATA_DIR.glob(f"{TICKER}_{date_str}_*_message_0.csv"))
    if not fpath:
        continue
    df_msgs = load_messages(fpath[0])
    mp_t, mp_p = build_mid_series(df_msgs)
    if len(mp_t) == 0:
        continue

    eod_mid = float(mp_p[-1])
    day_events = high_events[high_events.date == date_str] if len(high_events) else pd.DataFrame()

    rows = []
    for rec in day_life.itertuples(index=False):
        t_sub  = rec.t_sub
        p_sub  = rec.sub_price
        d_i    = 1 if rec.dir == "bid" else -1
        size   = rec.sub_size
        mid_sub = lookup_mid(mp_t, mp_p, t_sub)

        row = {
            "date":       rec.date,
            "firm":       rec.firm,
            "order_id":   rec.oid,
            "direction":  rec.dir,
            "size":       size,
            "sub_price":  round(p_sub, 4),
            "t_submit":   round(t_sub, 6),
            "t_terminal": round(rec.t_term, 6) if pd.notna(rec.t_term) else None,
            "term_type":  rec.term_type if pd.notna(rec.term_type) else None,
            "hold_s":     round(rec.hold_s, 6) if pd.notna(rec.hold_s) else None,
            "filled":     bool(rec.filled),
            "mid_at_sub": round(mid_sub, 4) if not np.isnan(mid_sub) else None,
            "price_to_mid_bps": round((p_sub - mid_sub) / (mid_sub + 1e-9) * 10000, 2)
                                 if not np.isnan(mid_sub) else None,
        }

        # PnL at each horizon (all orders — MTM whether filled or not)
        if pd.notna(rec.t_term):
            mid_at_del = lookup_mid(mp_t, mp_p, rec.t_term)
            row["mtm_pnl_at_deletion_usd"] = round(
                d_i * (mid_at_del - p_sub) * size, 4)
        else:
            row["mtm_pnl_at_deletion_usd"] = None

        # Horizon PnL for fills only
        for T in PNL_HORIZONS:
            if rec.filled:
                mid_T = lookup_mid(mp_t, mp_p, t_sub + T)
                row[f"pnl_{T}s"] = round(d_i * (mid_T - p_sub) * size, 4)
            else:
                row[f"pnl_{T}s"] = None

        # EOD PnL (fills only)
        row["pnl_EOD"] = round(d_i * (eod_mid - p_sub) * size, 4) if rec.filled else None

        # Nearest high-importance event at submission time
        if len(day_events):
            ev = day_events.copy()
            ev["dt"] = (ev["time_sec"] - t_sub).abs()
            nearest = ev.loc[ev.dt.idxmin()]
            row["nearest_event"]    = nearest.get("event", "")[:80]
            row["nearest_event_dt_s"] = round(float(nearest["dt"]), 1)
        else:
            row["nearest_event"]    = ""
            row["nearest_event_dt_s"] = None

        rows.append(row)

    pnl_parts.append(pd.DataFrame(rows))
    print(f"  [{date_str}] {len(rows):,} orders processed")

orders_df = pd.concat(pnl_parts, ignore_index=True)
print(f"\n  Total: {len(orders_df):,} orders with PnL computed")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Load or fetch firm identities
# ═══════════════════════════════════════════════════════════════════════════════
id_path = OUTPUT_DIR / "firm_identities.json"
if id_path.exists():
    firm_ids = json.loads(id_path.read_text())
    id_map   = {e["mpid"]: e for e in firm_ids if "mpid" in e}
    print(f"\nLoaded {len(id_map)} firm identities from cache")
else:
    id_map = {}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Deep research + annotation — one Sonnet+web_search call per firm,
#    one Haiku call to annotate the order dataset
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Deep research + annotation per firm")
print("="*60)

research_cache = {}
spend_log = []

# load prior spend log if resuming
spend_log_path = OUTPUT_DIR / "api_spend.csv"
if spend_log_path.exists():
    prior = pd.read_csv(spend_log_path)
    spend_log = prior.to_dict(orient="records")

RESEARCH_SYSTEM = """\
You are a market-microstructure researcher with deep knowledge of US equity
market participants, HFT firms, broker-dealers, and regulatory filings.
Use your web search tool to thoroughly research the given firm.
Return a JSON object (no markdown) covering:
{
  "mpid": "...",
  "firm_name": "...",
  "firm_type": "HFT | market-maker | broker-dealer | prop-trader | unknown",
  "known_strategies": ["...", "..."],
  "regulatory_notes": "any FINRA/SEC actions, notable disclosures",
  "typical_instruments": "equities, ETFs, options, futures, etc.",
  "hq": "city, state",
  "est_order_volume_tier": "top-5 / mid-tier / small",
  "public_info_summary": "3-4 sentence summary of what is publicly known"
}"""

ANNOTATE_SYSTEM = """\
You are a quantitative analyst reviewing order-level data for a specific HFT/market-making firm.
Given firm research and summary statistics, produce a JSON annotation:
{
  "firm": "...",
  "strategy_classification": "...",
  "confidence": "high|medium|low",
  "fill_rate_interpretation": "...",
  "holding_time_interpretation": "...",
  "pnl_interpretation": "...",
  "event_sensitivity": "...",
  "key_findings": ["...", "..."],
  "anomalies": ["..."],
  "overall_assessment": "3-5 sentence narrative"
}
Be specific. Reference the actual numbers provided."""

annotation_results = {}

for firm in focus_firms:
    print(f"\n--- {firm} ---")

    # resume: skip if CSV already written from a prior run
    out_path = OUTPUT_DIR / f"firm_{firm}_orders.csv"
    report_path = OUTPUT_DIR / "labeled_analysis_report.json"
    if out_path.exists():
        if report_path.exists():
            try:
                prior_report = json.loads(report_path.read_text())
                if firm in prior_report:
                    annotation_results[firm] = prior_report[firm]
                    print(f"  Skipping {firm} — already done (CSV exists)")
                    continue
            except Exception:
                pass

    identity = id_map.get(firm, {"mpid": firm, "firm_name": "Unknown"})
    firm_orders = orders_df[orders_df.firm == firm].copy()
    fills       = firm_orders[firm_orders.filled]

    # ── 4a. Deep web research ────────────────────────────────────────────────
    print(f"  Running deep research on {firm} ({identity.get('firm_name','?')})...")
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        system=RESEARCH_SYSTEM,
        messages=[{"role": "user", "content":
            f"Research this firm: NASDAQ MPID={firm}, "
            f"identity hint: {identity.get('firm_name','unknown')} "
            f"({identity.get('firm_type','unknown')}). "
            f"Focus on their equity market-making strategies, COST stock involvement "
            f"if any, regulatory disclosures, and order flow characteristics."}]
    )
    raw_r = "".join(b.text for b in r.content if hasattr(b, "text"))
    research = safe_json(raw_r) or {"mpid": firm, "public_info_summary": raw_r[:300]}
    research_cache[firm] = research
    u_r = r.usage
    cost_r = u_r.input_tokens * 3/1e6 + u_r.output_tokens * 15/1e6
    spend_log.append({"firm": firm, "step": "research",
                      "input_tok": u_r.input_tokens, "output_tok": u_r.output_tokens,
                      "cost_usd": round(cost_r, 4)})
    print(f"  Research done: {research.get('firm_type','?')} | "
          f"in={u_r.input_tokens:,} out={u_r.output_tokens} cost=${cost_r:.3f}")
    print(f"  Summary: {str(research.get('public_info_summary',''))[:120]}")

    # ── 4b. Build summary stats for this firm ────────────────────────────────
    pnl_cols  = [c for c in firm_orders.columns if c.startswith("pnl_")]
    fill_stats = {}
    if len(fills):
        for col in pnl_cols:
            s = fills[col].dropna()
            if len(s):
                fill_stats[col] = {"mean": round(s.mean(),4), "median": round(s.median(),4),
                                   "std": round(s.std(),4), "n": len(s)}

    mtm_s = firm_orders["mtm_pnl_at_deletion_usd"].dropna()
    summary_stats = {
        "total_orders": len(firm_orders),
        "total_fills": int(firm_orders.filled.sum()),
        "fill_rate": round(firm_orders.filled.mean(), 5),
        "avg_hold_s": round(firm_orders.hold_s.mean(), 3),
        "med_hold_s": round(firm_orders.hold_s.median(), 3),
        "avg_size": round(firm_orders["size"].mean(), 1),
        "avg_price_to_mid_bps": round(firm_orders.price_to_mid_bps.mean(), 3),
        "bid_frac": round((firm_orders.direction=="bid").mean(), 3),
        "mtm_pnl_mean_usd": round(mtm_s.mean(), 4) if len(mtm_s) else None,
        "mtm_pnl_median_usd": round(mtm_s.median(), 4) if len(mtm_s) else None,
        "fill_pnl_by_horizon": fill_stats,
    }

    # ── 4c. Haiku annotation call ────────────────────────────────────────────
    print(f"  Annotating with Haiku...")
    a = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1200,
        system=[{"type":"text","text":ANNOTATE_SYSTEM,
                 "cache_control":{"type":"ephemeral"}}],
        messages=[{"role":"user","content":
            f"Firm: {firm}\n"
            f"Research findings:\n{json.dumps(research, indent=2, default=str)[:2000]}\n\n"
            f"Order statistics:\n{json.dumps(summary_stats, indent=2, default=str)}\n\n"
            f"Known events during analysis period (Oct 1-14 2025):\n"
            f"- US government shutdown (all BLS data suspended)\n"
            f"- Oct 3 10:00: ISM Services PMI miss (50.0 vs 52 consensus)\n"
            f"- Oct 8 14:00: FOMC September minutes release\n"
            f"- Oct 8 16:15: COST September sales beat (+8% YoY, after close)\n"
            f"- Oct 9: S&P 500 new ATH; 94.6% FOMC cut probability\n"
            f"- Oct 13 09:30: S&P -2.7%, VIX +31.8%, Trump 100% China tariff\n"
            f"- Oct 14: Goldman/Wells/Citi earnings beat; Powell NABE speech 12:20 PM"}]
    )
    raw_a = "".join(b.text for b in a.content if hasattr(b, "text"))
    annotation = safe_json(raw_a) or {"firm": firm, "overall_assessment": raw_a[:500]}
    annotation_results[firm] = {"research": research, "stats": summary_stats,
                                  "annotation": annotation}
    u_a = a.usage
    cost_a = u_a.input_tokens * 0.8/1e6 + u_a.output_tokens * 4/1e6
    spend_log.append({"firm": firm, "step": "annotation",
                      "input_tok": u_a.input_tokens, "output_tok": u_a.output_tokens,
                      "cost_usd": round(cost_a, 5)})
    print(f"  Annotation: {annotation.get('strategy_classification','?')} "
          f"({annotation.get('confidence','?')}) cost=${cost_a:.4f}")
    print(f"  Key: {annotation.get('key_findings',[''])[:2]}")

    # ── 4d. Add research fields to orders and write per-firm CSV ─────────────
    firm_orders["firm_name"]        = research.get("firm_name", identity.get("firm_name","?"))
    firm_orders["firm_type"]        = research.get("firm_type", "unknown")
    firm_orders["known_strategies"] = str(research.get("known_strategies", []))[:200]
    firm_orders["strategy_class"]   = annotation.get("strategy_classification","?")
    firm_orders["annotation_conf"]  = annotation.get("confidence","?")

    out_path = OUTPUT_DIR / f"firm_{firm}_orders.csv"
    firm_orders.to_csv(out_path, index=False)
    print(f"  Saved {out_path.name} ({len(firm_orders):,} rows)")

    # save report incrementally so a later crash doesn't lose prior firms
    with open(OUTPUT_DIR / "labeled_analysis_report.json", "w") as f:
        json.dump(annotation_results, f, indent=2, default=str)
    pd.DataFrame(spend_log).to_csv(OUTPUT_DIR / "api_spend.csv", index=False)

    time.sleep(0.3)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Summary statistics CSV (one row per firm)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Building summary statistics CSV...")
summary_rows = []
for firm in focus_firms:
    info  = annotation_results.get(firm, {})
    stats = info.get("stats", {})
    ann   = info.get("annotation", {})
    res   = info.get("research", {})

    row = {
        "firm": firm,
        "firm_name": res.get("firm_name","?"),
        "firm_type": res.get("firm_type","?"),
        "strategy_classification": ann.get("strategy_classification","?"),
        "confidence": ann.get("confidence","?"),
        **{k: v for k, v in stats.items() if not isinstance(v, dict)},
        "key_findings": str(ann.get("key_findings",[])),
        "overall_assessment": str(ann.get("overall_assessment",""))[:500],
        "regulatory_notes": res.get("regulatory_notes","")[:200],
    }
    # flatten fill PnL means for key horizons
    fill_pnl = stats.get("fill_pnl_by_horizon", {})
    for T in [0.01, 0.1, 1, 10, 60, 300]:
        col = f"pnl_{T}s"
        row[f"fill_pnl_mean_{T}s"] = fill_pnl.get(col, {}).get("mean") if col in fill_pnl else None
    row["fill_pnl_mean_EOD"] = fill_pnl.get("pnl_EOD", {}).get("mean") if "pnl_EOD" in fill_pnl else None
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "firm_summary_stats.csv", index=False)
print(f"Saved firm_summary_stats.csv")
print(summary_df[["firm","firm_name","strategy_classification","confidence",
                   "total_orders","total_fills","fill_rate",
                   "avg_hold_s","mtm_pnl_mean_usd"]].to_string(index=False))

# ── Spend summary ──────────────────────────────────────────────────────────────
spend_df = pd.DataFrame(spend_log)
total_spend = spend_df.cost_usd.sum()
spend_df.to_csv(OUTPUT_DIR / "api_spend.csv", index=False)
print(f"\nTotal API spend: ${total_spend:.4f}  (budget: $50.00)")
if total_spend > 50:
    print("WARNING: Budget exceeded!")
else:
    print(f"Budget remaining: ${50 - total_spend:.2f}")

# ── Save full annotation report ────────────────────────────────────────────────
with open(OUTPUT_DIR / "labeled_analysis_report.json", "w") as f:
    json.dump(annotation_results, f, indent=2, default=str)
print("\nSaved labeled_analysis_report.json")
print("\nDone.")
