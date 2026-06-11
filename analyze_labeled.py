"""
analyze_labeled.py

Focuses exclusively on the firm-attributed orders in the LOBSTER 7th column.
For each identified firm (MPID):
  1. Build full order lifecycle: submit -> delete/execute, compute holding time & PnL
  2. Compute per-firm-per-day signature metrics
  3. Bin activity into 1-minute windows to detect event-driven spikes
  4. Use Claude API (sonnet + web_search) to:
       a. Look up what each MPID firm is (HFT, broker-dealer, market maker, etc.)
       b. Interpret each firm's behavioral signature against the event timeline
       c. Generate a structured per-firm strategy report
"""

import os, json, re, time, sys
import pandas as pd
import numpy as np
from pathlib import Path
import anthropic

SKIP_IDENTITY = "--skip-identity" in sys.argv

DATA_DIR   = Path(r"C:\Users\komal\Downloads\_data_dwn_67_491__COST_2025-10-01_2025-10-14_0")
OUTPUT_DIR = Path(r"C:\Users\komal\Project285J\output")
OUTPUT_DIR.mkdir(exist_ok=True)

PRICE_SCALE = 0.0001
TICK_USD    = 0.01

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("Set $env:ANTHROPIC_API_KEY before running")
client = anthropic.Anthropic(api_key=api_key)

# ── Load known events ──────────────────────────────────────────────────────────
events_path = OUTPUT_DIR / "events.json"
known_events = json.loads(events_path.read_text()) if events_path.exists() else []
# Add manually confirmed events from web search output
known_events += [
    {"date":"2025-10-01","time_et":"00:00","time_sec":0,
     "event":"US Government Shutdown begins","category":"macro","importance":"high",
     "description":"All BLS/Census data releases suspended (NFP, CPI, PPI, Retail Sales, JOLTS)."},
    {"date":"2025-10-03","time_et":"08:30","time_sec":30600,
     "event":"NFP Sep-2025 CANCELLED - Government Shutdown","category":"macro","importance":"high",
     "description":"Bureau of Labor Statistics did not publish September employment report."},
    {"date":"2025-10-08","time_et":"16:15","time_sec":58500,
     "event":"COST September 2025 Sales - Beat","category":"costco","importance":"high",
     "description":"Net sales $26.58B, +8.0% YoY. U.S. comps +5.1%. Digital +26.1%. Above expectations."},
    {"date":"2025-10-10","time_et":"10:57","time_sec":39420,
     "event":"Trump tariff threat - China","category":"macro","importance":"high",
     "description":"Truth Social post threatening 'massive' tariff hike on China. Risk-off spike."},
    {"date":"2025-10-13","time_et":"09:30","time_sec":34200,
     "event":"S&P 500 Major Selloff","category":"equity_market","importance":"high",
     "description":"S&P 500 significant decline at open."},
    {"date":"2025-10-14","time_et":"06:45","time_sec":24300,
     "event":"JPMorgan Q3 2025 Earnings - Pre-market","category":"equity_market","importance":"high",
     "description":"JPMorgan Chase Q3 earnings release before market open."},
]
events_by_date = {}
for e in known_events:
    events_by_date.setdefault(e["date"], []).append(e)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load all labeled orders from all files
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading all labeled orders...")
all_labeled = []

for fpath in sorted(DATA_DIR.glob("COST_*_message_0.csv")):
    date_str = fpath.name.split("_")[1]
    df = pd.read_csv(fpath, header=None,
                     names=["time","type","oid","size","price","dir","firm"],
                     low_memory=False)
    labeled = df[df["firm"].notna() & ~df["firm"].isin(["null",""])].copy()
    labeled["date"]  = date_str
    labeled["price_usd"] = labeled["price"].astype(float) * PRICE_SCALE
    labeled["dir_str"]   = labeled["dir"].map({1:"bid", -1:"ask"})
    all_labeled.append(labeled)
    print(f"  [{date_str}] {len(labeled):,} labeled orders from {labeled['firm'].nunique()} firms")

ldf = pd.concat(all_labeled, ignore_index=True)
ldf["type"] = ldf["type"].map({1:"submit",2:"cancel",3:"delete",
                                4:"vis_exec",5:"hid_exec"}).fillna(ldf["type"].astype(str))
print(f"\nTotal: {len(ldf):,} labeled orders | {ldf['firm'].nunique()} unique firms")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Build order lifecycles: match submit -> first terminal event per order_id
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding order lifecycles...")
submits   = ldf[ldf.type=="submit"][["date","firm","oid","time","price_usd","dir_str","size"]].copy()
submits   = submits.rename(columns={"time":"t_submit","price_usd":"sub_price","size":"sub_size"})

terminals = ldf[ldf.type.isin(["cancel","delete","vis_exec","hid_exec"])].copy()
terminals = (terminals.sort_values("time")
                      .drop_duplicates(subset=["date","oid"], keep="first")
                      [["date","oid","time","type","price_usd"]]
                      .rename(columns={"time":"t_terminal","type":"terminal_type",
                                       "price_usd":"terminal_price"}))

lifecycle = submits.merge(terminals, on=["date","oid"], how="left")
lifecycle["holding_s"] = lifecycle["t_terminal"] - lifecycle["t_submit"]
lifecycle["filled"]    = lifecycle["terminal_type"].isin(["vis_exec","hid_exec"])
lifecycle = lifecycle[lifecycle.holding_s >= 0]

lifecycle.to_csv(OUTPUT_DIR / "labeled_lifecycles.csv", index=False)
print(f"  {len(lifecycle):,} order lifecycles saved")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Per-firm-per-day signature metrics
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing per-firm metrics...")
metrics_rows = []

for (date_str, firm), grp in lifecycle.groupby(["date","firm"]):
    n_sub    = len(grp)
    n_filled = grp.filled.sum()
    fill_rt  = n_filled / n_sub if n_sub else 0

    hold     = grp.holding_s.dropna()
    avg_hold = hold.mean() if len(hold) else np.nan
    med_hold = hold.median() if len(hold) else np.nan

    # price aggressiveness: mean |sub_price - mid| ... approx via spread implied by bid/ask pairs
    bids = grp[grp.dir_str=="bid"]["sub_price"]
    asks = grp[grp.dir_str=="ask"]["sub_price"]
    bid_ask_ratio = len(bids) / len(asks) if len(asks) else np.nan

    # 1-minute burst detection: does this firm cluster orders in time?
    times_sub = grp.t_submit.values
    if len(times_sub) > 1:
        iat = np.diff(np.sort(times_sub))   # inter-arrival times
        med_iat = float(np.median(iat))
        burst   = float((iat < 0.001).mean())  # fraction within 1ms of previous order
    else:
        med_iat, burst = np.nan, 0.0

    metrics_rows.append({
        "date": date_str, "firm": firm,
        "n_submits": n_sub, "n_filled": int(n_filled),
        "fill_rate": round(fill_rt, 4),
        "avg_holding_s": round(avg_hold, 4) if not np.isnan(avg_hold) else None,
        "med_holding_s": round(med_hold, 4) if not np.isnan(med_hold) else None,
        "bid_ask_ratio": round(bid_ask_ratio, 3) if not np.isnan(bid_ask_ratio) else None,
        "med_iat_s":     round(med_iat, 6) if not np.isnan(med_iat) else None,
        "burst_frac":    round(burst, 4),
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUTPUT_DIR / "labeled_firm_metrics.csv", index=False)

# ── Aggregate across days ──────────────────────────────────────────────────────
agg = (metrics_df.groupby("firm")
                 .agg(total_submits=("n_submits","sum"),
                      total_fills=("n_filled","sum"),
                      mean_fill_rate=("fill_rate","mean"),
                      mean_hold_s=("avg_holding_s","mean"),
                      med_hold_s=("med_holding_s","median"),
                      mean_bid_ask_ratio=("bid_ask_ratio","mean"),
                      mean_iat_s=("med_iat_s","mean"),
                      mean_burst=("burst_frac","mean"))
                 .round(4)
                 .sort_values("total_submits", ascending=False))

agg.to_csv(OUTPUT_DIR / "labeled_firm_agg.csv")
print(agg.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 1-minute activity bins per firm to detect event reactions
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding 1-minute activity time series per firm...")
focus_firms = agg[agg.total_submits >= 500].index.tolist()
print(f"  Focus firms (>=500 submits): {focus_firms}")

submits_only = ldf[ldf.type == "submit"].copy()
submits_only["minute_bin"] = (submits_only["time"] // 60).astype(int)

activity = (submits_only[submits_only.firm.isin(focus_firms)]
              .groupby(["date","firm","minute_bin"])["size"]
              .agg(n_orders="count", total_size="sum")
              .reset_index())

activity.to_csv(OUTPUT_DIR / "labeled_activity_bins.csv", index=False)
print(f"  Saved {len(activity):,} (date, firm, minute) bins")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Claude API analysis — firm identity lookup + behavioral interpretation
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_FIRM_LOOKUP = """\
You are a financial markets expert with deep knowledge of US equity market participants,
FINRA MPID codes, broker-dealers, HFT firms, and market makers.
When given MPID codes (4-letter Market Participant Identifiers used on NASDAQ),
use your web search tool to identify exactly what firm each code belongs to,
their business model (HFT, market maker, broker-dealer, prop trading, etc.),
and any public information about their trading strategies.
Return results as a JSON array only — no markdown, no prose."""

SYSTEM_BEHAVIOR = """\
You are a quantitative finance researcher specializing in market microstructure
and HFT strategy analysis. You have been given:
  1. Behavioral metrics for a specific firm's COST (Costco) orders over 11 trading days
  2. A timeline of known market events during that period
  3. 1-minute order-submission activity bins showing when the firm was most active

Your job is to:
  - Identify whether the firm's activity spikes correlate with specific market events
  - Classify their strategy: market-making, latency arbitrage, directional/informed, stat-arb
  - Describe any distinctive behavioral patterns (e.g., always submits at open,
    goes silent after news, alternates bid/ask rapidly)
  - Assess the strength of evidence for each conclusion

Be specific and data-driven. Output JSON only:
{
  "firm": "...",
  "identity": "...",
  "strategy_classification": "...",
  "confidence": "high|medium|low",
  "event_reactions": [{"event": "...", "reaction": "...", "evidence": "..."}],
  "key_patterns": ["...", "..."],
  "interpretation": "3-5 sentence narrative"
}"""

# ── Step 5a: look up firm identities — one API call per MPID ──────────────────
firm_identities = []
id_map = {}

if SKIP_IDENTITY and (OUTPUT_DIR / "firm_identities.json").exists():
    firm_identities = json.loads((OUTPUT_DIR / "firm_identities.json").read_text())
    id_map = {e["mpid"]: e for e in firm_identities if "mpid" in e}
    print(f"\nReusing cached firm identities ({len(firm_identities)} firms):")
    for fi in firm_identities:
        print(f"  {fi.get('mpid','?'):6s}  {fi.get('firm_name','?')}")
else:
    print("\nLooking up firm identities via Claude API + web search (one call per MPID)...")

for mpid in focus_firms:
    id_resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        system=SYSTEM_FIRM_LOOKUP,
        messages=[{"role": "user", "content":
            f'Search for NASDAQ MPID code "{mpid}" and return a single JSON object:\n'
            f'{{"mpid":"{mpid}","firm_name":"...","firm_type":"...","hq_location":"...","brief_description":"..."}}\n'
            f'Return ONLY that JSON object, nothing else.'}]
    )
    raw = "".join(b.text for b in id_resp.content if hasattr(b, "text")).strip()
    u = id_resp.usage
    # robustly extract the first {...} from the response
    obj = None
    for m in re.finditer(r'\{[^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(m.group())
            break
        except json.JSONDecodeError:
            continue
    if obj is None:
        # fall back: store raw text so at least firm name is preserved
        obj = {"mpid": mpid, "firm_name": raw[:120], "firm_type": "unknown",
               "hq_location": "unknown", "brief_description": raw[:120]}
    firm_identities.append(obj)
    id_map[mpid] = obj
    print(f"  {mpid}: {obj.get('firm_name','?')} [{obj.get('firm_type','?')}]  "
          f"(in={u.input_tokens} out={u.output_tokens})")
    time.sleep(0.2)

    with open(OUTPUT_DIR / "firm_identities.json", "w") as f:
        json.dump(firm_identities, f, indent=2)
    print(f"  Saved firm_identities.json ({len(firm_identities)} firms)")

# ── Step 5b: per-firm behavioral analysis ─────────────────────────────────────
print("\nRunning per-firm behavioral analysis...")
firm_reports = []
total_in, total_out = 0, 0

for firm in focus_firms:
    print(f"  Analyzing {firm}...")
    identity_info = id_map.get(firm, {})

    # metrics for this firm
    firm_metrics = agg.loc[firm].to_dict() if firm in agg.index else {}

    # per-day activity bins with event annotation
    firm_bins = activity[(activity.firm == firm)].copy()
    firm_bins["time_et"] = firm_bins["minute_bin"].apply(
        lambda m: f"{(m*60)//3600:02d}:{((m*60)%3600)//60:02d}")
    bins_summary = (firm_bins.groupby(["date","time_et"])["n_orders"]
                             .sum()
                             .reset_index()
                             .sort_values("n_orders", ascending=False)
                             .head(20)
                             .to_dict(orient="records"))

    # events for relevant dates
    relevant_events = []
    for d in firm_bins.date.unique():
        relevant_events.extend(events_by_date.get(d, []))
    relevant_events = sorted(relevant_events, key=lambda x: (x["date"], x.get("time_sec",0)))

    payload = {
        "firm_mpid": firm,
        "identity": identity_info,
        "aggregate_metrics": firm_metrics,
        "top_20_active_minute_bins": bins_summary,
        "known_events_in_period": relevant_events[:15],
    }

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=[{"type": "text", "text": SYSTEM_BEHAVIOR,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content":
            f"Analyze this firm's COST order behavior and classify their strategy.\n\n"
            f"{json.dumps(payload, indent=2, default=str)}"}]
    )

    raw = "".join(b.text for b in response.content if hasattr(b, "text")).strip()

    # robust extraction: try outermost {...}, fall back to partial parse
    report = None
    for m in re.finditer(r'\{[\s\S]*\}', raw):
        try:
            report = json.loads(m.group())
            break
        except json.JSONDecodeError:
            continue
    if report is None:
        report = {"firm": firm, "strategy_classification": "parse_error",
                  "confidence": "low", "key_patterns": [], "interpretation": raw[:500]}
    firm_reports.append(report)

    u = response.usage
    total_in  += u.input_tokens
    total_out += u.output_tokens
    print(f"    {firm}: {report.get('strategy_classification','?')} "
          f"(confidence={report.get('confidence','?')}) "
          f"tokens in={u.input_tokens} out={u.output_tokens} "
          f"cache_read={getattr(u,'cache_read_input_tokens',0)}")

    time.sleep(0.3)

print(f"\nTotal API usage: input={total_in:,} output={total_out:,}")

# ── Save all reports ───────────────────────────────────────────────────────────
final_output = {
    "firm_identities": firm_identities,
    "firm_reports": firm_reports,
    "known_events": known_events,
    "aggregate_metrics": agg.reset_index().to_dict(orient="records"),
}
with open(OUTPUT_DIR / "labeled_analysis_report.json", "w") as f:
    json.dump(final_output, f, indent=2, default=str)

print("\n=== STRATEGY SUMMARY ===")
for r in firm_reports:
    print(f"\n{r.get('firm','?')} -- {r.get('identity','?')}")
    print(f"  Strategy : {r.get('strategy_classification','?')} ({r.get('confidence','?')})")
    print(f"  Patterns : {r.get('key_patterns', [])}")
    print(f"  Narrative: {r.get('interpretation','')[:300]}")

print("\nDone. All outputs in output/")
