"""
fetch_events.py — Use Claude API + web search to find real market events
for each COST trading day Oct 1-14 2025, then save as events.json and
events.csv for use in step33.py.

Model: claude-sonnet-4-6 (needed for web search tool)
Expected cost: ~$0.05-0.10 (one call per day, ~2k tokens each)
"""

import os, json, time
import pandas as pd
from pathlib import Path
import anthropic

OUTPUT_DIR = Path(r"C:\Users\komal\Project285J\output")
OUTPUT_DIR.mkdir(exist_ok=True)

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("Set $env:ANTHROPIC_API_KEY before running")

client = anthropic.Anthropic(api_key=api_key)

TRADING_DAYS = [
    "2025-10-01", "2025-10-02", "2025-10-03",
    "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-10-10",
    "2025-10-13", "2025-10-14",
]

SYSTEM = """\
You are a financial market research assistant. When asked about market events on a specific date,
use your web search tool to look up what actually happened. Be thorough and accurate.
Always return results as a JSON array — no prose, no markdown fences, just the raw JSON array."""

def fetch_events_for_day(date: str) -> list[dict]:
    """Call Claude with web search to get market events for one trading day."""
    prompt = f"""Search for ALL significant US market events on {date} (2025).

I need every event that could move equity prices, especially COST (Costco Wholesale).

Search for:
1. US economic data releases: NFP, CPI, PPI, Core PCE, Retail Sales, ISM Manufacturing/Services,
   JOLTS, ADP employment, Initial Jobless Claims, Consumer Confidence, Housing data, GDP revisions
2. Federal Reserve: FOMC meeting decisions, minutes releases, Fed chair/governor speeches,
   rate decisions, balance sheet updates
3. Treasury: major auction results, yield curve moves
4. COST-specific: earnings releases, guidance updates, analyst rating/price target changes,
   insider filings, index inclusion/exclusion
5. Broad market: S&P 500 daily move, VIX level, any crash/rally >1%, major sector moves,
   geopolitical events affecting markets

For each event return a JSON object with these exact keys:
  "date": "{date}"
  "time_et": "HH:MM" (exact scheduled time, Eastern Time — e.g. "08:30" for 8:30 AM)
  "time_sec": integer seconds after midnight ET  (08:30 = 30600, 14:00 = 50400, 09:30 = 34200)
  "event": short name (e.g. "NFP September 2025")
  "description": what happened, actual number vs consensus if applicable
  "beat_miss": "beat" | "miss" | "in-line" | "N/A"
  "market_direction": "up" | "down" | "mixed" | "unknown"
  "category": "macro" | "fed" | "costco" | "equity_market" | "rates"
  "importance": "high" | "medium" | "low"

Return ONLY a JSON array. If nothing significant happened, return an empty array []."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )

    # extract the final text block (after any tool use)
    raw_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw_text = block.text.strip()

    # extract JSON array from anywhere in the response (handles markdown fences + preamble)
    import re
    match = re.search(r'\[[\s\S]*\]', raw_text)
    if match:
        try:
            events = json.loads(match.group())
            if not isinstance(events, list):
                events = [events]
        except json.JSONDecodeError as e:
            print(f"  [warn] JSON parse error for {date}: {e}\n  raw snippet: {match.group()[:200]}")
            events = []
    else:
        print(f"  [warn] no JSON array found for {date}, raw:\n{raw_text[:300]}")
        events = []

    input_tok  = response.usage.input_tokens
    output_tok = response.usage.output_tokens
    print(f"  tokens: in={input_tok} out={output_tok}  events found: {len(events)}")
    return events


# ── Main loop ──────────────────────────────────────────────────────────────────
all_events = []
total_cost_usd = 0.0

# claude-sonnet-4-6 pricing (approximate, per MTok)
INPUT_PRICE  = 3.00 / 1_000_000
OUTPUT_PRICE = 15.00 / 1_000_000

for date in TRADING_DAYS:
    print(f"\n[{date}] fetching events...")
    try:
        events = fetch_events_for_day(date)
        all_events.extend(events)
        time.sleep(0.5)   # avoid rate limiting
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nTotal events collected: {len(all_events)}")

# ── Save raw JSON ──────────────────────────────────────────────────────────────
with open(OUTPUT_DIR / "events.json", "w") as f:
    json.dump(all_events, f, indent=2)
print(f"Saved output/events.json")

# ── Save structured CSV ────────────────────────────────────────────────────────
if all_events:
    ev_df = pd.DataFrame(all_events)

    # ensure time_sec is numeric
    if "time_sec" in ev_df.columns:
        ev_df["time_sec"] = pd.to_numeric(ev_df["time_sec"], errors="coerce")

    ev_df = ev_df.sort_values(["date", "time_sec"]).reset_index(drop=True)
    ev_df.to_csv(OUTPUT_DIR / "events.csv", index=False)
    print(f"Saved output/events.csv  ({len(ev_df)} rows)")

    # Print summary
    print("\n=== Events by day ===")
    for date, grp in ev_df.groupby("date"):
        print(f"\n  {date}:")
        for _, row in grp.iterrows():
            imp = row.get("importance","?")
            cat = row.get("category","?")
            bm  = row.get("beat_miss","?")
            t   = row.get("time_et","??:??")
            print(f"    {t}  [{imp}/{cat}]  {row.get('event','')}  ({bm})")
            print(f"         {row.get('description','')[:120]}")

print("\nDone.")
