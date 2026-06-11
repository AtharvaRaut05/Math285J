"""
Step 3.3 — News-Event Reaction Analysis

For each known macro event e_j at timestamp tau_j:
  1. Pre-event OFI in [tau - 60s, tau)      — positioning signal
  2. Reaction latency: first execution after tau_j
  3. Post-event directional accuracy in [tau, tau + 30s]
  4. Claude Haiku API interpretation (prompt-cached system prompt)

API config: claude-haiku-4-5-20251001 with prompt caching.
Set ANTHROPIC_API_KEY env var before running.
Expected cost: < $0.005 per run (< 2000 input tokens total).
"""

import os, json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR   = Path(r"C:\Users\komal\Downloads\_data_dwn_67_491__COST_2025-10-01_2025-10-14_0")
OUTPUT_DIR = Path(r"C:\Users\komal\Project285J\output")
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER        = "COST"
PRICE_SCALING = 0.0001
PRE_WINDOW    = 60   # seconds
POST_WINDOW   = 30   # seconds
EXEC_TYPES    = {"vis_exec", "hid_exec"}

# Known macro events during Oct 1–14 2025 (ET, seconds after midnight)
# Sources: BLS release calendar (NFP/CPI), Federal Reserve press release schedule (FOMC)
MACRO_EVENTS = [
    {"date": "2025-10-03", "name": "NFP Sep-2025",        "tau": 30600},  # 08:30 AM
    {"date": "2025-10-08", "name": "FOMC Minutes Sep-2025","tau": 50400},  # 02:00 PM
    {"date": "2025-10-10", "name": "CPI Sep-2025",         "tau": 30600},  # 08:30 AM
    {"date": "2025-10-14", "name": "Retail Sales Sep-2025","tau": 30600},  # 08:30 AM
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_messages(date_str: str) -> pd.DataFrame | None:
    """Load lowest-start-time file for the date (prefers pre-market over regular)."""
    files = sorted(DATA_DIR.glob(f"{TICKER}_{date_str}_*_message_0.csv"))
    if not files:
        return None
    columns  = ["Time","Type","OrderID","Size","Price","Direction"]
    dtype_map = {"Time":float,"Type":"Int64","OrderID":"Int64",
                 "Size":"Int64","Price":"Int64","Direction":"Int64"}
    df = pd.read_csv(files[0], header=None, names=columns, usecols=range(6),
                     dtype=dtype_map, na_values=["","NA"], low_memory=False)
    df["Type"]      = df["Type"].map({1:"submit",2:"cancel",3:"delete",
                                      4:"vis_exec",5:"hid_exec",6:"cross",7:"halt"})
    df["Direction"] = df["Direction"].map({-1:"ask", 1:"bid"})
    return df.dropna(subset=["Time","Type","Direction"])


def compute_ofi(window: pd.DataFrame) -> dict:
    """Size-based OFI — no LOB state required, computed from raw message counts."""
    bs = window[(window.Type=="submit")  & (window.Direction=="bid")]["Size"].sum()
    as_ = window[(window.Type=="submit") & (window.Direction=="ask")]["Size"].sum()
    bd = window[window.Type.isin({"cancel","delete"}) & (window.Direction=="bid")]["Size"].sum()
    ad = window[window.Type.isin({"cancel","delete"}) & (window.Direction=="ask")]["Size"].sum()
    ofi = int((bs - bd) - (as_ - ad))
    return {"ofi": ofi,
            "net": "bullish" if ofi > 0 else ("bearish" if ofi < 0 else "neutral"),
            "bid_sub": int(bs), "ask_sub": int(as_),
            "bid_del": int(bd), "ask_del": int(ad)}


def compute_reaction(post: pd.DataFrame, tau: float) -> dict:
    """Post-event reaction: latency, dominant direction, directional accuracy."""
    execs = post[post.Type.isin(EXEC_TYPES)].copy()
    if execs.empty:
        return {"n_execs":0, "latency_ms":None, "dominant":None,
                "correct_direction":None, "price_move_bps":None}

    latency_ms = round((execs.iloc[0].Time - tau) * 1000, 2)

    bid_vol = int(execs[execs.Direction=="bid"]["Size"].sum())
    ask_vol = int(execs[execs.Direction=="ask"]["Size"].sum())
    dominant = "bid" if bid_vol >= ask_vol else "ask"

    p0 = float(execs.iloc[0].Price)  * PRICE_SCALING
    p1 = float(execs.iloc[-1].Price) * PRICE_SCALING
    move_bps = round((p1 - p0) / p0 * 10000, 4) if p0 > 0 else 0.0

    correct = (dominant == "bid" and move_bps > 0) or (dominant == "ask" and move_bps < 0)
    return {"n_execs": len(execs), "latency_ms": latency_ms,
            "dominant": dominant, "correct_direction": correct, "price_move_bps": move_bps}


# ── Main analysis ──────────────────────────────────────────────────────────────

results = []

for ev in MACRO_EVENTS:
    date_str, name, tau = ev["date"], ev["name"], ev["tau"]
    print(f"\n[{name}] {date_str}  tau={tau}s ({tau//3600:02d}:{(tau%3600)//60:02d} ET)")

    df = load_messages(date_str)
    if df is None:
        print("  skip: no file")
        continue

    t_min, t_max = df.Time.min(), df.Time.max()
    if not (t_min <= tau <= t_max):
        print(f"  skip: tau outside file range [{t_min:.0f}, {t_max:.0f}]")
        continue

    pre  = df[(df.Time >= tau - PRE_WINDOW) & (df.Time <  tau)]
    post = df[(df.Time >= tau)              & (df.Time <= tau + POST_WINDOW)]

    ofi      = compute_ofi(pre)
    reaction = compute_reaction(post, tau)

    row = {"event": name, "date": date_str, "tau_sec": tau,
           "pre_ofi": ofi["ofi"], "pre_net": ofi["net"],
           "pre_bid_sub": ofi["bid_sub"], "pre_ask_sub": ofi["ask_sub"],
           **{f"post_{k}": v for k, v in reaction.items()}}
    results.append(row)

    print(f"  Pre-event OFI: {ofi['ofi']:+,} ({ofi['net']})")
    if reaction["n_execs"]:
        print(f"  Post-event: n={reaction['n_execs']}  latency={reaction['latency_ms']} ms  "
              f"dominant={reaction['dominant']}  move={reaction['price_move_bps']:+.2f} bps  "
              f"correct={reaction['correct_direction']}")
    else:
        print("  Post-event: no executions in window")

# ── Save CSV ──────────────────────────────────────────────────────────────────
ev_df = pd.DataFrame(results)
ev_df.to_csv(OUTPUT_DIR / "step33_event_reactions.csv", index=False)
print(f"\nSaved step33_event_reactions.csv  ({len(ev_df)} events)")

# ── Claude API interpretation ──────────────────────────────────────────────────
# Model : claude-haiku-4-5-20251001 (cheapest Claude model, $0.80/MTok input)
# Caching: system prompt marked ephemeral → $0.08/MTok on cache hits (90% cheaper)
# One API call total, < 2000 tokens → cost < $0.002 per run

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("\nANTHROPIC_API_KEY not set — skipping Claude interpretation.")
    print("Set it with:  $env:ANTHROPIC_API_KEY='sk-ant-...'")
else:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    SYSTEM = """\
You are a market-microstructure expert interpreting HFT order flow around macro news events.

Classify pre-event positioning and post-event reaction against these archetypes:
- LATENCY ARB: <1 ms latency, no pre-positioning, high directional accuracy
- INFORMED:    1–100 ms latency, bullish/bearish pre-OFI, high directional accuracy
- MOMENTUM:    >100 ms latency, post-event following, moderate accuracy
- MARKET MAKE: flat OFI, random direction, very fast or irrelevant latency

For each event output JSON:
{ "event": "...", "archetype": "...", "latency_class": "co-location/<1ms | direct-feed/1-10ms | slow/>100ms",
  "pre_positioning": "yes/no/ambiguous", "interpretation": "1-2 sentences" }
Return a JSON array."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=700,
        system=[{"type":"text", "text": SYSTEM,
                 "cache_control": {"type":"ephemeral"}}],
        messages=[{"role":"user", "content":
            "Analyze COST order-flow reactions to these macro events.\n\n"
            + json.dumps(results, indent=2, default=str)}]
    )

    text = message.content[0].text
    u    = message.usage
    print(f"\nClaude usage: input={u.input_tokens}  output={u.output_tokens}  "
          f"cache_write={getattr(u,'cache_creation_input_tokens',0)}  "
          f"cache_read={getattr(u,'cache_read_input_tokens',0)}")

    out = {"events": results, "claude": text,
           "model": "claude-haiku-4-5-20251001",
           "usage": {"input": u.input_tokens, "output": u.output_tokens}}
    with open(OUTPUT_DIR / "step33_claude_interpretation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n=== Claude Interpretation ===")
    print(text)

print("\nDone.")
