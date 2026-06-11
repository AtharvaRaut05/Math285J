"""
Steps 3.1 and 3.2 of HFT Order Intelligence methodology.

3.1 — Order Matching and Context Reconstruction
    For each execution event in the LOBSTER message stream, reconstruct the
    contemporaneous LOB snapshot and record: best bid/ask, spread,
    queue depth at the execution price, estimated fill probability, and slippage.

3.2 — PnL Attribution
    Compute realized PnL at horizons T ∈ {0.01 s, 0.5 s, 1 s, EOD} for every
    filled order using:
        PnL_T = d_i * (mid_{t+T} - p_i) * q_i   (in USD)
    where d_i = +1 (bid) or -1 (ask).

Outputs (saved to ./output/):
    step31_lob_context.csv   — one row per execution with LOB features
    step32_pnl_attribution.csv — one row per execution with PnL at each horizon
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from lobster_reconstructor import Orderbook, LobsterSim, Order

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\komal\Downloads\_data_dwn_67_491__COST_2025-10-01_2025-10-14_0")
OUTPUT_DIR = Path(r"C:\Users\komal\Project285J\output")
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER        = "COST"
PRICE_SCALING = 0.0001   # raw integer → USD
TICK_SIZE     = 100       # $0.01 in raw units
NLEVELS       = 10

PNL_HORIZONS = [0.01, 0.5, 1.0]   # seconds; EOD handled separately

EXEC_TYPES = {"vis_exec", "hid_exec"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def queue_depth_at(book: Orderbook, price: int, direction: str) -> int:
    side = book.bids if direction == "bid" else book.asks
    return sum(o.size for o in side.get(price, {}).values())


def slippage_usd(exec_price: int, mid: float, direction: str) -> float:
    """Signed implementation shortfall in USD per share (positive = cost to trader)."""
    if direction == "bid":
        return (exec_price - mid) * PRICE_SCALING
    return (mid - exec_price) * PRICE_SCALING


def lookup_mid(mp_times: np.ndarray, mp_prices: np.ndarray, target: float) -> float:
    """Return the mid-price immediately before or at `target` seconds."""
    idx = np.searchsorted(mp_times, target, side="right") - 1
    idx = max(idx, 0)
    return mp_prices[idx]


# ── Main loop ──────────────────────────────────────────────────────────────────

msg_files = sorted(DATA_DIR.glob(f"{TICKER}_*_message_0.csv"))
if not msg_files:
    sys.exit(f"No message files found in {DATA_DIR}")

all_context_rows: list[dict] = []
all_pnl_rows:     list[dict] = []

for msg_file in msg_files:
    # Extract date from filename, e.g. COST_2025-10-01_...
    date_str = msg_file.name.split("_")[1]
    print(f"[{date_str}] loading {msg_file.name} …")

    # Create a fresh book; LobsterSim loads dataM at init without simulating.
    book = Orderbook(nlevels=NLEVELS, ticker=TICKER, tick_size=TICK_SIZE,
                     price_scaling=PRICE_SCALING)
    sim  = LobsterSim(orderbook=book, msg_book_file_path=str(msg_file))
    messages = sim.dataM  # DataFrame: Time, Type, OrderID, Size, Price, Direction

    # Drop any rows with missing essential fields
    messages = messages.dropna(subset=["Time", "Type", "Size", "Price", "Direction"])

    context_rows: list[dict] = []
    mp_times:  list[float] = []
    mp_prices: list[float] = []

    for row in messages.itertuples(index=False):
        # ── Step 3.1: capture LOB context BEFORE processing execution ──────────
        if row.Type in EXEC_TYPES:
            mid = book.mid_price()
            if mid is not None:
                best_bid   = book.highest_bid_price()
                best_ask   = book.lowest_ask_price()
                spread_raw = book.bid_ask_spread()
                q_depth    = queue_depth_at(book, row.Price, row.Direction)
                slip       = slippage_usd(row.Price, mid, row.Direction)

                context_rows.append({
                    "date":            date_str,
                    "timestamp":       row.Time,
                    "exec_type":       row.Type,
                    "direction":       row.Direction,
                    "exec_price_usd":  row.Price   * PRICE_SCALING,
                    "size_shares":     row.Size,
                    "best_bid_usd":    best_bid    * PRICE_SCALING,
                    "best_ask_usd":    best_ask    * PRICE_SCALING,
                    "spread_usd":      spread_raw  * PRICE_SCALING,
                    "queue_depth":     q_depth,
                    "fill_probability": 1.0,       # confirmed fill from LOBSTER
                    "slippage_usd":    round(slip, 6),
                    "mid_at_exec_usd": mid         * PRICE_SCALING,
                })

        # Process the event to update the LOB
        try:
            order = Order(row.Time, row.Type, row.OrderID, row.Size,
                          row.Price, row.Direction)
            book.process_order(order)
        except (ValueError, KeyError):
            continue

        # Record mid-price after every update (for PnL horizon lookups)
        mid_after = book.mid_price()
        if mid_after is not None:
            mp_times.append(row.Time)
            mp_prices.append(mid_after)

    if not context_rows or not mp_times:
        print(f"  [skip] no executions or mid-price data for {date_str}")
        continue

    mp_t = np.asarray(mp_times,  dtype=np.float64)
    mp_p = np.asarray(mp_prices, dtype=np.float64)
    eod_mid_usd = mp_p[-1] * PRICE_SCALING

    print(f"  {len(context_rows):,} executions | EOD mid = ${eod_mid_usd:.2f}")

    all_context_rows.extend(context_rows)

    # ── Step 3.2: PnL attribution ─────────────────────────────────────────────
    for rec in context_rows:
        t_exec  = rec["timestamp"]
        p_exec  = rec["exec_price_usd"]   # already in USD
        size    = rec["size_shares"]
        d_i     = 1 if rec["direction"] == "bid" else -1

        pnl_row: dict = {
            "date":         rec["date"],
            "timestamp":    t_exec,
            "exec_type":    rec["exec_type"],
            "direction":    rec["direction"],
            "exec_price_usd": p_exec,
            "size_shares":  size,
        }

        for T in PNL_HORIZONS:
            mid_T_raw = lookup_mid(mp_t, mp_p, t_exec + T)
            mid_T_usd = mid_T_raw * PRICE_SCALING
            pnl       = d_i * (mid_T_usd - p_exec) * size
            pnl_row[f"mid_T{T}s_usd"] = round(mid_T_usd, 4)
            pnl_row[f"pnl_T{T}s_usd"] = round(pnl, 4)

        # EOD
        eod_mid_raw = mp_p[-1]
        mid_eod_usd = eod_mid_raw * PRICE_SCALING
        pnl_eod     = d_i * (mid_eod_usd - p_exec) * size
        pnl_row["mid_EOD_usd"] = round(mid_eod_usd, 4)
        pnl_row["pnl_EOD_usd"] = round(pnl_eod, 4)

        all_pnl_rows.append(pnl_row)

# ── Save Step 3.1 output ───────────────────────────────────────────────────────
if all_context_rows:
    ctx_df = pd.DataFrame(all_context_rows)
    ctx_path = OUTPUT_DIR / "step31_lob_context.csv"
    ctx_df.to_csv(ctx_path, index=False)
    print(f"\nStep 3.1 -> {ctx_path}  ({len(ctx_df):,} rows)")

# ── Save Step 3.2 output ───────────────────────────────────────────────────────
if all_pnl_rows:
    pnl_df = pd.DataFrame(all_pnl_rows)
    pnl_path = OUTPUT_DIR / "step32_pnl_attribution.csv"
    pnl_df.to_csv(pnl_path, index=False)
    print(f"\nStep 3.2 -> {pnl_path}  ({len(pnl_df):,} rows)")

    print("\nPnL summary (USD, all executions, all days):")
    cols = [f"pnl_T{T}s_usd" for T in PNL_HORIZONS] + ["pnl_EOD_usd"]
    labels = [f"T={T}s" for T in PNL_HORIZONS] + ["EOD"]
    for col, label in zip(cols, labels):
        s = pnl_df[col]
        print(f"  {label:>6s}: mean={s.mean():+.4f}  median={s.median():+.4f}"
              f"  std={s.std():.4f}  n={s.count():,}")

print("\nDone.")
