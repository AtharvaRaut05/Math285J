"""
Step 4bc v2 — Improved Feature Collection and Unsupervised Strategy Detection

What was wrong with v1:
  - Mid-price was approximated from execution prices instead of the reconstructed LOB
  - Synthetic labels were circular: rules were written on the same features the model
    would learn, so XGBoost was just re-learning the synthetic rule definitions
  - "Validation" measured accuracy on synthetic data — meaningless as ground truth

What this version does instead:

  FEATURES  (18 total, all derived from step31 reconstructed LOB + execution sequence)
    - Momentum at 1s / 5s / 30s from actual mid_at_exec series
    - Realised volatility (std of mid returns in 30s window)
    - Execution-based OFI (bid vol - ask vol in trailing 60s)
    - Inter-arrival time (time since last execution, any side + same side)
    - Size z-score and round-lot flag
    - Price-to-mid in bps (how far inside / outside the spread)
    - Spread in ticks, queue depth z-score
    - Directional persistence (fraction bid in trailing 60s executions)
    - Session phase (pre-market / open / mid-day / power-hour)

  STRATEGY CLASSIFICATION  (Step 4c)
    K-means clustering (unsupervised, k chosen by silhouette + elbow)
    No synthetic labels — clusters are interpreted post-hoc by their feature
    profiles compared to literature benchmarks.

  VALIDATION  (three layers, no synthetic ground truth)
    1. Internal:  silhouette score, Davies-Bouldin index, inertia elbow
    2. Temporal:  fit K-means on days 1-8, project days 9-11 to nearest centroid,
                  compute Jensen-Shannon divergence of cluster distributions
    3. Discriminability: XGBoost trained on cluster labels from train period,
                  tested on held-out test period — high AUC means clusters
                  are repeatable and separable, not just artefacts of fitting

  REGRESSION  (Step 4b)
    Logistic regression with is_hidden × momentum interaction term.
    Run separately on visible vs hidden executions to avoid the is_hidden
    coefficient swamping all other signals.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.special import expit as sigmoid

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split
import xgboost as xgb

OUTPUT_DIR = Path(r"C:\Users\komal\Project285J\output")
OUTPUT_DIR.mkdir(exist_ok=True)

TICK_USD  = 0.01   # COST minimum tick
K_RANGE   = range(2, 9)
K_FINAL   = 4      # overridden by silhouette if a better k is found
N_TRAIN_DAYS = 8   # temporal split point

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load step31 output (reconstructed LOB context for every execution)
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading step31 LOB context...")
ctx = pd.read_csv(OUTPUT_DIR / "step31_lob_context.csv")
ctx = ctx.sort_values(["date", "timestamp"]).reset_index(drop=True)
ctx["is_hidden"] = (ctx["exec_type"] == "hid_exec").astype(int)
ctx["is_bid"]    = (ctx["direction"] == "bid").astype(int)

dates_sorted = sorted(ctx["date"].unique())
train_dates  = set(dates_sorted[:N_TRAIN_DAYS])
test_dates   = set(dates_sorted[N_TRAIN_DAYS:])
print(f"  {len(ctx):,} executions | train: {sorted(train_dates)} | test: {sorted(test_dates)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature engineering — entirely from step31 columns + execution sequence
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_lookback(times: np.ndarray, values: np.ndarray, t: float, window: float):
    """Return values array slice for the window (t-window, t)."""
    lo = int(np.searchsorted(times, t - window, side="left"))
    hi = int(np.searchsorted(times, t, side="right"))
    return values[lo:hi]


def session_phase(t_sec: float) -> int:
    """0=pre-market, 1=open rush (9:30-10:00), 2=mid-day, 3=power hour (≥15:00)."""
    if t_sec < 34200:    return 0
    if t_sec < 36000:    return 1
    if t_sec < 54000:    return 2
    return 3


def build_features_for_day(day_df: pd.DataFrame) -> pd.DataFrame:
    day_df = day_df.copy().reset_index(drop=True)
    n = len(day_df)

    times     = day_df["timestamp"].values
    mids      = day_df["mid_at_exec_usd"].values
    bid_flag  = day_df["is_bid"].values
    sizes     = day_df["size_shares"].values.astype(float)

    # day-level normalisation constants
    mean_size = float(np.mean(sizes))
    std_size  = float(np.std(sizes)) + 1e-9
    mean_q    = float(day_df["queue_depth"].mean())
    std_q     = float(day_df["queue_depth"].std()) + 1e-9

    feats = []
    for i in range(n):
        t    = times[i]
        mid  = mids[i]
        side = bid_flag[i]
        sz   = sizes[i]

        # ── Momentum (from actual reconstructed mid series) ───────────────────
        def mom_bps(lag_s):
            lo = int(np.searchsorted(times, t - lag_s, side="right")) - 1
            if lo < 0 or mids[lo] == 0:
                return 0.0
            return (mid - mids[lo]) / (mids[lo] + 1e-9) * 10000

        m1  = mom_bps(1.0)
        m5  = mom_bps(5.0)
        m30 = mom_bps(30.0)

        # ── Realised volatility (std of returns in 30s window) ────────────────
        lo30 = int(np.searchsorted(times, t - 30.0, side="left"))
        hi30 = i + 1
        if hi30 - lo30 >= 5:
            m_win  = mids[lo30:hi30]
            rets   = np.diff(m_win) / (m_win[:-1] + 1e-9)
            vol30  = float(np.std(rets)) * 10000
        else:
            vol30  = np.nan

        # ── Execution-based OFI in trailing 60s ───────────────────────────────
        lo60 = int(np.searchsorted(times, t - 60.0, side="left"))
        win_bid = (bid_flag[lo60:i] * sizes[lo60:i]).sum()
        win_ask = ((1 - bid_flag[lo60:i]) * sizes[lo60:i]).sum()
        total60 = win_bid + win_ask + 1e-9
        oib60   = float((win_bid - win_ask) / total60)   # [-1, +1]
        exec_count60 = max(i - lo60, 1)

        # ── Directional persistence (fraction bid in trailing 60s execs) ──────
        dir_pers = float(bid_flag[lo60:i].mean()) if i > lo60 else 0.5

        # ── Inter-arrival times ───────────────────────────────────────────────
        if i > 0:
            inter_any = float(t - times[i - 1])
        else:
            inter_any = np.nan

        # same-side inter-arrival
        same_side = np.where(bid_flag[:i] == side)[0]
        if len(same_side):
            inter_same = float(t - times[same_side[-1]])
        else:
            inter_same = np.nan

        # ── Size z-score ──────────────────────────────────────────────────────
        size_z   = (sz - mean_size) / std_size
        round_lot = float(sz % 100 == 0)

        # ── Price-to-mid (how aggressively priced) ────────────────────────────
        ep = float(day_df["exec_price_usd"].iloc[i])
        p2m_bps = (ep - mid) / (mid + 1e-9) * 10000

        # ── Spread + queue ────────────────────────────────────────────────────
        spread_ticks = float(day_df["spread_usd"].iloc[i]) / TICK_USD
        q_z          = (float(day_df["queue_depth"].iloc[i]) - mean_q) / std_q

        feats.append({
            "mom_1s":       m1,
            "mom_5s":       m5,
            "mom_30s":      m30,
            "vol_30s":      vol30,
            "oib_60s":      oib60,
            "dir_pers_60s": dir_pers,
            "exec_cnt_60s": float(exec_count60),
            "inter_any_s":  inter_any,
            "inter_same_s": inter_same,
            "size_z":       size_z,
            "is_round_lot": round_lot,
            "price_to_mid": p2m_bps,
            "spread_ticks": spread_ticks,
            "queue_z":      q_z,
            # is_hidden and is_bid come from ctx; omit here to avoid dup columns
            "session_phase":float(session_phase(t)),
            "slippage_bps": float(day_df["slippage_usd"].iloc[i]) / (mid + 1e-9) * 10000,
        })

    return pd.DataFrame(feats)


print("Building features (may take ~2 min)...")
feat_parts = []
for date_str, day_df in ctx.groupby("date"):
    print(f"  [{date_str}] {len(day_df):,} execs")
    feat_parts.append(build_features_for_day(day_df))

feat_df = pd.concat(feat_parts, ignore_index=True)

FEATURE_COLS = list(feat_df.columns) + ["is_hidden", "is_bid"]

full = pd.concat([ctx.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
full = full.dropna(subset=FEATURE_COLS).reset_index(drop=True)
full["split"] = full["date"].apply(lambda d: "train" if d in train_dates else "test")

full.to_csv(OUTPUT_DIR / "step4bc_v2_features.csv", index=False)
print(f"\nFeature matrix: {len(full):,} rows x {len(FEATURE_COLS)} features")
print(f"  train: {(full.split=='train').sum():,}   test: {(full.split=='test').sum():,}")

X_all   = full[FEATURE_COLS].values
X_train = full.loc[full.split=="train", FEATURE_COLS].values
X_test  = full.loc[full.split=="test",  FEATURE_COLS].values

scaler  = StandardScaler()
Xs_all  = scaler.fit_transform(X_all)
Xs_tr   = scaler.transform(X_train)
Xs_te   = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Step 4b — Logistic Regression with interaction term, split by exec type
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "-"*60)
print("STEP 4b -- Logistic Regression (visible execs only, interaction)")
print("-"*60)

REG_FEATURES = ["oib_60s","mom_1s","mom_5s","vol_30s",
                "spread_ticks","queue_z","size_z","price_to_mid","session_phase"]

for subset_name, mask in [("visible", full.is_hidden == 0),
                           ("hidden",  full.is_hidden == 1)]:
    sub = full[mask].dropna(subset=REG_FEATURES)
    if len(sub) < 100:
        continue
    Xr = sub[REG_FEATURES].values
    yr = sub["is_bid"].values

    # add interaction: mom_1s * oib_60s
    mom_x_oib = (Xr[:, REG_FEATURES.index("mom_1s")] *
                 Xr[:, REG_FEATURES.index("oib_60s")]).reshape(-1,1)
    Xr_aug = np.hstack([Xr, mom_x_oib])
    feat_names = REG_FEATURES + ["mom1_x_oib"]

    Xr_sc  = StandardScaler().fit_transform(Xr_aug)
    Xr_sm  = sm.add_constant(Xr_sc)

    res = sm.Logit(yr, Xr_sm).fit(method="lbfgs", maxiter=500, disp=False)
    ci  = res.conf_int()

    tbl = pd.DataFrame({
        "feature": ["const"] + feat_names,
        "coef":    np.asarray(res.params),
        "z":       np.asarray(res.tvalues),
        "p":       np.asarray(res.pvalues),
        "ci_lo":   np.asarray(ci)[:, 0],
        "ci_hi":   np.asarray(ci)[:, 1],
    }).round(5)
    tbl.to_csv(OUTPUT_DIR / f"step4b_{subset_name}.csv", index=False)

    sig = tbl[tbl["p"] < 0.05].sort_values("z", key=abs, ascending=False)
    print(f"\n  {subset_name.upper()} executions (n={len(sub):,}, pseudo-R2={res.prsquared:.3f})")
    print(sig[["feature","coef","z","p"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Step 4c — K-means clustering + 3-layer validation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "-"*60)
print("STEP 4c -- K-means clustering (unsupervised)")
print("-"*60)

# ── 4c.1  Elbow + silhouette to choose k ──────────────────────────────────────
print("\nElbow / silhouette scan (k=2..8) on training set...")
sample_idx = np.random.default_rng(0).choice(len(Xs_tr),
                                              min(30_000, len(Xs_tr)),
                                              replace=False)
Xs_sample = Xs_tr[sample_idx]

elbow_rows = []
best_sil, best_k = -1.0, K_FINAL
for k in K_RANGE:
    km  = KMeans(n_clusters=k, n_init=10, random_state=42)
    lbl = km.fit_predict(Xs_sample)
    sil = silhouette_score(Xs_sample, lbl, sample_size=10_000, random_state=0)
    db  = davies_bouldin_score(Xs_sample, lbl)
    elbow_rows.append({"k": k, "inertia": km.inertia_,
                       "silhouette": round(sil, 4), "davies_bouldin": round(db, 4)})
    print(f"  k={k}  inertia={km.inertia_:,.0f}  sil={sil:.4f}  DB={db:.4f}")
    if sil > best_sil:
        best_sil, best_k = sil, k

pd.DataFrame(elbow_rows).to_csv(OUTPUT_DIR / "step4c_elbow.csv", index=False)
print(f"\nBest k by silhouette: k={best_k}  (sil={best_sil:.4f})")

# ── 4c.2  Final model on full training set ────────────────────────────────────
print(f"\nFitting final KMeans(k={best_k}) on {len(Xs_tr):,} training executions...")
km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42)
km_final.fit(Xs_tr)

train_labels = km_final.labels_
test_labels  = km_final.predict(Xs_te)
all_labels   = km_final.predict(Xs_all)

full["cluster"] = all_labels

sil_full = silhouette_score(Xs_tr, train_labels, sample_size=20_000, random_state=0)
db_full  = davies_bouldin_score(Xs_tr[:20_000], train_labels[:20_000])
print(f"Full-data silhouette={sil_full:.4f}  Davies-Bouldin={db_full:.4f}")

# ── 4c.3  Temporal stability (Jensen-Shannon divergence) ─────────────────────
train_dist = np.bincount(train_labels, minlength=best_k) / len(train_labels)
test_dist  = np.bincount(test_labels,  minlength=best_k) / len(test_labels)
js_div = jensenshannon(train_dist, test_dist)
print(f"\nTemporal stability (JS divergence train vs test): {js_div:.4f}")
print(f"  (0=identical distribution, 1=completely different; <0.05 = stable)")
print(f"  Train cluster dist: {np.round(train_dist, 3)}")
print(f"  Test  cluster dist: {np.round(test_dist,  3)}")

stab_df = pd.DataFrame({"cluster": range(best_k),
                         "train_frac": train_dist.round(4),
                         "test_frac":  test_dist.round(4)})
stab_df["shift"] = (stab_df.test_frac - stab_df.train_frac).round(4)
stab_df.to_csv(OUTPUT_DIR / "step4c_temporal_stability.csv", index=False)

# ── 4c.4  Discriminability: XGBoost predicts cluster from test features ───────
print("\nDiscriminability test: XGBoost on train labels -> predict test labels...")
clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="mlogloss", random_state=42)
clf.fit(Xs_tr, train_labels, verbose=False)
test_preds = clf.predict(Xs_te)
test_proba = clf.predict_proba(Xs_te)

disc_report = classification_report(test_labels, test_preds, output_dict=True)
disc_df = pd.DataFrame(disc_report).T
disc_df.to_csv(OUTPUT_DIR / "step4c_discriminability.csv")

ova_auc = roc_auc_score(
    np.eye(best_k)[test_labels], test_proba, average="macro", multi_class="ovr"
)
print(f"  OVA AUC (macro): {ova_auc:.4f}  (0.5=random, 1.0=perfect)")
print(f"  Accuracy on test: {disc_report['accuracy']:.4f}")
print("  Note: XGBoost here validates whether clusters generalise across time,")
print("        NOT whether they match a ground-truth strategy label.")

imp_df = pd.DataFrame({"feature": FEATURE_COLS,
                        "importance": clf.feature_importances_}
                      ).sort_values("importance", ascending=False)
imp_df.to_csv(OUTPUT_DIR / "step4c_importance.csv", index=False)
print("\nTop-8 discriminative features:")
print(imp_df.head(8).to_string(index=False))

# ── 4c.5  Cluster feature profiles + archetype matching ──────────────────────
print("\nCluster feature profiles (mean per cluster, scaled):")
profile_cols = ["mom_1s","mom_5s","vol_30s","oib_60s","dir_pers_60s",
                "inter_any_s","size_z","spread_ticks","is_hidden","price_to_mid",
                "session_phase","slippage_bps"]

profiles = (full.groupby("cluster")[profile_cols]
                .mean()
                .round(4))
profiles.to_csv(OUTPUT_DIR / "step4c_profiles.csv")
print(profiles.to_string())

# Heuristic archetype labels based on profile characteristics
ARCHETYPE_RULES = {
    "market_maker":  lambda r: r["spread_ticks"] < 3 and abs(r["oib_60s"]) < 0.15 and r["inter_any_s"] < 0.5,
    "momentum":      lambda r: r["mom_5s"] * (1 if r["dir_pers_60s"] > 0.5 else -1) > 0.5,
    "stat_arb":      lambda r: r["mom_1s"] * (1 if r["dir_pers_60s"] > 0.5 else -1) < -0.3,
    "latency_arb":   lambda r: r["inter_any_s"] < 0.2 and abs(r["mom_1s"]) > 0.3,
}

print("\nHeuristic archetype assignment:")
label_map = {}
for cl in range(best_k):
    row = profiles.loc[cl]
    matched = [name for name, rule in ARCHETYPE_RULES.items() if rule(row)]
    label = matched[0] if len(matched) == 1 else ("mixed: " + "+".join(matched) if matched else "unclassified")
    label_map[cl] = label
    n_cl = (full.cluster == cl).sum()
    print(f"  Cluster {cl} ({n_cl:,} execs, {100*n_cl/len(full):.1f}%): {label}")

full["archetype"] = full["cluster"].map(label_map)
full.to_csv(OUTPUT_DIR / "step4c_predictions_v2.csv", index=False)

# ── 4c.6  Per-day cluster distribution (shows strategy mix varies by day) ─────
day_dist = (full.groupby(["date","cluster"])
                .size()
                .unstack(fill_value=0)
                .apply(lambda r: r / r.sum(), axis=1)
                .round(3))
day_dist.to_csv(OUTPUT_DIR / "step4c_daily_distribution.csv")
print("\nPer-day cluster distribution:")
print(day_dist.to_string())

print("\nDone.")
