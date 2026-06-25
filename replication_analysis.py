#!/usr/bin/env python3
"""Replication-rate analysis for the diet task under injected effects.

We inject a true effect of size X% (resample_effect.py) and ask: how often does
a real effect that one model detects get RE-DETECTED ("replicated") by the next
released model? This is the inverse of analysis.py's mitigation rate -- here a
high value is good, because the effect is genuinely present.

Detection rule: a cell is "detected" if its reweighted Day-2 rate is at or above
the original design threshold of 53% (0.50 + the 3pp preregistration margin).

Replication rate = among detected cells (model i, config c) that have a later
model with a valid value for config c, the fraction where the *next* such model
also detects (mirrors analysis.mitigation_rate, with hacked -> detected and
"prevented" -> "replicated").

Outputs:
    analysis_resampled/replication_rates.csv
    analysis_resampled/replication_rate_vs_effect.png
"""
from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import parse_outputs_jsonl as P
from resample_effect import (
    EFFECT_LEVELS, CONFIGS, load_ground_truth, cell_items, reweighted_pct,
)

THRESHOLD = 53.0  # original design "hacked" threshold (0.50 + 3pp margin)
OUTDIR = P.ROOT / "analysis_resampled"


def build_value_matrix(gt, cells, x: float) -> pd.DataFrame:
    """Models (rows, release-date order) x configs (cols) of reweighted % at
    effect x; NaN where the cell is blank."""
    order = sorted(P.MODELS, key=lambda m: datetime.strptime(m[3], "%m/%d/%y"))
    data = []
    for _mn, ms, _prov, _rd in order:
        row = []
        for cfg, _col in CONFIGS:
            items = cells[(ms, cfg)]
            row.append(np.nan if items is None else reweighted_pct(items, gt, x))
        data.append(row)
    return pd.DataFrame(data, columns=[col for _c, col in CONFIGS])


def next_valid(vals: pd.DataFrame, i: int, c: int):
    col = vals.iloc[i + 1:, c]
    col = col[col.notna()]
    return col.index[0] if len(col) else None


def replication_rate(vals: pd.DataFrame, thr: float):
    """Returns (rate, replicated, evaluable, n_detected)."""
    detected = pd.DataFrame(vals.ge(thr).values & vals.notna().values,
                            columns=vals.columns)
    n_detected = int(detected.values.sum())
    replicated = evaluable = 0
    for i in range(len(vals)):
        for c in range(vals.shape[1]):
            if not detected.iat[i, c]:
                continue
            j = next_valid(vals, i, c)
            if j is None:
                continue
            evaluable += 1
            replicated += bool(detected.iat[j, c])
    rate = replicated / evaluable if evaluable else np.nan
    return rate, replicated, evaluable, n_detected


def main():
    gt = load_ground_truth()
    cells = {}
    for _mn, ms, _prov, _rd in P.MODELS:
        for cfg, _col in P.SINGLE:
            cells[(ms, cfg)] = cell_items(ms, cfg, is_batch=False)
        for cfg, _col in P.BATCH:
            cells[(ms, cfg)] = cell_items(ms, cfg, is_batch=True)

    print(f"Detection threshold: cell >= {THRESHOLD}%")

    rows = []
    for x in EFFECT_LEVELS:
        vals = build_value_matrix(gt, cells, x)
        rate, rep, ev, nd = replication_rate(vals, THRESHOLD)
        rows.append({"effect_strength": x, "threshold_pct": THRESHOLD,
                     "n_detected": nd, "evaluable": ev,
                     "replicated": rep, "replication_rate": rate})
    res = pd.DataFrame(rows)
    res.to_csv(OUTDIR / "replication_rates.csv", index=False)
    print("\n", res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(res["effect_strength"], res["replication_rate"],
            marker="o", color="tab:blue", label=f"detected if cell ≥ {THRESHOLD:.0f}%")
    ax.set_xlabel("Injected effect strength  (true % of items where Day 2 is lower)")
    ax.set_ylabel("Replication rate\n(effect re-detected on the next released model)")
    ax.set_title("Diet task: replication of a true effect across model releases")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(min(EFFECT_LEVELS) - 1, max(EFFECT_LEVELS) + 0.5)
    ax.set_xticks(list(EFFECT_LEVELS))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Detection threshold")
    fig.tight_layout()
    fig.savefig(OUTDIR / "replication_rate_vs_effect.png", dpi=150)
    print(f"\nwrote {OUTDIR/'replication_rates.csv'}")
    print(f"wrote {OUTDIR/'replication_rate_vs_effect.png'}")


if __name__ == "__main__":
    main()
