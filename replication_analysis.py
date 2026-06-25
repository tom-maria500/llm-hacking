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

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import parse_outputs_jsonl as P
from resample_effect import (
    EFFECT_LEVELS, CONFIGS, TARGET_LABEL, load_ground_truth, cell_items,
    reweighted_pct,
)

N_BOOT = 4000  # bootstrap draws for the same-model test-retest baseline

THRESHOLD = 53.0  # original design "hacked" threshold (0.50 + 3pp margin)
OUTDIR = P.ROOT / "analysis_resampled"


def build_value_matrix(gt, cells, x: float) -> pd.DataFrame:
    """Models (rows, release-date order) x configs (cols) of reweighted % at
    effect x; NaN where the cell is blank."""
    order = P.models_by_release_date()
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


def bootstrap_detect_prob(items, gt, x: float, thr: float, rng, n_boot: int):
    """Probability that confirming this SAME model on a fresh sample of the 100
    study items still clears thr. Resamples within each ground-truth group -- the
    two rates the reweighting estimates -- so it isolates 100-item sampling noise,
    the only source of variation since the model is deterministic here. Like the
    protocol, the discovery is the cell's real (observed) value; the same-model
    confirmation therefore carries the regression-to-the-mean (winner's curse)
    that re-using the same model incurs and a different model avoids."""
    a = np.array([lbl == TARGET_LABEL for iid, lbl in items if gt.get(iid) is True], float)
    b = np.array([lbl == TARGET_LABEL for iid, lbl in items if gt.get(iid) is False], float)
    nA, nB = len(a), len(b)
    if nA and nB:
        pct = x * (rng.binomial(nA, a.mean(), n_boot) / nA) \
            + (100 - x) * (rng.binomial(nB, b.mean(), n_boot) / nB)
    elif nA:  # one group only: matches reweighted_pct's fallback
        pct = 100.0 * (rng.binomial(nA, a.mean(), n_boot) / nA)
    elif nB:
        pct = 100.0 * (rng.binomial(nB, b.mean(), n_boot) / nB)
    else:
        return np.nan
    return float((pct >= thr).mean())


N_CELL_BOOT = 2000  # cell-level bootstrap draws for the 95% CIs


def evaluate_with_ci(vals: pd.DataFrame, gt, cells, order, x: float, thr: float,
                     rng, n_boot: int = N_BOOT, n_cell_boot: int = N_CELL_BOOT):
    """For one effect level, collect both confirmation outcomes per detected cell
    and return point estimates with 95% CIs.

    For every detected cell (model i, config c) that has a later model to confirm
    on, we record two paired outcomes:
      protocol : 1 if the NEXT released model also clears thr, else 0
      baseline : P(the SAME model on a fresh item sample clears thr)  [bootstrap]

    The replication rate / same-model baseline are the means of those two columns
    over cells. The cells are NOT independent -- one model contributes up to 11
    correlated configs -- so 95% CIs come from a CLUSTER bootstrap that resamples
    the discovering models (the unit of generalization, ~19 of them) with
    replacement and pools all cells of the chosen models. The same resample is
    applied to both columns, so the two bands are comparable.
    Returns dict with point estimates, CI bounds, and counts."""
    detected = vals.ge(thr).values & vals.notna().values
    rep, base, disc_model = [], [], []
    for i in range(len(vals)):
        for c in range(vals.shape[1]):
            if not detected[i, c]:
                continue
            j = next_valid(vals, i, c)
            if j is None:
                continue
            p = bootstrap_detect_prob(cells[(order[i][1], CONFIGS[c][0])],
                                      gt, x, thr, rng, n_boot)
            if np.isnan(p):
                continue
            rep.append(1.0 if detected[j, c] else 0.0)
            base.append(p)
            disc_model.append(i)  # cluster id = discovering model
    rep, base = np.asarray(rep), np.asarray(base)
    n = len(rep)
    if n == 0:
        nan = float("nan")
        return dict(rate=nan, rate_lo=nan, rate_hi=nan,
                    base=nan, base_lo=nan, base_hi=nan, n=0, n_clusters=0)

    # group cell indices by discovering model (the cluster)
    clusters = {}
    for k, m in enumerate(disc_model):
        clusters.setdefault(m, []).append(k)
    cluster_ids = list(clusters.keys())
    rep_bs = np.empty(n_cell_boot)
    base_bs = np.empty(n_cell_boot)
    for t in range(n_cell_boot):
        chosen = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([clusters[m] for m in chosen])
        rep_bs[t] = rep[idx].mean()
        base_bs[t] = base[idx].mean()
    return dict(
        rate=float(rep.mean()),
        rate_lo=float(np.percentile(rep_bs, 2.5)),
        rate_hi=float(np.percentile(rep_bs, 97.5)),
        base=float(base.mean()),
        base_lo=float(np.percentile(base_bs, 2.5)),
        base_hi=float(np.percentile(base_bs, 97.5)),
        n=n,
        n_clusters=len(cluster_ids),
    )


def write_pgfplots(res: pd.DataFrame, path):
    """Emit a self-contained pgfplots/TikZ figure with a shaded 95% CI band for
    both lines (uses the `fillbetween` library)."""
    def coords(xs, ys):
        return " ".join(f"({x},{y:.4f})" for x, y in zip(xs, ys))
    xs = res["effect_strength"].tolist()
    tex = rf"""% Requires: \usepackage{{pgfplots}} \pgfplotsset{{compat=1.18}}
%           \usepgfplotslibrary{{fillbetween}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=10cm, height=7cm,
    xlabel={{Injected effect strength (true \% of items where Day~2 is lower)}},
    ylabel={{Effect re-detected on confirmation (cell $\geq$ {THRESHOLD:.0f}\%)}},
    title={{Diet task: replication of a true effect across model releases}},
    xmin={min(xs)-1}, xmax={max(xs)+0.5}, ymin=0, ymax=1.02,
    xtick={{{','.join(str(x) for x in xs)}}},
    grid=both, grid style={{gray!20}},
    legend pos=south east, legend cell align=left,
]
% ---- protocol 95% CI band ----
\addplot[draw=none, name path=protoU, forget plot] coordinates {{{coords(xs, res['rate_hi'])}}};
\addplot[draw=none, name path=protoL, forget plot] coordinates {{{coords(xs, res['rate_lo'])}}};
\addplot[blue!12, forget plot] fill between[of=protoU and protoL];
% ---- baseline 95% CI band ----
\addplot[draw=none, name path=baseU, forget plot] coordinates {{{coords(xs, res['base_hi'])}}};
\addplot[draw=none, name path=baseL, forget plot] coordinates {{{coords(xs, res['base_lo'])}}};
\addplot[black!12, forget plot] fill between[of=baseU and baseL];
% ---- mean lines ----
\addplot[blue, mark=*, thick] coordinates {{{coords(xs, res['replication_rate'])}}};
\addlegendentry{{our protocol (confirm on next model)}}
\addplot[black!60, dashed, mark=square*, thick] coordinates {{{coords(xs, res['same_model_baseline'])}}};
\addlegendentry{{baseline (re-measure same model)}}
\end{{axis}}
\end{{tikzpicture}}
"""
    Path(path).write_text(tex, encoding="utf-8")


def main():
    gt = load_ground_truth()
    cells = {}
    for _mn, ms, _prov, _rd in P.MODELS:
        for cfg, _col in P.SINGLE:
            cells[(ms, cfg)] = cell_items(ms, cfg, is_batch=False)
        for cfg, _col in P.BATCH:
            cells[(ms, cfg)] = cell_items(ms, cfg, is_batch=True)

    print(f"Detection threshold: cell >= {THRESHOLD}%")

    order = P.models_by_release_date()
    rng = np.random.default_rng(0)
    rows = []
    for x in EFFECT_LEVELS:
        vals = build_value_matrix(gt, cells, x)
        nd = int((vals.ge(THRESHOLD).values & vals.notna().values).sum())
        ci = evaluate_with_ci(vals, gt, cells, order, x, THRESHOLD, rng)
        rows.append({"effect_strength": x, "threshold_pct": THRESHOLD,
                     "n_detected": nd, "evaluable": ci["n"],
                     "n_clusters": ci["n_clusters"],
                     "replication_rate": ci["rate"],
                     "rate_lo": ci["rate_lo"], "rate_hi": ci["rate_hi"],
                     "same_model_baseline": ci["base"],
                     "base_lo": ci["base_lo"], "base_hi": ci["base_hi"]})
    res = pd.DataFrame(rows)
    res.to_csv(OUTDIR / "replication_rates.csv", index=False)
    print("\n", res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- plot with 95% CI bands ----
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = res["effect_strength"]
    ax.fill_between(xs, res["rate_lo"], res["rate_hi"], color="tab:blue", alpha=0.15)
    ax.plot(xs, res["replication_rate"], marker="o", color="tab:blue",
            label="our protocol (confirm on next released model)")
    ax.fill_between(xs, res["base_lo"], res["base_hi"], color="gray", alpha=0.15)
    ax.plot(xs, res["same_model_baseline"], marker="s", linestyle="--", color="tab:gray",
            label="baseline (re-measure the same model)")
    ax.set_xlabel("Injected effect strength  (true % of items where Day 2 is lower)")
    ax.set_ylabel(f"Effect re-detected on confirmation  (cell ≥ {THRESHOLD:.0f}%)")
    ax.set_title("Diet task: replication of a true effect across model releases")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(min(EFFECT_LEVELS) - 1, max(EFFECT_LEVELS) + 0.5)
    ax.set_xticks(list(EFFECT_LEVELS))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Confirmation step (shaded = 95% CI)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "replication_rate_vs_effect.png", dpi=150)

    write_pgfplots(res, OUTDIR / "replication_rate_vs_effect.tex")
    print(f"\nwrote {OUTDIR/'replication_rates.csv'}")
    print(f"wrote {OUTDIR/'replication_rate_vs_effect.png'}")
    print(f"wrote {OUTDIR/'replication_rate_vs_effect.tex'}")


if __name__ == "__main__":
    main()
