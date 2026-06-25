#!/usr/bin/env python3
"""Replication-rate analysis for the diet task under injected effects.

We inject a true effect of size X% and ask: how often does a real effect that one
model detects get RE-DETECTED ("replicated") by the next released model? This is
the inverse of analysis.py's mitigation rate -- here a high value is good, because
the effect is genuinely present.

Effect injection (sampling, not reweighting): for each (model, config) cell we
draw the cell's valid items WITH REPLACEMENT, picking each item from the
true-Day-2 group with probability X/100 and from the other group with (100-X)/100
-- i.e. group membership is drawn proportional to the injected effect strength
(resample_effect.sampled_pct). The cell value is the observed % of 'Day 2'
outputs in that draw. Unlike deterministic reweighting, this carries genuine
finite-sample noise, so we Monte-Carlo it over N_INJECT rounds.

Detection rule: a cell is "detected" if its sampled Day-2 rate is at or above the
original design threshold of 53% (0.50 + the 3pp preregistration margin).

Replication rate = among detected cells (model i, config c) that have a later
model with data for config c, the fraction where the *next* such model also
detects, pooled over the injection rounds (mirrors analysis.mitigation_rate, with
hacked -> detected and "prevented" -> "replicated").

Because detection is now itself noisy, the same-model baseline -- re-drawing the
SAME discovering cell -- carries the regression-to-the-mean (winner's curse) that
re-using a model incurs and the next model avoids.

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
    cell_groups, sampled_pct_from_groups,
)

N_INJECT = 500   # Monte Carlo effect-injection rounds (sampling with replacement)
N_CELL_BOOT = 2000  # cluster-bootstrap draws for the 95% CIs

THRESHOLD = 53.0  # original design "hacked" threshold (0.50 + 3pp margin)
OUTDIR = P.ROOT / "analysis_resampled"


def build_groups(gt, cells, order):
    """Precompute, once, each cell's (a, b) ground-truth group indicator arrays
    and a structural availability mask. Returns (groups, avail):
        groups[(i, c)] = (a, b) float arrays, or None for a blank cell
        avail[i, c]    = True iff the cell has data to sample from
    A cell is available iff it has valid items in at least one ground-truth group;
    availability is a property of the DATA (not of any random draw), so it fixes
    which later model counts as the 'next' confirmer."""
    nmod, ncfg = len(order), len(CONFIGS)
    groups = {}
    avail = np.zeros((nmod, ncfg), bool)
    for i, (_mn, ms, _prov, _rd) in enumerate(order):
        for c, (cfg, _col) in enumerate(CONFIGS):
            items = cells[(ms, cfg)]
            if items is None:
                groups[(i, c)] = None
                continue
            a, b = cell_groups(items, gt)
            if len(a) + len(b) == 0:
                groups[(i, c)] = None
            else:
                groups[(i, c)] = (a, b)
                avail[i, c] = True
    return groups, avail


def next_valid_map(avail):
    """nxt[i, c] = index of the next available model after i for config c, or -1.
    Precomputed from the structural availability mask (see build_groups)."""
    nmod, ncfg = avail.shape
    nxt = np.full((nmod, ncfg), -1, int)
    for c in range(ncfg):
        later = -1
        for i in range(nmod - 1, -1, -1):
            nxt[i, c] = later
            if avail[i, c]:
                later = i
    return nxt


def evaluate_with_ci(groups, avail, nxt, order, x: float, thr: float, rng,
                     n_inject: int = N_INJECT, n_cell_boot: int = N_CELL_BOOT):
    """For one effect level, Monte-Carlo the sampled-with-replacement injection
    over n_inject rounds and return point estimates with 95% CIs.

    Each round draws a discovery value for every available cell
    (sampled_pct_from_groups, group membership prop. to the effect x). A cell is
    detected if its draw clears thr. For every detected cell (model i, config c)
    that has a later model j with data for config c we record two paired 0/1
    outcomes:
      protocol : 1 if the NEXT released model's draw also clears thr
      baseline : 1 if a FRESH independent draw of the SAME cell clears thr

    Pooling over rounds, the replication rate / same-model baseline are the means
    of those columns. Detection is itself noisy here, so a near-threshold cell is
    selected partly by luck and the same-model baseline regresses on its fresh
    draw -- the winner's curse a different (next) model does not inherit.

    Cells are NOT independent -- one model contributes up to 11 correlated configs
    -- so 95% CIs come from a CLUSTER bootstrap that resamples the discovering
    models (the unit of generalization, ~19 of them) with replacement and pools
    all of the chosen models' detected-cell outcomes (across all rounds). The same
    resample drives both bands, so they are comparable."""
    nmod, ncfg = len(order), len(CONFIGS)
    rep, base, disc_model = [], [], []
    det_per_round = np.zeros(n_inject)
    for t in range(n_inject):
        disc = np.full((nmod, ncfg), np.nan)
        for i in range(nmod):
            for c in range(ncfg):
                g = groups[(i, c)]
                if g is not None:
                    disc[i, c] = sampled_pct_from_groups(g[0], g[1], x, rng)
        detected = disc >= thr  # NaN >= thr is False
        det_per_round[t] = detected.sum()
        for i in range(nmod):
            for c in range(ncfg):
                if not detected[i, c]:
                    continue
                j = nxt[i, c]
                if j < 0:
                    continue
                g = groups[(i, c)]
                fresh = sampled_pct_from_groups(g[0], g[1], x, rng)
                rep.append(1.0 if detected[j, c] else 0.0)
                base.append(1.0 if fresh >= thr else 0.0)
                disc_model.append(i)  # cluster id = discovering model
    rep, base = np.asarray(rep), np.asarray(base)
    n = len(rep)
    if n == 0:
        nan = float("nan")
        return dict(rate=nan, rate_lo=nan, rate_hi=nan, base=nan, base_lo=nan,
                    base_hi=nan, n=0.0, n_clusters=0, n_detected=0.0)

    # cluster = discovering model; precompute per-cluster sums for a fast bootstrap
    clusters = {}
    for k, m in enumerate(disc_model):
        clusters.setdefault(m, []).append(k)
    cluster_ids = list(clusters.keys())
    rep_sum = np.array([rep[clusters[m]].sum() for m in cluster_ids])
    base_sum = np.array([base[clusters[m]].sum() for m in cluster_ids])
    cnt = np.array([len(clusters[m]) for m in cluster_ids], float)
    K = len(cluster_ids)
    rep_bs = np.empty(n_cell_boot)
    base_bs = np.empty(n_cell_boot)
    for t in range(n_cell_boot):
        sel = rng.integers(0, K, size=K)  # resample models with replacement
        tot = cnt[sel].sum()
        rep_bs[t] = rep_sum[sel].sum() / tot
        base_bs[t] = base_sum[sel].sum() / tot
    return dict(
        rate=float(rep.mean()),
        rate_lo=float(np.percentile(rep_bs, 2.5)),
        rate_hi=float(np.percentile(rep_bs, 97.5)),
        base=float(base.mean()),
        base_lo=float(np.percentile(base_bs, 2.5)),
        base_hi=float(np.percentile(base_bs, 97.5)),
        n=float(n) / n_inject,                 # mean evaluable cells per round
        n_clusters=K,
        n_detected=float(det_per_round.mean()),  # mean detected cells per round
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
    print(f"Injection: sampling WITH REPLACEMENT, {N_INJECT} Monte Carlo rounds")

    order = P.models_by_release_date()
    groups, avail = build_groups(gt, cells, order)
    nxt = next_valid_map(avail)
    rng = np.random.default_rng(0)
    rows = []
    for x in EFFECT_LEVELS:
        ci = evaluate_with_ci(groups, avail, nxt, order, x, THRESHOLD, rng)
        rows.append({"effect_strength": x, "threshold_pct": THRESHOLD,
                     "n_detected": ci["n_detected"], "evaluable": ci["n"],
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
