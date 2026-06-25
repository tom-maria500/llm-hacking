#!/usr/bin/env python3
"""Create artificial-effect diet result matrices by reweighting the 100 study items.

For the diet task the natural base rate of "Day 2 truly lower in calories" is
~50% (diet_ground_truth.csv). To inject a real effect of size X% we reweight the
original 100 items by their ground-truth group:

    group A = items where Day 2 is truly lower (later_lower=True)
    group B = items where Day 1 is truly lower (later_lower=False)

so that group A carries X% of the weight and group B carries (100-X)%. Each cell
is then the reweighted % of the model's XOR-corrected outputs equal to "Day 2":

    cell(X) = X * (a / nA) + (100 - X) * (b / nB)

where a/b = # of "Day 2" outputs and nA/nB = # of valid outputs within each
ground-truth group, for that (model, config). Because models classify calories
with real accuracy, a higher injected effect X propagates into a higher observed
cell value — that is the signal whose reproduction we later measure.

Blank cells (fewer than MIN_VALID valid outputs, or missing data) stay blank,
exactly as in diet.csv.

Writes analysis_resampled/diet_effect_{X}.csv for each X in EFFECT_LEVELS, with
the same columns/rows as diet.csv.
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import parse_outputs_jsonl as P

EFFECT_LEVELS = [55, 60, 65, 70, 75, 80, 85, 90, 95]
TARGET_LABEL = "Day 2"
OUTDIR = P.ROOT / "analysis_resampled"

HEADER = ["Model", "Provider", "Release Date",
          *[col for _c, col in P.SINGLE], *[col for _c, col in P.BATCH]]
CONFIGS = [(c, col) for c, col in P.SINGLE] + [(c, col) for c, col in P.BATCH]


def load_ground_truth() -> dict[int, bool]:
    gt = pd.read_csv(P.ROOT / "diet_ground_truth.csv")
    return dict(zip(gt["user_id"].astype(int), gt["later_lower"].astype(bool)))


def cell_items(model_slug: str, cfg: str, is_batch: bool):
    """Return list of (item_id:int, table_label) for valid items, or None if the
    cell is blank (candidate below MIN_VALID / missing), matching diet.csv."""
    if is_batch:
        cand = P.batch_candidate(P.OUTPUTS_DIR, "diet", model_slug, cfg)
    else:
        cand = P.single_candidate(P.OUTPUTS_DIR, "diet", model_slug, cfg)
    if not cand:
        return None
    return [(int(r["item_id"]), r["parsed_table_label"])
            for r in cand["item_rows"] if r["is_valid"]]


def reweighted_pct(items, gt: dict[int, bool], x: float):
    """Reweight valid items so the true-Day-2 group has weight x% and the other
    group (100-x)%; return reweighted % of 'Day 2' outputs."""
    a = nA = b = nB = 0
    for iid, label in items:
        truth = gt.get(iid)
        if truth is None:
            continue  # no ground truth -> excluded from reweighting
        if truth:  # group A: Day 2 truly lower
            nA += 1
            a += (label == TARGET_LABEL)
        else:      # group B: Day 1 truly lower
            nB += 1
            b += (label == TARGET_LABEL)
    if nA and nB:
        return x * (a / nA) + (100 - x) * (b / nB)
    if nA:  # only one group present: fall back to its own rate
        return 100 * a / nA
    if nB:
        return 100 * b / nB
    return None


def cell_groups(items, gt: dict[int, bool]):
    """Split a cell's valid, ground-truthed items into the two groups, returning
    (a, b) float arrays of 1/0 indicators for a 'Day 2' (TARGET_LABEL) output.
        a = group A items (Day 2 truly lower)
        b = group B items (Day 1 truly lower)
    Items with no ground truth are dropped, matching reweighted_pct()."""
    a = np.array([lbl == TARGET_LABEL
                  for iid, lbl in items if gt.get(iid) is True], float)
    b = np.array([lbl == TARGET_LABEL
                  for iid, lbl in items if gt.get(iid) is False], float)
    return a, b


def sampled_pct_from_groups(a, b, x: float, rng, n: int | None = None):
    """Inject a true effect of size x% by SAMPLING the cell's items WITH
    REPLACEMENT rather than reweighting them deterministically.

    Each of the n draws comes from group A (Day 2 truly lower) with probability
    x/100 and from group B with probability (100-x)/100 -- i.e. group membership
    is drawn proportional to the injected effect strength -- and then a uniform
    item *within* that group (with replacement). Returns the observed % of 'Day 2'
    outputs across the n draws.

    Its expectation equals reweighted_pct(); the difference is that this carries
    the finite-sample (n-item) sampling noise a real study of that size incurs,
    so a discovery near threshold can clear it by chance and regress on a fresh
    draw (the winner's curse). With n defaulting to the cell's valid-item count,
    the noise matches the original study size."""
    nA, nB = len(a), len(b)
    if n is None:
        n = nA + nB
    if n == 0:
        return None
    if nA and nB:
        from_a = int(rng.binomial(n, x / 100.0))  # group drawn prop. to effect
        hits = 0.0
        if from_a:
            hits += rng.choice(a, size=from_a, replace=True).sum()
        if n - from_a:
            hits += rng.choice(b, size=n - from_a, replace=True).sum()
        return 100.0 * hits / n
    if nA:  # one group only: matches reweighted_pct's fallback
        return 100.0 * rng.choice(a, size=n, replace=True).sum() / n
    return 100.0 * rng.choice(b, size=n, replace=True).sum() / n


def sampled_pct(items, gt: dict[int, bool], x: float, rng, n: int | None = None):
    """Convenience wrapper: build the (a, b) groups for a cell and draw one
    sampled-with-replacement effect realization (see sampled_pct_from_groups)."""
    a, b = cell_groups(items, gt)
    return sampled_pct_from_groups(a, b, x, rng, n)


def main():
    gt = load_ground_truth()
    OUTDIR.mkdir(exist_ok=True)

    # Precompute per-cell valid items once (reused across all effect levels).
    cells: dict[tuple[str, str], object] = {}
    for _mn, ms, _prov, _rd in P.MODELS:
        for cfg, _col in P.SINGLE:
            cells[(ms, cfg)] = cell_items(ms, cfg, is_batch=False)
        for cfg, _col in P.BATCH:
            cells[(ms, cfg)] = cell_items(ms, cfg, is_batch=True)

    for x in EFFECT_LEVELS:
        rows = [HEADER]
        for mn, ms, prov, rd in P.models_by_release_date():
            row = [mn, prov, rd]
            for cfg, _col in CONFIGS:
                items = cells[(ms, cfg)]
                if items is None:
                    row.append("")
                else:
                    pct = reweighted_pct(items, gt, x)
                    row.append("" if pct is None else f"{pct:.2f}")
            rows.append(row)
        path = OUTDIR / f"diet_effect_{x}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
