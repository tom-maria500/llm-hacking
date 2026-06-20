#!/usr/bin/env python3
"""Analyses of the protocol's mitigation of LLM p-hacking, run on the two
results CSVs (reviews.csv, diet.csv).

A cell (model, config) is "hacked" if its value >= threshold
(default: 3.0 for reviews, 53.0 for diet). NA cells are excluded.

Analyses:
 1. Threshold sensitivity: mitigation rate as the hacking threshold varies.
 2. Adversarial prompt choice: instead of any hacked config on the previous
    model, the researcher preregisters the single most hackable config so far
    (two variants: highest value on the latest model; most frequently hacked
    across all models so far).
 3. Single model class: the preregistered eligible set is one provider, so
    the confirmatory model is the next release from that provider only.

Mitigation rate = among hacked cells (model i, config c) for which a later
model has a valid value for config c, the fraction where the *next* such
model is not hacked on config c.

Usage: python analysis.py reviews.csv diet.csv [outdir]

Vibe coded using Fable 5 by Nihar Shah.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = {"reviews": 3.0, "diet": 53.0}
THRESHOLD_GRID = {
    "reviews": np.arange(1.0, 10.5, 0.5),
    "diet": np.arange(51.0, 60.5, 0.5),  # 50% = null (no preference); thresholds start above it
}
ID_COLS = ["Model", "Provider", "Release Date"]


def load(path):
    df = pd.read_csv(path)
    df["Release Date"] = pd.to_datetime(df["Release Date"], format="%m/%d/%y")
    return df.sort_values("Release Date").reset_index(drop=True)


def value_matrix(df):
    return df.drop(columns=ID_COLS).astype(float)  # NA -> NaN


def next_valid(vals, i, c):
    """Index of the first model after i with a valid value for config c."""
    col = vals.iloc[i + 1:, c]
    col = col[col.notna()]
    return col.index[0] if len(col) else None


def mitigation_rate(df, thr):
    """Baseline protocol: for every hacked cell, does the next valid model
    stay below the threshold? Returns (rate, n_prevented, n_evaluable,
    n_hacked_total)."""
    vals = value_matrix(df)
    hacked = vals.ge(thr)
    n_hacked = int(hacked.sum().sum())
    prevented = evaluable = 0
    for i in range(len(vals)):
        for c in range(vals.shape[1]):
            if not hacked.iat[i, c]:
                continue
            j = next_valid(vals, i, c)
            if j is None:
                continue
            evaluable += 1
            prevented += not hacked.iat[j, c]
    rate = prevented / evaluable if evaluable else np.nan
    return rate, prevented, evaluable, n_hacked


def threshold_sensitivity(df, name):
    rows = []
    for thr in THRESHOLD_GRID[name]:
        rate, prev, ev, nh = mitigation_rate(df, thr)
        rows.append({"threshold": thr, "hacked_cells": nh,
                     "evaluable": ev, "prevented": prev,
                     "mitigation_rate": rate})
    return pd.DataFrame(rows)


def adversarial_choice(df, thr):
    """At each release point i (preregistration happens just after model i),
    the adversary picks one config and we test it on the next model.

    Strategies:
      latest_max : config with the highest value on model i (must be hacked)
      most_freq  : config hacked on the largest number of models 0..i
                   (ties broken by highest mean value so far)
    Returns a DataFrame of per-strategy success counts.
    """
    vals = value_matrix(df)
    hacked = vals.ge(thr)
    out = {s: {"success": 0, "trials": 0} for s in ("latest_max", "most_freq")}
    for i in range(len(vals) - 1):
        past_hacked = hacked.iloc[: i + 1]
        if not past_hacked.values.any():
            continue  # nothing hackable to preregister yet

        # latest_max: best config on the most recent model, if any is hacked
        if hacked.iloc[i].any():
            c = int(np.nanargmax(vals.iloc[i].where(hacked.iloc[i]).values))
            j = next_valid(vals, i, c)
            if j is not None:
                out["latest_max"]["trials"] += 1
                out["latest_max"]["success"] += bool(hacked.iat[j, c])

        # most_freq: config with most hacked models so far
        counts = past_hacked.sum()
        best = counts[counts == counts.max()].index
        if len(best) > 1:  # tie-break on mean value so far
            best = [vals[best].iloc[: i + 1].mean().idxmax()]
        c = vals.columns.get_loc(best[0])
        j = next_valid(vals, i, c)
        if j is not None:
            out["most_freq"]["trials"] += 1
            out["most_freq"]["success"] += bool(hacked.iat[j, c])

    rows = []
    for s, d in out.items():
        rate = d["success"] / d["trials"] if d["trials"] else np.nan
        rows.append({"strategy": s, "adversary_success": d["success"],
                     "trials": d["trials"], "success_rate": rate,
                     "mitigation_rate": 1 - rate if d["trials"] else np.nan})
    return pd.DataFrame(rows)


def single_provider(df, thr):
    """Restrict the eligible set to one provider: the confirmatory model is
    the next release from that provider. Mitigation rate per provider."""
    rows = []
    for prov, sub in df.groupby("Provider"):
        sub = sub.sort_values("Release Date").reset_index(drop=True)
        rate, prev, ev, nh = mitigation_rate(sub, thr)
        rows.append({"provider": prov, "models": len(sub),
                     "hacked_cells": nh, "evaluable": ev,
                     "prevented": prev, "mitigation_rate": rate})
    allr, allp, alle, allh = mitigation_rate(df, thr)
    rows.append({"provider": "ALL (baseline)", "models": len(df),
                 "hacked_cells": allh, "evaluable": alle,
                 "prevented": allp, "mitigation_rate": allr})
    return pd.DataFrame(rows)


def main():
    paths = {"reviews": Path(sys.argv[1]), "diet": Path(sys.argv[2])}
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 120)

    for name, path in paths.items():
        df = load(path)
        thr = DEFAULT_THRESHOLD[name]
        print(f"\n{'=' * 70}\n{name.upper()} experiment (default threshold >= {thr})\n{'=' * 70}")

        sens = threshold_sensitivity(df, name)
        sens.to_csv(outdir / f"{name}_threshold_sensitivity.csv", index=False)
        print("\n[1] Threshold sensitivity")
        print(sens.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        adv = adversarial_choice(df, thr)
        adv.to_csv(outdir / f"{name}_adversarial.csv", index=False)
        print("\n[2] Adversarial prompt choice (at default threshold)")
        print(adv.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        prov = single_provider(df, thr)
        prov.to_csv(outdir / f"{name}_single_provider.csv", index=False)
        print("\n[3] Single model class (at default threshold)")
        print(prov.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
