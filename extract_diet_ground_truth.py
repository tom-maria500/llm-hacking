#!/usr/bin/env python3
"""Extract ground-truth 'later day lower in calories' for the 100 diet study
users, using the same first-consecutive-pair-per-user logic as
diet_base_rate.ipynb. Restricted to the study item ids so the 2GB TSV scan
only needs to track those users.

Writes diet_ground_truth.csv: user_id, earlier_cal, later_cal, later_lower, tie
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import parse_outputs_jsonl as P

TSV_PATH = "mfp-diaries.tsv"
CHUNKSIZE = 50_000

STUDY_IDS = {int(k) for k in P.CANONICAL_SWAP_MAP}  # the 100 study users


def parse_json_safe(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_number(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def extract_total_calories(totals_obj):
    if not isinstance(totals_obj, dict):
        return None
    total_list = totals_obj.get("total")
    if not isinstance(total_list, list):
        return None
    for item in total_list:
        if isinstance(item, dict) and item.get("name") == "Calories":
            return parse_number(item.get("value"))
    return None


def main():
    user_days: dict[int, list] = {}
    reader = pd.read_csv(
        TSV_PATH, sep="\t", header=None,
        names=["user_id", "date", "meals_json", "totals_json"],
        chunksize=CHUNKSIZE, dtype={"user_id": "Int64", "date": "string"},
        engine="c",
    )
    for ci, chunk in enumerate(reader):
        chunk = chunk.dropna(subset=["user_id", "date"])
        chunk["user_id"] = chunk["user_id"].astype(int)
        chunk = chunk[chunk["user_id"].isin(STUDY_IDS)]  # only study users
        if len(chunk) == 0:
            continue
        chunk["totals"] = chunk["totals_json"].apply(parse_json_safe)
        chunk["calories_total"] = chunk["totals"].apply(extract_total_calories)
        chunk["date_dt"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["date_dt", "calories_total"])
        for uid, dt, cal in zip(chunk["user_id"], chunk["date_dt"], chunk["calories_total"]):
            user_days.setdefault(int(uid), []).append((dt, float(cal)))

    rows = []
    for uid, recs in user_days.items():
        dfu = (pd.DataFrame(recs, columns=["date_dt", "calories_total"])
               .drop_duplicates(subset=["date_dt"]).sort_values("date_dt")
               .reset_index(drop=True))
        for i in range(len(dfu) - 1):
            if (dfu.loc[i + 1, "date_dt"] - dfu.loc[i, "date_dt"]).days == 1:
                earlier = dfu.loc[i, "calories_total"]
                later = dfu.loc[i + 1, "calories_total"]
                rows.append({
                    "user_id": uid,
                    "earlier_cal": earlier,
                    "later_cal": later,
                    "later_lower": bool(later < earlier),
                    "tie": bool(later == earlier),
                })
                break

    gt = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
    gt.to_csv("diet_ground_truth.csv", index=False)
    missing = sorted(STUDY_IDS - set(gt["user_id"]))
    print(f"study users: {len(STUDY_IDS)}  with ground-truth pair: {len(gt)}")
    print(f"missing (no consecutive pair found): {missing}")
    print(f"later_lower (Day 2 truly lower): {int(gt['later_lower'].sum())}/{len(gt)} "
          f"= {gt['later_lower'].mean():.4f}")
    print(f"ties: {int(gt['tie'].sum())}")
    print("wrote diet_ground_truth.csv")


if __name__ == "__main__":
    main()
