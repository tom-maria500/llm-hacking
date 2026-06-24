#!/usr/bin/env python3
"""Parse JSONL model outputs under outputs_new/ and build result-table cells in memory

Jandles two kinds of outputs:

1. Single-item runs
2. Batch runs
   - For diet batch outputs, this applies the per-item swap map
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs_new"
MIN_VALID = 50

MODELS: List[Tuple[str, str, str, str]] = [
    ("Grok 4 Fast", "grok-4-fast", "xAI", "09/19/25"),
    ("Claude Sonnet 4.5", "claude-sonnet-4.5", "Anthropic", "09/29/25"),
    ("GPT 5 Chat", "gpt-5-chat", "OpenAI", "10/06/25"),
    ("GPT 5.1 Chat", "gpt-5.1-chat", "OpenAI", "11/13/25"),
    ("Gemini 3 Pro Preview", "gemini-3-pro-preview", "Google", "11/18/25"),
    ("Grok 4.1 Fast", "grok-4.1-fast", "xAI", "11/19/25"),
    ("Claude Opus 4.5", "claude-opus-4.5", "Anthropic", "11/24/25"),
    ("GPT 5.2 Chat", "gpt-5.2-chat", "OpenAI", "12/10/25"),
    ("Gemini 3 Flash Preview", "gemini-3-flash-preview", "Google", "12/17/25"),
    ("Claude Opus 4.6", "claude-opus-4.6", "Anthropic", "02/02/26"),
    ("Claude Sonnet 4.6", "claude-sonnet-4.6", "Anthropic", "02/17/26"),
    ("Gemini 3.1 Pro Preview", "gemini-3.1-pro-preview", "Google", "02/19/26"),
    ("GPT 5.3 Chat", "gpt-5.3-chat", "OpenAI", "03/03/26"),
    ("GPT 5.4", "gpt-5.4", "OpenAI", "03/05/26"),
    ("Grok 4.20", "grok-4.20", "xAI", "03/31/26"),
    ("Claude Opus 4.7", "claude-opus-4.7", "Anthropic", "04/16/26"),
    ("GPT 5.5", "gpt-5.5", "OpenAI", "04/24/26"),
    ("Grok 4.3", "grok-4.3", "xAI", "04/30/26"),
    ("Gemini 3.5 Flash", "gemini-3.5-flash", "Google", "05/19/26"),
    ("Claude Opus 4.8", "claude-opus-4.8", "Anthropic", "05/27/26"),
]

SINGLE: List[Tuple[str, str]] = [
    ("zero_shot", "Zero-Shot"),
    ("counterfactual", "Counterfactual"),
    ("instructional", "Instructional"),
    ("directional", "Directional"),
    ("few_shot", "Few-Shot"),
    ("framed_imputation_rules", "Framed+Imputation Rules"),
    ("low_top_p", "Low Top-p"),
    ("high_temp", "High Temp"),
]

BATCH: List[Tuple[str, str]] = [
    ("batched_zero_shot", "Batched Zero-Shot"),
    ("batched_low_top_p", "Batched Low Top-p"),
    ("batched_high_temp", "Batched High Temp"),
]

# For each diet item ID for raw batches:
#   False means the raw parsed label is already is same position 
#   True means the raw parsed label needs to be flipped
CANONICAL_SWAP_MAP: Dict[str, bool] = {
    "1": False,
    "2": True,
    "3": False,
    "4": False,
    "5": True,
    "6": False,
    "7": False,
    "8": False,
    "10": True,
    "11": True,
    "12": True,
    "13": False,
    "15": False,
    "16": False,
    "17": True,
    "18": True,
    "19": False,
    "20": True,
    "21": False,
    "22": False,
    "23": False,
    "24": True,
    "25": False,
    "26": False,
    "29": False,
    "30": True,
    "31": True,
    "32": True,
    "33": True,
    "34": False,
    "35": False,
    "36": False,
    "37": True,
    "38": True,
    "40": True,
    "41": True,
    "42": True,
    "43": True,
    "44": True,
    "45": False,
    "46": True,
    "47": False,
    "48": False,
    "49": True,
    "50": False,
    "51": False,
    "52": True,
    "53": True,
    "54": False,
    "56": True,
    "57": True,
    "59": True,
    "60": False,
    "62": False,
    "63": False,
    "64": False,
    "65": True,
    "66": False,
    "67": True,
    "68": True,
    "70": False,
    "71": True,
    "72": False,
    "73": False,
    "74": False,
    "75": False,
    "76": False,
    "77": True,
    "78": True,
    "79": True,
    "81": True,
    "82": True,
    "84": False,
    "85": True,
    "86": True,
    "88": True,
    "89": True,
    "90": False,
    "91": False,
    "92": False,
    "93": False,
    "94": True,
    "95": False,
    "98": True,
    "99": True,
    "100": True,
    "101": False,
    "102": True,
    "103": True,
    "104": False,
    "105": True,
    "106": False,
    "107": True,
    "108": True,
    "109": True,
    "110": False,
    "111": True,
    "112": True,
    "113": True,
    "114": False,
}


def parse_reviews_label(value: Optional[object]) -> Optional[str]:
    """Parse a reviews-task output into either 'AI', 'Human', or None
    The reviews task expects the model to label a review as AI-written or
    human-written."""

    if value is None:
        return None
    match = re.search(r'"label"\s*:\s*"([^"]+)"', str(value))
    label = (match.group(1) if match else str(value)).strip().strip('"').lower()
    if label == "ai":
        return "AI"
    if label == "human":
        return "Human"
    return None


def parse_diet_label(value: Optional[object]) -> Optional[str]:
    """Parse a diet-task output into either 'Day 1', 'Day 2', or None

    The diet task expects the model to choose between Day 1 and Day 2
    """
    if value is None:
        return None
    label = re.sub(r"\s+", " ", str(value)).strip().lower()
    if "day 2" in label:
        return "Day 2"
    if "day 1" in label:
        return "Day 1"
    return None


def flip_diet_label(label: Optional[str]) -> Optional[str]:
    """Flip a diet label from Day 1 to Day 2, or Day 2 to Day 1.

    Used for diet batch outputs when an item was swapped.
    """
    if label == "Day 2":
        return "Day 1"
    if label == "Day 1":
        return "Day 2"
    return None


def read_jsonl(path: str | Path) -> List[dict]:
    """Read a JSONL file into a list of dictionaries
    JSONL means one JSON object per line
    Invalid JSON lines are skipped 
    """
    records: List[dict] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def relative(path: str | Path, root: Path = ROOT) -> str:
    return os.path.relpath(path, root)


def single_json_records(
    outputs_dir: Path, task: str, model: str, cfg: str
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Load single-item JSONL records for one task/model/config"""
    records: Dict[str, dict] = {}
    sources: Dict[str, str] = {}
    for phase in ("preregistration", "postregistration"):
        directory = outputs_dir / phase / task / model / cfg
        strict = glob.glob(str(directory / "*strict_json.jsonl"))
        if strict:
            for record in read_jsonl(strict[0]):
                item_id = str(record.get("id"))
                if item_id not in records:
                    records[item_id] = record
                    sources[item_id] = strict[0]
        else:
            for suffix in ("minimal.jsonl", "minimal_extra.jsonl"):
                for path in glob.glob(str(directory / f"*__{suffix}")):
                    for record in read_jsonl(path):
                        item_id = str(record.get("id"))
                        if item_id not in records:
                            records[item_id] = record
                            sources[item_id] = path
    return records, sources


def label_for_single_record(task: str, record: dict) -> Tuple[object, object, Optional[str], Optional[str]]:
    """Get the raw label and the final table label for a single-item record"""
    if task == "reviews":
        raw_value = record.get("output_raw") if "output_raw" in record else record.get("output")
        label = parse_reviews_label(raw_value)
        return raw_value, raw_value, label, label

    # Diet single-item outputs:
    #   output_raw is the original model response
    #   output_xor_corrected is the corrected value used for the table
    raw_value = record.get("output_raw")
    source_value = record.get("output_xor_corrected") or record.get("output")
    raw_label = parse_diet_label(raw_value)
    table_label = parse_diet_label(source_value)
    return source_value, raw_value, raw_label, table_label


def single_candidate(
    outputs_dir: Path,
    task: str,
    model: str,
    cfg: str,
    root: Path = ROOT,
) -> Optional[Dict[str, Any]]:
    """Compute one single-item table cell

    A table cell corresponds to: one task + one model + one config
    The percentage is: target_count / valid_count * 100

    Target labels:
        reviews -> "AI"
        diet    -> "Day 2"

    Returns None if:
        - no records are found
        - fewer than MIN_VALID records are found
    """
    records, sources = single_json_records(outputs_dir, task, model, cfg)
    if not records:
        return None

    item_rows = []
    valid = target = 0
    target_label = "AI" if task == "reviews" else "Day 2"
    for item_id, record in records.items():
        source_value, raw_value, raw_label, table_label = label_for_single_record(task, record)
        if table_label:
            valid += 1
            target += table_label == target_label
        # Store row-level details 
        item_rows.append({
            "item_id": item_id,
            "source_file": relative(sources[item_id], root),
            "source_kind": "json",
            "source_output": source_value,
            "raw_output": raw_value,
            "parsed_raw_label": raw_label,
            "parsed_table_label": table_label,
            "batch_id": "",
            "prediction_position": "",
            "is_valid": bool(table_label),
        })

    if valid < MIN_VALID:
        return None

    return {
        "source_kind": "json",
        "source_files": sorted({row["source_file"] for row in item_rows}),
        "pct": 100 * target / valid if valid else None,
        "valid": valid,
        "target": target,
        "item_rows": item_rows,
    }


def parse_batch_predictions(raw: str) -> List[Optional[str]]:
    """Parse the list of labels from a raw batch output.

    Expected format is usually JSON like:

        {
            "predictions": [
                {"label": "AI"},
                {"label": "Human"}
            ]
        }
    """
    try:
        predictions = json.loads(raw).get("predictions", [])
        return [prediction.get("label") for prediction in predictions]
    except Exception:
        return [match.group(1) for match in re.finditer(r'"label"\s*:\s*"([^"]+)"', raw)]


def batch_candidate(
    outputs_dir: Path,
    task: str,
    model: str,
    cfg: str,
    swap_map: Dict[str, bool] = CANONICAL_SWAP_MAP,
    root: Path = ROOT,
) -> Optional[Dict[str, Any]]:
    """Compute one batch table cell

    A table cell corresponds to: one task + one model + one batch config

    For batch outputs, each JSONL line contains a batch with multiple item IDs
    and multiple predictions. This function maps each prediction back to the
    corresponding item ID

    For diet only: the fixed swap_map is used to flip labels for swapped items

    Returns None if:
        - no raw_batches.jsonl file is found
        - fewer than MIN_VALID valid item predictions are parsed
    """
    path = None
    for phase in ("preregistration", "postregistration"):
        paths = glob.glob(str(outputs_dir / phase / task / model / cfg / "*raw_batches.jsonl"))
        if paths:
            path = paths[0]
            break
    if not path:
        return None

    per_item: Dict[str, dict] = {}
    for batch in read_jsonl(path):
        ids = batch.get("item_ids") or []
        labels = parse_batch_predictions((batch.get("output_raw") or "").strip())
        for position, (item_id, label_value) in enumerate(zip(ids, labels), start=1):
            item_id = str(item_id)
            if item_id in per_item:
                continue
            raw_label = (
                parse_reviews_label(label_value) if task == "reviews" else parse_diet_label(label_value)
            )
            if task == "diet":
                table_label = (
                    flip_diet_label(raw_label) if swap_map.get(item_id, False) else raw_label
                )
            else:
                table_label = raw_label
            per_item[item_id] = {
                "item_id": item_id,
                "source_file": relative(path, root),
                "source_kind": "raw_batches",
                "source_output": label_value,
                "raw_output": label_value,
                "parsed_raw_label": raw_label,
                "parsed_table_label": table_label,
                "batch_id": batch.get("batch_id", ""),
                "prediction_position": position,
                "is_valid": bool(table_label),
            }

    valid = sum(1 for row in per_item.values() if row["is_valid"])
    if valid < MIN_VALID:
        return None

    target_label = "AI" if task == "reviews" else "Day 2"
    target = sum(1 for row in per_item.values() if row["parsed_table_label"] == target_label)
    return {
        "source_kind": "raw_batches",
        "source_files": [relative(path, root)],
        "pct": 100 * target / valid if valid else None,
        "valid": valid,
        "target": target,
        "item_rows": list(per_item.values()),
    }


def build_tables(
    outputs_dir: Path = OUTPUTS_DIR,
    root: Path = ROOT,
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse all table cells for reviews and diet
    Returns lists of cell dicts"""
    outputs_dir = outputs_dir.resolve()
    tables: Dict[str, List[Dict[str, Any]]] = {"reviews": [], "diet": []}

    for task in ("reviews", "diet"):
        for model_name, model_slug, provider, release_date in MODELS:
            for cfg, column in SINGLE:
                candidate = single_candidate(outputs_dir, task, model_slug, cfg, root=root)
                tables[task].append({
                    "task": task,
                    "model": model_name,
                    "model_slug": model_slug,
                    "provider": provider,
                    "release_date": release_date,
                    "config": cfg,
                    "table_column": column,
                    "candidate": candidate,
                })
            for cfg, column in BATCH:
                candidate = batch_candidate(outputs_dir, task, model_slug, cfg, root=root)
                tables[task].append({
                    "task": task,
                    "model": model_name,
                    "model_slug": model_slug,
                    "provider": provider,
                    "release_date": release_date,
                    "config": cfg,
                    "table_column": column,
                    "candidate": candidate,
                })

    return tables


def format_pct(candidate: Optional[Dict[str, Any]]) -> str:
    """Format as percentage string for the CSV."""
    if not candidate or candidate.get("pct") is None:
        return ""
    return f"{candidate['pct']:.2f}"


if __name__ == "__main__":
    tables = build_tables()
    # Prints how many nonblank cells were parsed for each task.
    for task in ("reviews", "diet"):
        cells = [c for c in tables[task] if c["candidate"]]
        print(f"{task}: {len(cells)} nonblank cells parsed")
