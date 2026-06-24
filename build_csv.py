"""Make the final reviews.csv and diet.csv files

Does not parse raw model outputs directly
parse_outputs.py builds the in-memory result tables, and then writes those tables to reviews.csv and
diet.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from parse_outputs_jsonl import (
    BATCH,
    MODELS,
    OUTPUTS_DIR,
    ROOT,
    SINGLE,
    build_tables,
    format_pct,
)

# First 3 columns describe the model
# Remaining columns are the prompt/config result columns
HEADER = [
    "Model",
    "Provider",
    "Release Date",
    *[col for _cfg, col in SINGLE],
    *[col for _cfg, col in BATCH],
]


def table_rows(task: str, tables: Dict[str, List[Dict[str, Any]]]) -> List[List[str]]:
    """Turn parsed table cells into rows for one CSV file"""
    rows: List[List[str]] = [HEADER]

    # Lookup table to get each cell directly
    cell_by_model_config: Dict[Tuple[str, str], Dict[str, Any]] = {
        (cell["model_slug"], cell["config"]): cell
        for cell in tables[task]
    }

    # Keep the output columns in same order
    configs = [cfg for cfg, _col in SINGLE] + [cfg for cfg, _col in BATCH]

    for model_name, model_slug, provider, release_date in MODELS:
        row = [model_name, provider, release_date]

        for cfg in configs:
            cell = cell_by_model_config.get((model_slug, cfg))
            candidate = cell.get("candidate") if cell else None
            row.append(format_pct(candidate))

        rows.append(row)

    return rows


def write_csv(rows: List[List[str]], path: Path) -> None:
    """Write rows to a CSV file"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def generate_tables(outdir: Path = ROOT, outputs_dir: Path = OUTPUTS_DIR) -> None:
    """Parse the outputs folder and write reviews.csv and diet.csv."""
    outputs_dir = outputs_dir.resolve()
    outdir = outdir.resolve()

    # build_tables() parses outputs and computes percentages
    # returns an in-memory dictionary with parsed cells for both tasks
    tables = build_tables(outputs_dir=outputs_dir)

    for task in ("reviews", "diet"):
        path = outdir / f"{task}.csv"
        write_csv(table_rows(task, tables), path)
        print(f"wrote {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT,
        help="Folder where reviews.csv and diet.csv should be written",
    )

    parser.add_argument(
        "--outputs",
        type=Path,
        default=OUTPUTS_DIR,
        help="Folder containing the model output files",
    )

    args = parser.parse_args(argv)

    generate_tables(outdir=args.outdir, outputs_dir=args.outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())