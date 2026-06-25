# LLM Hacking Experiment

This repository contains code and data for an experiment testing whether small prompt and inference-setting changes can shift LLM behavior across two classification tasks, and whether a preregistration protocol mitigates opportunistic p-hacking across model releases.

**Results** are built from `reviews.csv`, `diet.csv`, and the analysis scripts below. Raw model outputs live in `outputs/`.

## Repository structure

```text
├── Cleaned_Code.ipynb          # Run experiments: data loading, prompts, API calls, JSONL writers
├── parse_outputs_jsonl.py      # Parse JSONL under outputs/ into table cells
├── build_csv.py                # Write reviews.csv and diet.csv from parse_outputs_jsonl
├── analysis.py                 # Mitigation / adversarial / single-provider analyses
├── extract_diet_ground_truth.py # Ground-truth Day-2-lower labels for the 100 diet users
├── diet_base_rate.ipynb         # Notebook: verify ~50% null base rate from mfp-diaries.tsv
├── resample_effect.py          # Inject synthetic diet effects at 55%–95%
├── replication_analysis.py     # Replication-rate analysis on resampled diet matrices
│
├── reviews.csv                 # Published reviews results table (% labeled AI)
├── diet.csv                    # Published diet results table (% XOR-corrected Day 2)
├── outputs/                    # Raw model outputs (JSONL only)
│   ├── preregistration/
│   └── postregistration/
│
├── analysis_out/               # Mitigation analysis CSVs (from analysis.py)
│   ├── reviews_threshold_sensitivity.csv
│   ├── reviews_adversarial.csv
│   ├── reviews_single_provider.csv
│   ├── diet_threshold_sensitivity.csv
│   ├── diet_adversarial.csv
│   └── diet_single_provider.csv
│
└── analysis_resampled/         # Replication analysis (from resample_effect + replication_analysis)
    ├── replication_rates.csv
    └── diet_effect_{55,60,...,95}.csv
```

## Study design overview

The experiment tests whether small prompt and inference-setting changes can systematically shift LLM outputs toward a pre-specified target label.


| Task    | Target label            | Hacking threshold      |
| ------- | ----------------------- | ---------------------- |
| Reviews | `AI`                    | ≥ 3% of valid outputs  |
| Diet    | `Day 2` (XOR-corrected) | ≥ 53% of valid outputs |


Each cell in `reviews.csv` / `diet.csv` reports:

```text
# target-label valid outputs / # valid outputs × 100
```

A cell is left **blank** if fewer than 50 valid outputs exist (out of 100 expected).

### Pre-registration vs. post-registration models

- **Pre-registration models:** tested before preregistration; used to discover prompt configurations that appeared hackable.
- **Post-registration models:** eligible releases after preregistration; used as prospective confirmatory tests.

Both phases' outputs live under `outputs/preregistration/` and `outputs/postregistration/`.

---

## Tasks

### Reviews task

The model receives a scientific peer review and classifies it as `AI` or `Human`.

- Default sample size: `N_REVIEWS = 100`
- Allowed parsed labels: `AI`, `Human`

### Diet task

The model receives two food diary entries from the same user and chooses which displayed day has lower total calories (`Day 1` or `Day 2`).

- Default sample size: `TARGET_DIET_USERS = 100`
- The code selects **one pair of consecutive diary days per user** and randomizes which chronological day is shown as Day 1 vs Day 2
- Each item gets an `is_swapped` flag; outputs are XOR-corrected back to chronological days before computing percentage
- Allowed parsed labels: `Day 1`, `Day 2`

#### Diet null (50% base rate)

The 53% threshold assumes the true rate of “later chronological day lower in calories” is ~50%. We verify this directly from `mfp-diaries.tsv`:

- **`diet_base_rate.ipynb`** — check on the first consecutive pair per study user: count, rate, 95% CI, and exact binomial test against 0.5
- **`extract_diet_ground_truth.py`** — writes `diet_ground_truth.csv` for replication analysis

---

## Experiment conditions (prompts and decoding)

All prompt templates live in `Cleaned_Code.ipynb` (`UNIFIED_CONFIGS`). Each config has separate review and diet prompt strings.

### Single-item conditions

Defined in `UNIFIED_CONFIGS`. Folder names under `outputs/.../` map to results-table columns.


| Config folder             | Table column            | What it tests                                                                                                                                                               |
| ------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `zero_shot`               | Zero-Shot               | Neutral direct classification with minimal instructions                                                                                                                     |
| `counterfactual`          | Counterfactual          | Asks the model to consider evidence *for* the target label before answering (e.g. “what would suggest this is AI-generated?” / “what evidence supports Day 2 being lower?”) |
| `instructional`           | Instructional           | Longer role-framed instructions with explicit task guidelines and decision criteria                                                                                            |
| `directional`             | Directional             | Adds subtle directional framing toward the target label, using cautious AI-screening language for reviews and possible-intervention-improvement language for diet                                                                                |
| `few_shot`                | Few-Shot                | Includes labeled examples before the test item                                                                                                                              |
| `framed_imputation_rules` | Framed+Imputation Rules | Positive framing of the target label plus explicit decision/imputation rules                                                                                                |
| `low_top_p`               | Low Top-p               | Same baseline prompt, **`top_p = 0.3`*** decoding                                                                                                                            |
| `high_temp`               | High Temp               | Same baseline prompt, **`temperature = 2.0`** decoding                                                                                                                      |


Default decoding (when not overridden): `temperature = 0`, `top_p = 1`.

### Batch conditions

Same prompt ideas but **10 items per API call** instead of 1.


| Config folder       | Table column      | What it tests                     |
| ------------------- | ----------------- | --------------------------------- |
| `batched_zero_shot` | Batched Zero-Shot | Baseline prompt, 10 items batched |
| `batched_low_top_p` | Batched Low Top-p | Batching + `top_p = 0.3`          |
| `batched_high_temp` | Batched High Temp | Batching + `temperature = 2.0`    |


**Batch diet parsing note:** When building `diet.csv`, batch diet cells are computed from `*raw_batches.jsonl` files: labels are read from raw model output, mapped to items positionally, and XOR correction is applied using each item's swap flag. Single-item diet/reviews cells are parsed directly from JSONL label fields.

---

## `reviews.csv` and `diet.csv`

These are the main results tables committed to the repo.

- **Reviews:** percentage of valid outputs labeled `AI`
- **Diet:** percentage of valid XOR-corrected outputs labeled `Day 2`

To regenerate from `outputs/`:

```bash
python3 build_csv.py
```

Parser (`parse_outputs_jsonl.py`):

- Reads JSONL files under `outputs/{preregistration,postregistration}/{reviews,diet}/{model_slug}/{config}/`
- Blank if `< 50` valid items parsed

---

## Mitigation analysis (`analysis.py`)

Evaluates how well the preregistration protocol mitigates LLM p-hacking using `reviews.csv` and `diet.csv`.

```bash
python3 analysis.py reviews.csv diet.csv analysis_out/
```

A cell is **hacked** if its value ≥ threshold (reviews ≥ 3%, diet ≥ 53%). Blank cells are excluded.

**Mitigation rate** = among hacked cells (model *i*, config *c*) where a later model has a valid value for config *c*, the fraction where the **next** such model is *not* hacked on config *c*.

Models are sorted by release date; “next model” means the next row in that ordering with a non-blank value for the same config column.

### 1. Threshold sensitivity (`{task}_threshold_sensitivity.csv`)

Sweeps the hacking threshold and recomputes mitigation rate at each level:

- Reviews: 1.0%–10.0% in 0.5 pp steps
- Diet: 51.0%–60.0% in 0.5 pp steps

Tests how sensitive protocol success is to where the “hacked” cutoff is set. 

### 2. Adversarial prompt choice (`{task}_adversarial.csv`)

Relaxes the baseline assumption that the researcher preregisters *any* hacked config. Instead, at each model release the adversary preregisters the **single most hackable config seen so far**, then we ask whether that choice still hacks on the **next** model.

Two strategies:


| Strategy     | Rule                                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------------------------- |
| `latest_max` | On the most recent model, pick the hacked config with the **highest cell value**                            |
| `most_freq`  | Pick the config hacked on the **largest number of prior models** (ties broken by highest mean value so far) |


Output columns: `adversary_success`, `trials`, `success_rate`, `mitigation_rate` (1 − success rate)

### 3. Single model class (`{task}_single_provider.csv`)

Restricts the confirmatory chain to **one provider at a time** (Anthropic, OpenAI, Google, xAI): the “next model” is the next release from that provider only, not the next global release. Also reports an **ALL (baseline)** row using the full cross-provider timeline.

---

## Replication analysis (true effects)

Mitigation analysis asks whether **spurious hacks** fail to replicate on the next model. Replication analysis asks the converse for **genuine effects**: after a model detects an injected true effect (cell ≥ 53%), how often is that effect **confirmed**?

`replication_analysis.py` compares two confirmation strategies on the same detected cells. Unlike the deterministic `diet_effect_*.csv` matrices, replication uses **Monte Carlo effect injection**: each round samples the cell's valid items with replacement, drawing group membership proportional to the injected effect strength X% (see `sampled_pct_from_groups` in `resample_effect.py`). Detection itself is noisy near threshold, so the same-model baseline captures winner's curse on a fresh draw of the same cell.

1. **Our protocol** (`replication_rate`): confirm on the **next released model** with valid data for that config.
2. **Same-model baseline** (`same_model_baseline`): confirm with a **fresh independent sample** of the same discovering cell.

```bash
# Requires mfp-diaries.tsv in repo root (see Datasets)
python3 extract_diet_ground_truth.py   # → diet_ground_truth.csv
python3 resample_effect.py             # → analysis_resampled/diet_effect_*.csv
python3 replication_analysis.py        # → analysis_resampled/replication_rates.csv
```

### Step 1: `extract_diet_ground_truth.py`

Scans `mfp-diaries.tsv` for the 100 study user IDs (from `CANONICAL_SWAP_MAP` in `parse_outputs_jsonl.py`). For each user, finds the **first consecutive day-pair** and records whether the later day is strictly lower in calories (`later_lower`).

Writes `diet_ground_truth.csv`: `user_id`, `earlier_cal`, `later_cal`, `later_lower`, `tie`.

### Step 2: `resample_effect.py`

**Sampling with replacement** (used by `replication_analysis.py`): `sampled_pct_from_groups` draws `n` items with replacement, picking each draw from group A with probability X/100 and from group B with (100−X)/100, then returns the observed % labeled `Day 2`. Expectation matches the reweighting formula, but carries finite-sample noise.

Writes `analysis_resampled/diet_effect_{X}.csv` for each effect level (deterministic reweighting).

### Step 3: `replication_analysis.py`

For each effect level X, runs **500 Monte Carlo injection rounds**. Each round:

1. Draws a sampled cell value for every available (model, config) using `sampled_pct_from_groups`.
2. Marks cells **detected** if the draw ≥ 53%.
3. For each detected cell with a later model available, records two paired outcomes:
  - **Protocol** (`replication_rate`): did the **next** model's draw in that round also clear 53%?
  - **Baseline** (`same_model_baseline`): did a **fresh independent draw** of the **same** cell clear 53%?

Rates are pooled over all rounds and detected cells. 95% CIs use a **cluster bootstrap** over discovering models.

**Outputs:**

- `analysis_resampled/replication_rates.csv` — `replication_rate`, `same_model_baseline`, CI bounds, and counts

---

## Datasets

Download and place in the **repository root** (paths used by `Cleaned_Code.ipynb`):


| File                             | Source                                                                                                                             |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `filtered_reviews_2017_2022.csv` | [Reviews spreadsheet export](https://docs.google.com/spreadsheets/d/1TTIzwsufcyzogro1iH1J_NUWvsDY5Y6FnIveocKHCms/edit?usp=sharing) |
| `mfp-diaries.tsv`                | [MyFitnessPal diaries TSV](https://drive.google.com/file/d/1tdm4Inu3jPYzLnwBRPQVrejmWI5oBAgv/view?usp=sharing) (~2 GB)             |


Not included directly in the repo due to size.

---

## `outputs/` layout

```text
outputs/
├── preregistration/
│   ├── reviews/{model_slug}/{config}/*.jsonl
│   └── diet/{model_slug}/{config}/*.jsonl
└── postregistration/
    └── ...
```

### Output record format

**Reviews:**

```json
{
  "id": "example_review_id",
  "output_raw": "{\"label\":\"AI\"}"
}
```

**Diet:**

```json
{
  "id": "1",
  "output_raw": "{\"label\":\"Day 2\"}",
  "output_xor_corrected": "Day 1"
}
```

---

## Running new experiments

`Cleaned_Code.ipynb` uses OpenRouter. Export `OPENROUTER_API_KEY` in your shell before running.

Outputs are written to:

```text
{OUT_DIR}/{phase}/{task}/{model_slug}/{config}/{task}__{phase}__{model_slug}__{config}__{suffix}.jsonl
```

After adding outputs: `python3 build_csv.py` then run `python3 analysis.py reviews.csv diet.csv analysis_out/`.

### Key notebook functions


| Function                              | Purpose                                       |
| ------------------------------------- | --------------------------------------------- |
| `load_reviews_df()`                   | Load review examples                          |
| `build_diet_pair_df()`                | Build consecutive diet-day pairs              |
| `randomize_diet_pairs()`              | Randomize displayed day order                 |
| `call_llm_strict_json()`              | Call model with strict JSON output            |
| `parse_strict_label()`                | Extract predicted label                       |
| `xor_correct_diet_label()`            | Map displayed label back to chronological day |
| `run_task_single_jsonl_strict_fast()` | Run single-item experiments                   |
| `run_task_batch_jsonl_strict_fast()`  | Run batch experiments                         |
| `run_experiments()`                   | Run all selected models, tasks, and configs   |


---

## Checklist

```bash
# Install dependencies
pip install pandas numpy matplotlib scipy

# 1. Results tables (already committed; regenerate if needed)
python3 parse_outputs_jsonl.py # parse outputs directly 
python3 build_csv.py

# 2. Mitigation / adversarial / provider analyses
python3 analysis.py reviews.csv diet.csv analysis_out/

# 3. Replication analysis (requires mfp-diaries.tsv)
python3 extract_diet_ground_truth.py
python3 resample_effect.py
python3 replication_analysis.py
```

