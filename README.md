# LLM Hacking Experiment

This repository contains code and data for an experiment testing whether small prompt and inference-setting changes can shift LLM behavior across two classification tasks.

**Results are reported in the paper.** This repo contains the experiment notebook, raw model outputs, results tables, and mitigation analysis code.

## Repository Contents

```text
├── Cleaned_Code.ipynb        # Run experiments 
├── analysis.py               # Mitigation analysis on the results tables
├── analysis_out_correceted/  # Output CSVs from analysis.py
├── diet.csv                  # Diet results table (XOR-corrected Day 2 %)
├── reviews.csv               # Reviews results table (% labeled AI)
└── outputs/                  # Raw JSONL/CSV outputs by model and config
    ├── preregistration/
    └── postregistration/
```

## Study Design Overview

This experiment tests whether small prompt and inference-setting changes can systematically shift LLM outputs toward a pre-specified target label.

The two target labels are:

- **Reviews task:** `AI`
- **Diet task:** `Day 2`

For each model, the same set of prompt configurations and decoding settings are tested. The main question is whether any configuration increases the rate of target-label responses to or above the pre-specified hacking threshold.

The pre-specified hacking thresholds are:

- **Reviews:** hacking if at least 3% of valid outputs are labeled `AI`
- **Diet:** hacking if at least 53% of valid XOR-corrected outputs are labeled `Day 2`

## Main Files

### `Cleaned_Code.ipynb`

Main notebook for running the experiment. It includes data loading, preprocessing, prompt templates, experiment configs, model calls, JSONL writing, and single-item/batch runners.

### `reviews.csv` and `diet.csv`

Each cell reports:

```text
# target-label valid outputs / # valid outputs × 100
```

- **Reviews:** percentage of valid outputs labeled `AI`
- **Diet:** percentage of valid XOR-corrected outputs labeled `Day 2`

A valid output parses into one of the allowed labels for that task. If fewer than 50 valid outputs exist for a cell (out of 100 expected), the cell is left blank.

**Batch diet cells:** The in-notebook batch parser (`parse_batch_predictions`) often fails for diet because models return prediction ids inconsistently across providers (e.g. displayed item numbers vs. `user_id`s). When building the results tables, batch diet values were therefore computed from `raw_batches.jsonl` instead: labels were read from the raw model output, mapped to items positionally, and XOR correction was applied manually using each item’s `is_swapped` flag from the experiment data. This is not an issue with the LLM call itself. 

### `analysis.py`

Evaluates how well the preregistration protocol mitigates LLM p-hacking using `reviews.csv` and `diet.csv`. Models are ordered by release date. A cell is **hacked** if its value is at or above the task threshold (reviews ≥ 3%, diet ≥ 53%). Blank cells are excluded.

**Mitigation rate** = among hacked cells where a later model has valid data for that config, the fraction where the **next** such model is *not* hacked.

The script runs three analyses:

1. **Threshold sensitivity** — Sweeps the hacking threshold (reviews 1.0–10.0%, step 0.5; diet 51.0–60.0%, step 0.5) and reports the mitigation rate at each level. Tests how sensitive the protocol’s success is to where the “hacked” cutoff is set.
2. **Adversarial prompt choice** — At each model release, assumes the researcher preregisters the single most hackable config seen so far, then checks whether that choice still hacks on the next model. Two strategies: `latest_max` (highest hacked value on the most recent model) and `most_freq` (config hacked on the most prior models; ties broken by mean value).
3. **Single model class** — Same baseline mitigation check as above, but restricted to each provider’s own release timeline (e.g. only Anthropic → next Anthropic), plus an aggregate across all providers.

```bash
pip install pandas numpy
python analysis.py reviews.csv diet.csv analysis_out/
```

`analysis_out/` contains six CSVs: `{task}_threshold_sensitivity.csv`, `{task}_adversarial.csv`, and `{task}_single_provider.csv` for each task.

## Tasks

### Reviews Task

The model receives a product review and classifies it as `AI` or `Human`.

Default sample size: `N_REVIEWS = 100`

### Diet Task

The model receives two food diary entries from the same user and chooses which displayed day has lower total calories (`Day 1` or `Day 2`).

The code selects one pair of consecutive diary days per user and randomizes which chronological day is shown as Day 1. Each item gets an `is_swapped` flag. Diet outputs include `output_xor_corrected` to map the model’s displayed-label answer back to the chronological day.

Default sample size: `TARGET_DIET_USERS = 100`

## Datasets Used

- Reviews Dataset: [https://docs.google.com/spreadsheets/d/1TTIzwsufcyzogro1iH1J_NUWvsDY5Y6FnIveocKHCms/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1TTIzwsufcyzogro1iH1J_NUWvsDY5Y6FnIveocKHCms/edit?usp=sharing)
- Diet Dataset: [https://drive.google.com/file/d/1tdm4Inu3jPYzLnwBRPQVrejmWI5oBAgv/view?usp=sharing](https://drive.google.com/file/d/1tdm4Inu3jPYzLnwBRPQVrejmWI5oBAgv/view?usp=sharing)

## Pre-registration vs. Post-registration Models

- **Pre-registration models:** tested before preregistration; used to identify prompt configurations that appeared to produce hacking effects.
- **Post-registration models:** eligible models released after preregistration; used as a prospective test of whether those effects generalize.

## Model Inclusion and Exclusion Criteria

Models are selected from OpenAI, Anthropic/Claude, Google/Gemini, and xAI/Grok via OpenRouter. Included models are general-purpose mid-tier or higher. Excluded: `mini`/`lite`/`nano` variants, code/codex models, embedding/image/audio models, models under 7B parameters, and special-purpose variants.

## Experiment Conditions

### Single-Item Conditions

Defined in `UNIFIED_CONFIGS` in `Cleaned_Code.ipynb`.


| Config folder (`outputs/.../`) | Results table column    | What it tests                                                                                       |
| ------------------------------ | ----------------------- | --------------------------------------------------------------------------------------------------- |
| `zero_shot`                    | Zero-Shot               | Neutral direct classification with minimal instructions                                             |
| `counterfactual`               | Counterfactual          | Whether asking the model to consider evidence for the target label shifts outputs toward that label |
| `instructional`                | Instructional           | Whether more detailed task instructions change outputs                                              |
| `directional`                  | Directional             | Whether framing the target label as plausible or expected shifts outputs toward that label          |
| `few_shot`                     | Few-Shot                | Whether providing examples changes classification behavior                                          |
| `framed_imputation_rules`      | Framed+Imputation Rules | Whether positive framing plus imputation/decision rules shifts outputs                              |
| `low_top_p`                    | Low Top-p               | Whether lower top-p decoding changes outputs                                                        |
| `high_temp`                    | High Temp               | Whether high-temperature decoding changes outputs                                                   |


### Batch Conditions


| Config folder (`outputs/.../`) | Results table column | What it tests                                             |
| ------------------------------ | -------------------- | --------------------------------------------------------- |
| `batched_zero_shot`            | Batched Zero-Shot    | Whether giving the model 10 items at once changes outputs |
| `batched_low_top_p`            | Batched Low Top-p    | Whether batching plus low top-p changes outputs           |
| `batched_high_temp`            | Batched High Temp    | Whether batching plus high temperature changes outputs    |


## Models

### Preregistration Models

- `x-ai/grok-4-fast`
- `anthropic/claude-sonnet-4.5`
- `openai/gpt-5-chat`
- `openai/gpt-5.1-chat`
- `google/gemini-3-pro-preview`
- `x-ai/grok-4.1-fast`
- `anthropic/claude-opus-4.5`
- `openai/gpt-5.2-chat`
- `google/gemini-3-flash-preview`
- `anthropic/claude-opus-4.6`
- `anthropic/claude-sonnet-4.6`
- `google/gemini-3.1-pro-preview`
- `openai/gpt-5.3-chat`

### Postregistration Models

- `x-ai/grok-4.20`
- `anthropic/claude-opus-4.7`
- `x-ai/grok-4.3`
- `google/gemini-3.5-flash`
- `anthropic/claude-opus-4.8`

## Outputs

Experiment files live under `outputs/preregistration/` and `outputs/postregistration/`, split by task (`diet` or `reviews`), model, and config folder.

Typical files per config:


| Suffix             | Description                                     |
| ------------------ | ----------------------------------------------- |
| `strict_json`      | Pre-registration single-item JSONL              |
| `minimal`          | Post-registration single-item JSONL             |
| `baseline10_items` | Item-level batch predictions, usually 100 rows  |
| `raw_batches`      | Raw batch API responses, usually 10 rows/batch  |
| `downloads_csv`    | CSV source used for some pre-registration cells |


Filename pattern:

```text
{task}__{phase}__{model}__{config}__{suffix}.jsonl
```

### Output Format

**Reviews** — allowed labels: `AI`, `Human`

```json
{
  "id": "example_review_id",
  "output_raw": "{\"label\":\"AI\"}"
}
```

**Diet** — allowed labels: `Day 1`, `Day 2`

```json
{
  "id": "1",
  "output_raw": "{\"label\":\"Day 2\"}",
  "output_xor_corrected": "Day 1"
}
```

## Running the Experiment

The notebook is designed to run in Google Colab using OpenRouter.

```python
MODELS = ["google/gemini-3.5-flash"]
OUT_DIR = "outputs"

all_run_outputs = run_experiments(
    models=MODELS,
    out_dir=OUT_DIR,
    single_config_names=[
        "zero_shot", "counterfactual", "instructional", "directional",
        "few_shot", "framed_imputation_rules", "low_top_p", "high_temp",
    ],
    batch_configs=[BATCH_CONFIG, BATCH_HIGH_TEMP_CONFIG, BATCH_LOW_P_CONFIG],
    single_limit=None,
    batch_limit=None,
    skip_existing=True,
)
```

Outputs are written to:

```text
{OUT_DIR}/{phase}/{task}/{model_slug}/{config_folder}/{task}__{phase}__{model_slug}__{config_folder}__{suffix}.jsonl
```

## Key Functions


| Function                              | Purpose                                      |
| ------------------------------------- | -------------------------------------------- |
| `load_reviews_df()`                   | Loads review examples                        |
| `build_diet_pair_df()`                | Builds consecutive diet-day pairs            |
| `randomize_diet_pairs()`              | Randomizes displayed diet day order          |
| `call_llm_strict_json()`              | Calls the model with strict JSON output      |
| `parse_strict_label()`                | Extracts the predicted label                 |
| `xor_correct_diet_label()`            | Corrects diet labels after randomization     |
| `run_task_single_jsonl_strict_fast()` | Runs single-item experiments                 |
| `run_task_batch_jsonl_strict_fast()`  | Runs batch experiments                       |
| `run_experiments()`                   | Runs all selected models, tasks, and configs |


