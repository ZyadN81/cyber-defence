# SUPERVISOR RECOMPUTE CHECKLIST

## Preparation

- Confirm Python environment is available.
- Confirm package files exist under `gold_standard_assessment/`.

## Regenerate deterministic artifacts

- Run:
  - `C:/Users/ahmad/AppData/Local/Programs/Python/Python312/python.exe gold_standard_assessment/scripts/run_all.py --sample-size 250 --seed 42`
- Confirm outputs:
  - `data/weak_labels_abstracts.csv`
  - `data/weak_labels_scenarios.csv`
  - `data/manually_validated_gold_subset.csv`
  - `evaluation/predictions_template.csv`
  - `evaluation/f1_recompute_template.xlsx`
  - `reports/gold_standard_summary.json`

## Manual-validation checkpoints

- Open `data/manually_validated_gold_subset.csv`.
- Fill annotator fields and final adjudicated fields.
- Set `inclusion_flag` to `INCLUDE` for rows used in final metrics.
- Set `validation_status` to `VALIDATED` or `ADJUDICATED` for completed rows.

## Prediction import

- Fill `evaluation/predictions_template.csv` with model outputs for the same `record_id` values.
- Use semicolon-separated labels.

## Recompute metrics

- Run:
  - `C:/Users/ahmad/AppData/Local/Programs/Python/Python312/python.exe gold_standard_assessment/scripts/run_evaluation.py`
- Verify output artifacts:
  - `evaluation/evaluation_metrics.json`
  - `evaluation/evaluation_metrics.csv`
  - `evaluation/mismatch_report.csv`

## Cross-check in Excel

- Open `evaluation/f1_recompute_template.xlsx`.
- Verify formulas and per-class sheets.
- Confirm micro/macro metrics are consistent with JSON/CSV outputs.

## Reporting guardrails

- If included manually validated rows are low, report findings as preliminary.
- Do not describe weak labels as gold truth.
