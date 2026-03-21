# Evaluation Protocol

## Objective

Compute transparent and reproducible multi-label metrics using manually validated data.

## Authoritative inputs

- Gold labels: `data/manually_validated_gold_subset.csv`
- Predictions: `evaluation/predictions_template.csv`
- Taxonomy: `data/label_taxonomy.csv`

## Eligible rows for final metrics

A row is evaluable only if:

- `inclusion_flag == INCLUDE`
- `validation_status` in `{VALIDATED, ADJUDICATED}`
- `final_adjudicated_labels` is non-empty
- prediction row exists for same `record_id`
- labels are valid under taxonomy

## Excluded rows

Rows are tracked as:

- `invalid_rows` (schema/taxonomy problems)
- `skipped_rows` (not included / not validated)
- `unlabeled_rows` (no final adjudicated labels)

## Metrics

- Per-class: precision, recall, F1, support
- Aggregate: micro/macro/weighted precision/recall/F1
- Diagnostic outputs: mismatch report, invalid/skipped/unlabeled summaries

## Outputs

- `evaluation/evaluation_metrics.json`
- `evaluation/evaluation_metrics.csv`
- `evaluation/mismatch_report.csv`

## Cross-check mode

Use `evaluation/f1_recompute_template.xlsx` to verify formulas and manually audit key rows.

## Scientific language constraints

- Weak labels are not final gold truth.
- Final thesis claims should explicitly state sample size and manual-validation coverage.
