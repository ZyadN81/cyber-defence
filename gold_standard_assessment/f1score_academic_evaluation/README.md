# Section 4.7 Academic Evidence Package

This package provides a clean, thesis-ready, reproducible evaluation workflow for Section 4.7.

## Design Principles

- Manual gold labels are mandatory for final metrics.
- Every F1 is traceable to TP/TN/FP/FN counts.
- Every thesis table can be linked to concrete input and output artifacts.
- Outputs are generated in structured CSV/JSON files suitable for appendices.

## Folder Architecture

- `input/`: human-curated gold labels and optional notes.
- `config/`: scenario and concept mapping artifacts.
- `scripts/`: deterministic evaluation logic.
- `outputs/`: generated matrices, confusion tables, and metrics.
- `docs/`: data dictionary and examiner navigation notes.

## Required Input

Create and fill:

- `input/f1score_manual_gold_matrix.csv`

Expected columns:

- `record_id`
- `simple_1`, `simple_2`, `simple_3`, `simple_4`, `simple_5`
- `advanced_1`, `advanced_2`, `advanced_3`, `advanced_4`, `advanced_5`

Values must be `0` or `1`.

## Run

From repository root:

```powershell
python gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py
```

If `f1score_manual_gold_matrix.csv` does not exist, the script generates a template and stops.

## Main Outputs

- `outputs/f1score_gold_matrix_resolved.csv`
- `outputs/f1score_scenario_concept_mapping.csv`
- `outputs/f1score_keyword_hits/*.csv`
- `outputs/f1score_predictions_by_method.csv`
- `outputs/f1score_confusion_by_scenario_method.csv`
- `outputs/f1score_method_metrics_summary.csv`
- `outputs/f1score_traceability_manifest.json`
- `docs/f1score_data_dictionary.md`
- `docs/f1score_examiner_guide.md`
