# Gold-Standard Assessment Package (Thesis Refactor)

This package separates weak supervision from manually validated gold evidence and provides reproducible supervisor-grade evaluation workflows.

## Methodological layers

1. Weak labels (silver-standard): ontology/rule-derived outputs.
2. Manually validated subset: human-reviewed records with adjudicated final labels.
3. Predictions: model outputs aligned by `record_id`.
4. Evaluation outputs: metrics, mismatches, invalid/skipped-row audits.

Important: weak labels are not equivalent to final gold truth.

## Folder map

- `START_HERE.md`: rapid orientation for supervisors.
- `SUPERVISOR_RECOMPUTE_CHECKLIST.md`: step-by-step recomputation checklist.
- `annotation_guidelines.md`: annotation and adjudication protocol.
- `data/`: weak labels, taxonomy, manual-validation subset, adjudication queue.
- `evaluation/`: prediction template, Excel workbooks, protocol and metric definitions.
- `reports/`: summary, dataset profile, validation status, gap analysis.
- `schemas/`: CSV schema constraints.
- `scripts/`: deterministic generation, validation, report, and evaluation pipelines.

## Authoritative artifacts

- Weak labels:
  - `data/weak_labels_abstracts.csv`
  - `data/weak_labels_scenarios.csv`
- Manual validation:
  - `data/manually_validated_gold_subset.csv`
  - `data/adjudication_queue.csv`
- Evaluation:
  - `evaluation/predictions_template.csv`
  - `evaluation/f1_recompute_template.xlsx`
  - `evaluation/confusion_matrix_template.xlsx`
  - `evaluation/evaluation_metrics.json` (after recomputation)
  - `evaluation/evaluation_metrics.csv` (after recomputation)
  - `evaluation/mismatch_report.csv` (after recomputation)

## Regeneration commands

From repository root:

```powershell
C:/Users/ahmad/AppData/Local/Programs/Python/Python312/python.exe gold_standard_assessment/scripts/run_all.py --sample-size 250 --seed 42
```

## Evaluation recomputation command

```powershell
C:/Users/ahmad/AppData/Local/Programs/Python/Python312/python.exe gold_standard_assessment/scripts/run_evaluation.py
```

## Formula and assumptions transparency

- Metric definitions: `evaluation/metrics_definition.md`
- End-to-end evaluation protocol: `evaluation/evaluation_protocol.md`
- Excel formula audit path: `evaluation/f1_recompute_template.xlsx` (`Supervisor_Audit` sheet)

## Scientific validity statement

- Backend-aligned mappings are preserved for reproducibility and auditability.
- Final thesis claims should be based on INCLUDE + VALIDATED/ADJUDICATED records from `manually_validated_gold_subset.csv`.
- If manual validation coverage is limited, findings must be reported as preliminary.
