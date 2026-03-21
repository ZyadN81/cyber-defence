# START HERE

This package is designed for thesis-grade transparency and supervisor recomputation.

## Fast orientation

1. Weak labels (silver-standard) live in `data/weak_labels_*.csv`.
2. Manual-validation scaffold lives in `data/manually_validated_gold_subset.csv`.
3. Evaluation templates live in `evaluation/`.
4. Final computed metrics are written to `evaluation/evaluation_metrics.json` and `evaluation/evaluation_metrics.csv`.
5. Status and gap reports live in `reports/`.

## One-command regeneration

From repository root:

```powershell
C:/Users/ahmad/AppData/Local/Programs/Python/Python312/python.exe gold_standard_assessment/scripts/run_all.py --sample-size 250 --seed 42
```

## One-command evaluation recomputation

```powershell
C:/Users/ahmad/AppData/Local/Programs/Python/Python312/python.exe gold_standard_assessment/scripts/run_evaluation.py
```

## Important scientific note

- Ontology/rule-derived labels are **weak labels**, not final gold truth.
- Final thesis claims should rely on rows with manual validation status `VALIDATED` or `ADJUDICATED` and inclusion flag `INCLUDE`.
