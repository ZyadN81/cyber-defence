# Metrics Definition

## Label regime

This package uses multi-label classification with binary indicators per class.

For each class:

- TP: predicted 1 and true 1
- FP: predicted 1 and true 0
- FN: predicted 0 and true 1

## Per-class metrics

- Precision: $TP/(TP+FP)$
- Recall: $TP/(TP+FN)$
- F1: $2PR/(P+R)$
- Support: number of true positives + false negatives for class

## Aggregate metrics

- Micro precision/recall/F1: computed over globally summed TP/FP/FN
- Macro precision/recall/F1: unweighted mean of per-class metrics
- Weighted precision/recall/F1: class-metric average weighted by support

## Excel assumptions

- Binary matrices in `Label_Matrix` are authoritative for spreadsheet metrics.
- Formula cells are visible and locked to reduce accidental changes.
- Input cells are highlighted and remain editable.

## Python assumptions

- `recompute_metrics.py` validates taxonomy membership before scoring.
- Non-evaluable rows are excluded and reported separately.

## Interpretation guardrails

- Macro metrics can be unstable with low support classes.
- Scenario-only results (N=10) are insufficient for broad performance claims.
- Final thesis claims require adequate manually validated sample size.
