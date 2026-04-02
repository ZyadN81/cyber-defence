# Data Dictionary

## Input
- `input/f1score_manual_gold_matrix.csv`: record-level manual gold labels (0/1) for all scenarios.

## Output
- `outputs/f1score_gold_matrix_resolved.csv`: validated manual gold matrix used for scoring.
- `outputs/f1score_scenario_concept_mapping.csv`: scenario-to-keyword/label/tactic mapping used by methods.
- `outputs/f1score_keyword_hits/*.csv`: binary keyword evidence tables by scenario.
- `outputs/f1score_predictions_by_method.csv`: record-level predictions for each method and scenario.
- `outputs/f1score_confusion_by_scenario_method.csv`: TP/TN/FP/FN + precision/recall/F1 per scenario and method.
- `outputs/f1score_method_metrics_summary.csv`: aggregate metrics by method.
- `outputs/f1score_traceability_manifest.json`: thesis-to-artifact traceability map.
