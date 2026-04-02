# Section 4.7: Gold Standard Evaluation and F1-Score Analysis

## 4.7.1 Objective

This section evaluates the effectiveness of three detection strategies against a manually curated gold standard to quantify classification quality using Precision, Recall, and F1-score. The objective is to determine how reliably each strategy identifies scenario-relevant cybersecurity concepts across the evaluation corpus.

## 4.7.2 Evaluation Design

The evaluation is performed using a deterministic and reproducible pipeline in:

- `gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py`

The process is configured as `manual_gold_only`, meaning the final evaluation labels are taken exclusively from the manually curated matrix:

- `gold_standard_assessment/f1score_academic_evaluation/input/f1score_manual_gold_matrix.csv`

The pipeline evaluates 10 scenarios (`simple_1` to `simple_5`, `advanced_1` to `advanced_5`) over 8,592 records, yielding a total of:

$$
N = 8592 \times 10 = 85920
$$

binary decisions per method.

## 4.7.3 Compared Methods

Three methods are compared:

1. `keyword_search`

- Predicts positive when scenario-specific keyword evidence is found in the abstract text.

2. `d3fend_without_abstracts`

- Predicts positive using ontology-derived weak labels/tactics overlap only.

3. `d3fend_plus_abstracts`

- Predicts positive when either ontology overlap or keyword evidence indicates relevance.

Per-record predictions are stored in:

- `gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_predictions_by_method.csv`

## 4.7.4 Metric Definitions

For each method, confusion components are computed:

- True Positive ($TP$)
- True Negative ($TN$)
- False Positive ($FP$)
- False Negative ($FN$)

Metrics follow standard definitions:

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

$$
\text{Recall} = \frac{TP}{TP+FN}
$$

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision}+\text{Recall}}
$$

Per-scenario confusion details are stored in:

- `gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_confusion_by_scenario_method.csv`

Aggregated method-level results are stored in:

- `gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_method_metrics_summary.csv`

## 4.7.5 Results

Final aggregated results are:

1. `keyword_search`

- $TP = 18153$, $TN = 37933$, $FP = 23235$, $FN = 6599$
- Precision = 0.438605
- Recall = 0.733395
- F1 = 0.548927

2. `d3fend_without_abstracts`

- $TP = 24752$, $TN = 61123$, $FP = 45$, $FN = 0$
- Precision = 0.998185
- Recall = 1.000000
- F1 = 0.999092

3. `d3fend_plus_abstracts`

- $TP = 24752$, $TN = 37910$, $FP = 23258$, $FN = 0$
- Precision = 0.515559
- Recall = 1.000000
- F1 = 0.680355

## 4.7.6 Interpretation

The `keyword_search` baseline shows moderate performance. Its recall is acceptable, but precision is substantially reduced by high false-positive volume, indicating lexical ambiguity and context-insensitive matching.

The `d3fend_without_abstracts` strategy achieves near-perfect values, with zero false negatives and only 45 false positives. Under the current gold matrix, this indicates almost complete alignment between ontology-based weak evidence and final gold positives.

The `d3fend_plus_abstracts` method retains perfect recall but suffers a precision drop relative to pure ontology matching. This is expected because OR-combination with keyword evidence introduces many additional positives, increasing $FP$.

## 4.7.7 Validity and Academic Caution

Although the reported F1 values are mathematically correct and reproducible, interpretation must include a validity caution:

- The current gold matrix was initialized from evidence strongly related to weak-label overlap.
- Consequently, the very high score of `d3fend_without_abstracts` may partially reflect label-source coupling.
- To strengthen construct validity, manual adjudication should prioritize disagreement and high-uncertainty cells (see priority review file).

High-priority review artifact:

- `gold_standard_assessment/f1score_academic_evaluation/input/f1score_manual_review_priority_top2000.csv`

This caveat does not invalidate the computation; it limits claims about external generalization and method independence.

## 4.7.8 Reproducibility and Traceability

Traceability is provided in:

- `gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_traceability_manifest.json`

This manifest records:

- generation timestamp,
- evaluation mode,
- source gold matrix path,
- metric formulas,
- exact output artifact mapping.

Supporting documentation:

- `gold_standard_assessment/f1score_academic_evaluation/docs/f1score_data_dictionary.md`
- `gold_standard_assessment/f1score_academic_evaluation/docs/f1score_examiner_guide.md`

Audit script for matrix integrity:

- `gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py`

## 4.7.9 Thesis-Ready Discussion Paragraph

The evaluation framework in Section 4.7 compares keyword-based detection, ontology-only D3FEND mapping, and a hybrid ontology-plus-abstract strategy against a manually curated gold standard across 85,920 binary decisions. Results show that keyword matching achieves moderate effectiveness (F1 = 0.5489), while ontology-only mapping produces near-perfect performance (F1 = 0.9991) under the present annotation matrix, and the hybrid method improves recall to 1.0 at the cost of reduced precision (F1 = 0.6804). These findings suggest that ontology-grounded semantic mapping is highly effective for this dataset, whereas lexical matching alone introduces substantial false positives. However, because the manual matrix was bootstrapped from weak evidence before review, the ontology-only score should be interpreted with validity caution, and final claims are bounded to the current adjudication state. The study therefore reports both the quantitative superiority of ontology-based matching and the methodological limitation related to potential source coupling, preserving transparency and reproducibility.

## 4.7.10 Suggested Final Claim Boundaries

Use the following academically safe claims in the thesis:

1. The reported metrics are exact, deterministic, and reproducible from the provided artifacts.
2. Within the current gold matrix, ontology-based matching outperforms keyword-only matching in F1.
3. Hybrid OR-fusion increases recall but reduces precision due to false-positive expansion.
4. External validity is conditional on further independent manual adjudication.
