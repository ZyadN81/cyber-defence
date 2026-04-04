# Detailed Script Description for Thesis Report

## Purpose

This document provides a detailed, report-ready description of the three scripts in the evaluation pipeline, using explicit line ranges (from-to) and functional descriptions.

## 1) Main Pipeline Script

File: [gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py)

| Line Range                                                                                                  | Description                                                                                                             |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [L1-L34](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L1-L34)       | Imports, typing setup, and the label-to-keyword dictionary used to build scenario keyword evidence.                     |
| [L37-L58](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L37-L58)     | Dataclasses (`Paths`, `Scenario`) defining structured configuration and scenario metadata.                              |
| [L61-L78](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L61-L78)     | Repository and dataset path resolution for inputs, outputs, docs, and backend abstracts.                                |
| [L80-L92](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L80-L92)     | CSV read/write utility functions for deterministic I/O.                                                                 |
| [L94-L114](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L94-L114)   | Text normalization and extraction of expected labels from scenario specification text.                                  |
| [L116-L130](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L116-L130) | Scenario parsing from the manual scenarios source into structured entries.                                              |
| [L132-L134](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L132-L134) | Taxonomy loading to map labels into D3FEND tactics.                                                                     |
| [L137-L166](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L137-L166) | Scenario registry construction: expected labels + generated keyword sets per scenario.                                  |
| [L168-L173](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L168-L173) | Template generation when manual gold matrix does not exist.                                                             |
| [L176-L189](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L176-L189) | Strict validation of manual gold labels (record ID presence and binary values only).                                    |
| [L191-L193](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L191-L193) | Safe division helper to avoid zero-division instability in metric computation.                                          |
| [L195-L212](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L195-L212) | Confusion matrix computation and per-vector precision/recall/F1 formulas.                                               |
| [L215-L235](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L215-L235) | Main flow initialization, scenario loading, and manual matrix enforcement (`manual_gold_only`).                         |
| [L237-L251](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L237-L251) | Gold matrix resolution and export to CSV as final scoring ground truth.                                                 |
| [L253-L287](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L253-L287) | Scenario-concept mapping export (labels, tactics, and keywords) for traceability.                                       |
| [L289-L303](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L289-L303) | Abstract text loading and keyword-hit output directory creation.                                                        |
| [L305-L376](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L305-L376) | Per-scenario evaluation loop: keyword predictions, D3FEND predictions, hybrid predictions, and per-cell confusion rows. |
| [L378-L399](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L378-L399) | Export of record-level predictions and scenario-method confusion outputs.                                               |
| [L401-L429](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L401-L429) | Aggregation of global method metrics (TP/TN/FP/FN, precision, recall, F1).                                              |
| [L431-L449](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L431-L449) | Creation of traceability manifest linking formulas and generated artifacts.                                             |
| [L451-L476](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L451-L476) | Auto-generation of supporting docs (data dictionary and examiner guide).                                                |
| [L482-L483](gold_standard_assessment/f1score_academic_evaluation/scripts/run_f1score_pipeline.py#L482-L483) | Script entry point.                                                                                                     |

## 2) Bootstrap Script for Manual Matrix

File: [gold_standard_assessment/f1score_academic_evaluation/scripts/bootstrap_f1score_manual_matrix.py](gold_standard_assessment/f1score_academic_evaluation/scripts/bootstrap_f1score_manual_matrix.py)

| Line Range                                                                                                             | Description                                                                                |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [L1-L25](gold_standard_assessment/f1score_academic_evaluation/scripts/bootstrap_f1score_manual_matrix.py#L1-L25)       | Imports and base CSV utility functions for reading/writing bootstrap artifacts.            |
| [L27-L45](gold_standard_assessment/f1score_academic_evaluation/scripts/bootstrap_f1score_manual_matrix.py#L27-L45)     | Extraction of expected labels per scenario from the scenario specification text.           |
| [L47-L55](gold_standard_assessment/f1score_academic_evaluation/scripts/bootstrap_f1score_manual_matrix.py#L47-L55)     | Main setup and declaration of output files: template, autofill, and audit notes.           |
| [L57-L71](gold_standard_assessment/f1score_academic_evaluation/scripts/bootstrap_f1score_manual_matrix.py#L57-L71)     | Loading template and weak-label data; basic existence and emptiness checks.                |
| [L73-L96](gold_standard_assessment/f1score_academic_evaluation/scripts/bootstrap_f1score_manual_matrix.py#L73-L96)     | Cell-level autofill logic based on weak-label overlap with scenario expected labels.       |
| [L97-L111](gold_standard_assessment/f1score_academic_evaluation/scripts/bootstrap_f1score_manual_matrix.py#L97-L111)   | Writing reviewer artifacts with explicit `autofill_value` and `needs_manual_review` flags. |
| [L117-L118](gold_standard_assessment/f1score_academic_evaluation/scripts/bootstrap_f1score_manual_matrix.py#L117-L118) | Script entry point.                                                                        |

## 3) Audit Script for Manual Matrix Quality

File: [gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py](gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py)

| Line Range                                                                                                         | Description                                                                                            |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| [L1-L31](gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py#L1-L31)       | Imports, CSV reader, and parsing of expected scenario labels used by audit checks.                     |
| [L34-L44](gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py#L34-L44)     | Main setup and loading of manual matrix, bootstrap audit notes, weak labels, and scenario definitions. |
| [L46-L67](gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py#L46-L67)     | Structural integrity checks: row count, duplicate IDs, missing/extra IDs.                              |
| [L69-L93](gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py#L69-L93)     | Logical consistency checks: invalid cells, mismatch with prefill logic, note differences.              |
| [L95-L99](gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py#L95-L99)     | Scenario prevalence computation for distribution sanity checks.                                        |
| [L101-L113](gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py#L101-L113) | Semantic spot checks using selected security terms from real abstract text.                            |
| [L115-L131](gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py#L115-L131) | Structured audit report output (`AUDIT_RESULT_START` to `AUDIT_RESULT_END`) for reproducible logs.     |
| [L134-L135](gold_standard_assessment/f1score_academic_evaluation/scripts/audit_f1score_manual_matrix.py#L134-L135) | Script entry point.                                                                                    |

## Suggested Thesis Usage

Use this description in the implementation chapter to justify:

1. Deterministic pipeline design.
2. Manual-gold-first evaluation governance.
3. Multi-method comparison protocol.
4. Reproducibility and auditability through explicit artifact generation and validation.
