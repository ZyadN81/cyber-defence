from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from lib import normalize_multilabel, read_csv_rows, resolve_paths, write_csv_rows


def _index_by_id(rows: List[Dict[str, str]], id_col: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        rid = row.get(id_col, "").strip()
        if rid:
            out[rid] = row
    return out


def main() -> None:
    paths = resolve_paths()
    taxonomy_rows = read_csv_rows(paths.data_dir / "label_taxonomy.csv")
    valid_labels: Set[str] = {r["label"].strip().lower() for r in taxonomy_rows}
    valid_tactics: Set[str] = {r.get("tactic", "").strip() for r in taxonomy_rows if r.get("tactic", "").strip()}

    gold_rows = read_csv_rows(paths.data_dir / "manually_validated_gold_subset.csv")
    pred_rows = read_csv_rows(paths.evaluation_dir / "predictions_template.csv")
    pred_by_id = _index_by_id(pred_rows, "record_id")

    valid_eval_rows = []
    mismatches = []
    invalid_rows = []
    skipped_rows = []
    unlabeled_rows = []

    for row in gold_rows:
        rid = row.get("record_id", "").strip()
        inclusion_flag = row.get("inclusion_flag", "").strip().upper()
        status = row.get("validation_status", "").strip().upper()

        if inclusion_flag != "INCLUDE":
            skipped_rows.append({"record_id": rid, "reason": f"inclusion_flag={inclusion_flag or 'EMPTY'}"})
            continue
        if status not in {"VALIDATED", "ADJUDICATED"}:
            skipped_rows.append({"record_id": rid, "reason": f"validation_status={status or 'EMPTY'}"})
            continue

        gold_labels = normalize_multilabel(row.get("final_adjudicated_labels", ""))
        if not gold_labels:
            unlabeled_rows.append({"record_id": rid, "reason": "final_adjudicated_labels empty"})
            continue

        pred_row = pred_by_id.get(rid)
        if not pred_row:
            invalid_rows.append({"record_id": rid, "field": "predictions", "issue": "missing prediction row"})
            continue

        pred_labels = normalize_multilabel(pred_row.get("predicted_labels", ""))
        pred_tactics = normalize_multilabel(pred_row.get("predicted_tactics", ""))

        bad_gold = [l for l in gold_labels if l not in valid_labels]
        bad_pred_labels = [l for l in pred_labels if l not in valid_labels]
        bad_pred_tactics = [t for t in pred_tactics if t not in valid_tactics]
        if bad_gold:
            invalid_rows.append({"record_id": rid, "field": "final_adjudicated_labels", "issue": f"invalid labels: {bad_gold}"})
            continue
        if bad_pred_labels:
            invalid_rows.append({"record_id": rid, "field": "predicted_labels", "issue": f"invalid labels: {bad_pred_labels}"})
            continue
        if bad_pred_tactics:
            invalid_rows.append({"record_id": rid, "field": "predicted_tactics", "issue": f"invalid tactics: {bad_pred_tactics}"})
            continue

        valid_eval_rows.append(
            {
                "record_id": rid,
                "gold_labels": gold_labels,
                "predicted_labels": pred_labels,
                "gold_set": set(gold_labels),
                "pred_set": set(pred_labels),
            }
        )

    classes = sorted({lbl for row in valid_eval_rows for lbl in (row["gold_labels"] + row["predicted_labels"])})

    metrics_json_path = paths.evaluation_dir / "evaluation_metrics.json"
    metrics_csv_path = paths.evaluation_dir / "evaluation_metrics.csv"
    mismatch_csv_path = paths.evaluation_dir / "mismatch_report.csv"

    if not classes or not valid_eval_rows:
        payload = {
            "status": "no_evaluable_rows",
            "counts": {
                "valid_rows": len(valid_eval_rows),
                "invalid_rows": len(invalid_rows),
                "skipped_rows": len(skipped_rows),
                "unlabeled_rows": len(unlabeled_rows),
            },
            "note": "At least one INCLUDE + ADJUDICATED/VALIDATED row with valid gold/pred labels is required.",
        }
        metrics_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_csv_rows(metrics_csv_path, [], ["metric", "value"])
        write_csv_rows(
            mismatch_csv_path,
            [{"record_id": "", "gold_labels": "", "predicted_labels": "", "missing": "", "extra": "", "jaccard": "", "status": "NO_EVALUABLE_ROWS"}],
            ["record_id", "gold_labels", "predicted_labels", "missing", "extra", "jaccard", "status"],
        )
        print("No evaluable rows. See evaluation_metrics.json for details.")
        return

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = np.zeros((len(valid_eval_rows), len(classes)), dtype=int)
    y_pred = np.zeros((len(valid_eval_rows), len(classes)), dtype=int)

    for r_idx, row in enumerate(valid_eval_rows):
        for label in row["gold_labels"]:
            y_true[r_idx, class_to_idx[label]] = 1
        for label in row["predicted_labels"]:
            y_pred[r_idx, class_to_idx[label]] = 1

        missing = sorted(row["gold_set"] - row["pred_set"])
        extra = sorted(row["pred_set"] - row["gold_set"])
        jaccard = 0.0
        union = row["gold_set"] | row["pred_set"]
        if union:
            jaccard = len(row["gold_set"] & row["pred_set"]) / len(union)

        mismatches.append(
            {
                "record_id": row["record_id"],
                "gold_labels": "; ".join(row["gold_labels"]),
                "predicted_labels": "; ".join(row["predicted_labels"]),
                "missing": "; ".join(missing),
                "extra": "; ".join(extra),
                "jaccard": f"{jaccard:.6f}",
                "status": "MATCH" if not missing and not extra else "MISMATCH",
            }
        )

    pr_micro, rc_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pr_weighted, rc_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    support_counter = Counter()
    for row in valid_eval_rows:
        for lbl in row["gold_labels"]:
            support_counter[lbl] += 1

    per_class = []
    for i, cls in enumerate(classes):
        per_class.append(
            {
                "class": cls,
                "precision": float(prec_c[i]),
                "recall": float(rec_c[i]),
                "f1": float(f1_c[i]),
                "support": int(sup_c[i]),
                "support_check": int(support_counter[cls]),
            }
        )

    payload = {
        "status": "ok",
        "counts": {
            "valid_rows": len(valid_eval_rows),
            "invalid_rows": len(invalid_rows),
            "skipped_rows": len(skipped_rows),
            "unlabeled_rows": len(unlabeled_rows),
            "class_count": len(classes),
        },
        "overall": {
            "micro_precision": float(pr_micro),
            "micro_recall": float(rc_micro),
            "micro_f1": float(f1_micro),
            "macro_precision": float(pr_macro),
            "macro_recall": float(rc_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(pr_weighted),
            "weighted_recall": float(rc_weighted),
            "weighted_f1": float(f1_weighted),
        },
        "per_class": per_class,
        "invalid_rows": invalid_rows,
        "skipped_rows": skipped_rows,
        "unlabeled_rows": unlabeled_rows,
    }

    metrics_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metrics_rows = [
        {"metric": "micro_precision", "value": float(pr_micro)},
        {"metric": "micro_recall", "value": float(rc_micro)},
        {"metric": "micro_f1", "value": float(f1_micro)},
        {"metric": "macro_precision", "value": float(pr_macro)},
        {"metric": "macro_recall", "value": float(rc_macro)},
        {"metric": "macro_f1", "value": float(f1_macro)},
        {"metric": "weighted_precision", "value": float(pr_weighted)},
        {"metric": "weighted_recall", "value": float(rc_weighted)},
        {"metric": "weighted_f1", "value": float(f1_weighted)},
    ]
    for row in per_class:
        metrics_rows.append({"metric": f"class::{row['class']}::precision", "value": row["precision"]})
        metrics_rows.append({"metric": f"class::{row['class']}::recall", "value": row["recall"]})
        metrics_rows.append({"metric": f"class::{row['class']}::f1", "value": row["f1"]})
        metrics_rows.append({"metric": f"class::{row['class']}::support", "value": row["support"]})

    write_csv_rows(metrics_csv_path, metrics_rows, ["metric", "value"])
    write_csv_rows(
        mismatch_csv_path,
        mismatches,
        ["record_id", "gold_labels", "predicted_labels", "missing", "extra", "jaccard", "status"],
    )

    print(f"Wrote {metrics_json_path}")
    print(f"Wrote {metrics_csv_path}")
    print(f"Wrote {mismatch_csv_path}")


if __name__ == "__main__":
    main()
