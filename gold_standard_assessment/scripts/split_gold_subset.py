from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone

from lib import normalize_multilabel, read_csv_rows, resolve_paths, write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manual-validation scaffolds from weak labels.")
    parser.add_argument("--sample-size", type=int, default=250, help="Number of abstracts to include in manual validation subset.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for subset sampling.")
    args = parser.parse_args()

    paths = resolve_paths()
    weak_path = paths.data_dir / "weak_labels_abstracts.csv"
    gold_subset_path = paths.data_dir / "manually_validated_gold_subset.csv"
    adjudication_path = paths.data_dir / "adjudication_queue.csv"

    weak_rows = read_csv_rows(weak_path)
    rng = random.Random(args.seed)

    # prioritize ambiguous/uncategorized records so manual effort improves validity fastest
    priority_rows = [
        r
        for r in weak_rows
        if r.get("derived_category", "") == "Uncategorized"
        or r.get("unknown_labels", "").strip()
        or int(r.get("label_count", "0") or 0) >= 3
    ]
    non_priority_rows = [r for r in weak_rows if r not in priority_rows]

    target = max(0, min(args.sample_size, len(weak_rows)))

    selected = []
    if priority_rows:
        take_priority = min(len(priority_rows), int(target * 0.6))
        selected.extend(rng.sample(priority_rows, take_priority))
    remaining = target - len(selected)
    if remaining > 0 and non_priority_rows:
        selected.extend(rng.sample(non_priority_rows, min(remaining, len(non_priority_rows))))

    selected_ids = {r["record_id"] for r in selected}

    now = datetime.now(timezone.utc).isoformat()

    gold_rows = []
    for row in selected:
        weak_labels = normalize_multilabel(row.get("weak_labels", ""))
        gold_rows.append(
            {
                "record_id": row["record_id"],
                "abstract_id": row.get("abstract_id", ""),
                "abstract_uri": row.get("abstract_uri", ""),
                "weak_labels": "; ".join(weak_labels),
                "weak_category": row.get("derived_category", ""),
                "weak_severity": row.get("derived_severity", ""),
                "annotator_1_labels": "",
                "annotator_2_labels": "",
                "final_adjudicated_labels": "",
                "annotator_1_category": "",
                "annotator_2_category": "",
                "final_adjudicated_category": "",
                "inclusion_flag": "PENDING",
                "exclusion_reason": "",
                "confidence": "",
                "adjudication_notes": "",
                "comments": "",
                "validation_status": "NOT_REVIEWED",
                "created_at_utc": now,
            }
        )

    adjudication_rows = []
    for row in weak_rows:
        if row["record_id"] in selected_ids:
            continue
        weak_labels = normalize_multilabel(row.get("weak_labels", ""))
        adjudication_rows.append(
            {
                "record_id": row["record_id"],
                "abstract_id": row.get("abstract_id", ""),
                "abstract_uri": row.get("abstract_uri", ""),
                "weak_labels": "; ".join(weak_labels),
                "weak_category": row.get("derived_category", ""),
                "priority_reason": "future_manual_validation_pool",
                "queue_status": "BACKLOG",
                "created_at_utc": now,
            }
        )

    write_csv_rows(
        gold_subset_path,
        gold_rows,
        [
            "record_id",
            "abstract_id",
            "abstract_uri",
            "weak_labels",
            "weak_category",
            "weak_severity",
            "annotator_1_labels",
            "annotator_2_labels",
            "final_adjudicated_labels",
            "annotator_1_category",
            "annotator_2_category",
            "final_adjudicated_category",
            "inclusion_flag",
            "exclusion_reason",
            "confidence",
            "adjudication_notes",
            "comments",
            "validation_status",
            "created_at_utc",
        ],
    )

    write_csv_rows(
        adjudication_path,
        adjudication_rows,
        [
            "record_id",
            "abstract_id",
            "abstract_uri",
            "weak_labels",
            "weak_category",
            "priority_reason",
            "queue_status",
            "created_at_utc",
        ],
    )

    print(f"Wrote {gold_subset_path} with {len(gold_rows)} rows")
    print(f"Wrote {adjudication_path} with {len(adjudication_rows)} rows")


if __name__ == "__main__":
    main()
