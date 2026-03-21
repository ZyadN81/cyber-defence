from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from lib import (
    extract_expected_labels,
    labels_to_string,
    load_taxonomy,
    ontology_abstract_rows,
    parse_manual_scenarios,
    resolve_paths,
    severity_for_labels,
    category_for_labels,
    write_csv_rows,
)


def main() -> None:
    paths = resolve_paths()
    taxonomy_path = paths.data_dir / "label_taxonomy.csv"
    weak_abs_path = paths.data_dir / "weak_labels_abstracts.csv"
    weak_sc_path = paths.data_dir / "weak_labels_scenarios.csv"
    summary_json_path = paths.reports_dir / "gold_standard_summary.json"

    taxonomy = load_taxonomy(taxonomy_path)

    abstract_rows = ontology_abstract_rows(paths.ontology_path, taxonomy)

    with paths.manual_scenarios_path.open("r", encoding="utf-8") as f:
        content = f.read()
    scenarios = parse_manual_scenarios(content)
    expected = extract_expected_labels(content)

    scenario_rows = []
    for sc in scenarios:
        labels = []
        for key, vals in expected.items():
            if sc["scenario_name"].startswith(key):
                labels = vals
                break

        tactics = []
        for label in labels:
            tactic = taxonomy.get(label, {}).get("tactic", "")
            tactics.append(tactic if tactic else f"UNMAPPED::{label}")

        category, category_rule = category_for_labels(labels, taxonomy)
        severity, severity_rule = severity_for_labels(labels, taxonomy)

        scenario_rows.append(
            {
                "record_id": sc["scenario_id"],
                "source_type": "manual_scenario",
                "scenario_id": sc["scenario_id"],
                "scenario_name": sc["scenario_name"],
                "scenario_type": sc["scenario_type"],
                "scenario_text": sc["scenario_text"],
                "weak_labels": labels_to_string(labels),
                "weak_tactics": labels_to_string(tactics),
                "derived_category": category,
                "derived_severity": severity,
                "category_rule": category_rule,
                "severity_rule": severity_rule,
                "weak_label_source": "manual_test_scenarios_expected_summary",
            }
        )

    write_csv_rows(
        weak_abs_path,
        abstract_rows,
        [
            "record_id",
            "source_type",
            "abstract_id",
            "abstract_uri",
            "weak_labels",
            "weak_tactics",
            "derived_category",
            "derived_severity",
            "category_rule",
            "severity_rule",
            "unknown_labels",
            "label_count",
            "tactic_count",
        ],
    )

    write_csv_rows(
        weak_sc_path,
        scenario_rows,
        [
            "record_id",
            "source_type",
            "scenario_id",
            "scenario_name",
            "scenario_type",
            "scenario_text",
            "weak_labels",
            "weak_tactics",
            "derived_category",
            "derived_severity",
            "category_rule",
            "severity_rule",
            "weak_label_source",
        ],
    )

    category_counts = Counter(str(r["derived_category"]) for r in abstract_rows)
    severity_counts = Counter(str(r["derived_severity"]) for r in abstract_rows)
    uncategorized_reasons = Counter(str(r["category_rule"]) for r in abstract_rows if r["derived_category"] == "Uncategorized")
    unknown_label_records = sum(1 for r in abstract_rows if str(r.get("unknown_labels", "")).strip())

    taxonomy_labels = set(taxonomy.keys())
    labels_in_data = set()
    for row in abstract_rows:
        for lbl in str(row["weak_labels"]).split(";"):
            lbl = lbl.strip().lower()
            if lbl:
                labels_in_data.add(lbl)
    missing_in_taxonomy = sorted(labels_in_data - taxonomy_labels)

    summary_payload = {
        "artifact_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "label_layer": "weak_labels_only",
            "note": "Ontology/rule-derived outputs are weak labels and not manually validated gold truth.",
        },
        "inputs": {
            "ontology_path": str(paths.ontology_path),
            "manual_scenarios_path": str(paths.manual_scenarios_path),
            "taxonomy_path": str(taxonomy_path),
        },
        "outputs": {
            "weak_labels_abstracts": str(weak_abs_path),
            "weak_labels_scenarios": str(weak_sc_path),
        },
        "counts": {
            "abstract_rows": len(abstract_rows),
            "scenario_rows": len(scenario_rows),
            "taxonomy_labels": len(taxonomy_labels),
            "records_with_unknown_labels": unknown_label_records,
        },
        "distribution": {
            "abstract_category_distribution": dict(category_counts),
            "abstract_severity_distribution": dict(severity_counts),
            "uncategorized_rule_distribution": dict(uncategorized_reasons),
        },
        "taxonomy_gaps": {
            "labels_missing_in_taxonomy": missing_in_taxonomy,
        },
    }

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote {weak_abs_path}")
    print(f"Wrote {weak_sc_path}")
    print(f"Wrote {summary_json_path}")


if __name__ == "__main__":
    main()
