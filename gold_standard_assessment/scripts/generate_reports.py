from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from lib import normalize_multilabel, read_csv_rows, resolve_paths


def main() -> None:
    paths = resolve_paths()

    weak_abs = read_csv_rows(paths.data_dir / "weak_labels_abstracts.csv")
    weak_sc = read_csv_rows(paths.data_dir / "weak_labels_scenarios.csv")
    gold_subset = read_csv_rows(paths.data_dir / "manually_validated_gold_subset.csv")
    taxonomy = read_csv_rows(paths.data_dir / "label_taxonomy.csv")

    category_counts = Counter(r.get("derived_category", "") for r in weak_abs)
    severity_counts = Counter(r.get("derived_severity", "") for r in weak_abs)

    unknown_label_rows = [r for r in weak_abs if r.get("unknown_labels", "").strip()]

    mapped_labels = {r.get("label", "").strip().lower() for r in taxonomy}
    observed_labels = set()
    for row in weak_abs:
        for lbl in normalize_multilabel(row.get("weak_labels", "")):
            observed_labels.add(lbl)

    missing_mappings = sorted(observed_labels - mapped_labels)
    taxonomy_missing_tactic = sorted(
        [r.get("label", "").strip().lower() for r in taxonomy if not r.get("tactic", "").strip()]
    )

    include_count = sum(1 for r in gold_subset if r.get("inclusion_flag", "").strip().upper() == "INCLUDE")
    pending_validation = sum(1 for r in gold_subset if r.get("validation_status", "").strip().upper() == "NOT_REVIEWED")
    weak_total = len(weak_abs)
    ratio = (include_count / weak_total) if weak_total else 0.0

    dataset_profile_md = paths.reports_dir / "dataset_profile.md"
    dataset_profile_md.write_text(
        "\n".join(
            [
                "# Dataset Profile",
                "",
                f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
                "",
                "## Core Volumes",
                f"- Weak-label abstracts: {len(weak_abs)}",
                f"- Weak-label scenarios: {len(weak_sc)}",
                f"- Manual-validation subset rows: {len(gold_subset)}",
                "",
                "## Category Distribution (Weak Labels)",
                *[f"- {k}: {v}" for k, v in sorted(category_counts.items())],
                "",
                "## Severity Distribution (Weak Labels)",
                *[f"- {k}: {v}" for k, v in sorted(severity_counts.items())],
                "",
                "## Notes",
                "- Weak labels are ontology/rule-derived and are not equivalent to manually adjudicated gold truth.",
                "- Scenario set (N=10) is useful for smoke/illustrative checks but insufficient alone for thesis-grade claims.",
            ]
        ),
        encoding="utf-8",
    )

    validation_status_md = paths.reports_dir / "validation_status.md"
    validation_status_md.write_text(
        "\n".join(
            [
                "# Validation Status",
                "",
                f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
                "",
                "## Manual Validation Progress",
                f"- Gold subset rows: {len(gold_subset)}",
                f"- Included for final evaluation: {include_count}",
                f"- Pending manual review: {pending_validation}",
                f"- Weak-to-gold include ratio: {ratio:.4f}",
                "",
                "## Interpretation",
                "- If include count is low, report findings as preliminary and avoid overclaiming final thesis validity.",
                "- Final thesis tables should be based on INCLUDE + ADJUDICATED/VALIDATED rows only.",
            ]
        ),
        encoding="utf-8",
    )

    gap_analysis_md = paths.reports_dir / "gap_analysis.md"
    gap_analysis_md.write_text(
        "\n".join(
            [
                "# Gap Analysis",
                "",
                f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
                "",
                "## Current Weaknesses",
                f"- Records in weak-label pool: {weak_total}",
                f"- Records with unknown labels: {len(unknown_label_rows)}",
                f"- Missing taxonomy labels observed in data: {len(missing_mappings)}",
                f"- Taxonomy entries missing tactic mapping: {len(taxonomy_missing_tactic)}",
                f"- Manual scenario count: {len(weak_sc)}",
                "",
                "## Why Records Become Uncategorized",
                "- No recognized weak labels for the record.",
                "- Weak labels present but absent from taxonomy.",
                "- Multi-category conflicts resolved by deterministic priority rule (documented in methodology).",
                "",
                "## Missing/Ambiguous Label Signals",
                *([f"- Missing taxonomy label: {x}" for x in missing_mappings] if missing_mappings else ["- No additional missing labels detected in current weak-label extraction."]),
                *([f"- Missing tactic mapping in taxonomy: {x}" for x in taxonomy_missing_tactic] if taxonomy_missing_tactic else ["- All taxonomy labels currently map to tactics."]),
                "",
                "## Thesis-Safe Recommendations",
                "- Treat ontology/rule labels as weak supervision (silver standard), not final gold truth.",
                "- Expand manually validated subset and report inter-annotator agreement before final claims.",
                "- Use scenario metrics as supplementary demonstration, not principal evidence.",
                "- Include mismatch analysis and invalid/skipped row counts in thesis appendix.",
            ]
        ),
        encoding="utf-8",
    )

    machine_gap_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weak_total": weak_total,
        "unknown_label_rows": len(unknown_label_rows),
        "missing_taxonomy_labels": missing_mappings,
        "taxonomy_labels_without_tactic": taxonomy_missing_tactic,
        "manual_scenario_count": len(weak_sc),
        "gold_subset_rows": len(gold_subset),
        "included_gold_rows": include_count,
        "pending_manual_review": pending_validation,
        "weak_to_gold_include_ratio": ratio,
    }
    (paths.reports_dir / "gap_analysis.json").write_text(json.dumps(machine_gap_json, indent=2), encoding="utf-8")

    print(f"Wrote {dataset_profile_md}")
    print(f"Wrote {validation_status_md}")
    print(f"Wrote {gap_analysis_md}")
    print(f"Wrote {paths.reports_dir / 'gap_analysis.json'}")


if __name__ == "__main__":
    main()
