from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from lib import normalize_multilabel, read_csv_rows, resolve_paths


@dataclass
class ValidationIssue:
    severity: str
    file: str
    row: int
    field: str
    message: str


def _load_schema(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_required_columns(file_path: Path, rows: List[Dict[str, str]], required: List[str]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not rows:
        return [ValidationIssue("error", str(file_path), 0, "*", "file has no data rows")]
    cols = set(rows[0].keys())
    missing = [c for c in required if c not in cols]
    for col in missing:
        issues.append(ValidationIssue("error", str(file_path), 0, col, "missing required column"))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CSV schema and taxonomy consistency.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    args = parser.parse_args()

    paths = resolve_paths()
    schema_path = paths.schemas_dir / "csv_schema.json" if hasattr(paths, "schemas_dir") else paths.package_root / "schemas" / "csv_schema.json"
    taxonomy_path = paths.data_dir / "label_taxonomy.csv"

    schema = _load_schema(schema_path)
    taxonomy_rows = read_csv_rows(taxonomy_path)
    taxonomy_labels: Set[str] = {r["label"].strip().lower() for r in taxonomy_rows}

    issues: List[ValidationIssue] = []

    for entry in schema["files"]:
        rel = entry["path"]
        file_path = paths.package_root / rel
        if not file_path.exists():
            issues.append(ValidationIssue("error", str(file_path), 0, "*", "file does not exist"))
            continue

        rows = read_csv_rows(file_path)
        issues.extend(_validate_required_columns(file_path, rows, entry.get("required_columns", [])))

        id_col = entry.get("id_column")
        if id_col and rows:
            seen: Set[str] = set()
            for i, row in enumerate(rows, start=2):
                rid = row.get(id_col, "").strip()
                if not rid:
                    issues.append(ValidationIssue("error", str(file_path), i, id_col, "empty id"))
                    continue
                if rid in seen:
                    issues.append(ValidationIssue("error", str(file_path), i, id_col, f"duplicate id: {rid}"))
                seen.add(rid)

        label_fields = entry.get("label_columns", [])
        for i, row in enumerate(rows, start=2):
            for field in label_fields:
                raw = row.get(field, "")
                if raw and "," in raw:
                    issues.append(ValidationIssue("warning", str(file_path), i, field, "labels should use ';' separator, not ','"))
                labels = normalize_multilabel(raw)
                for lbl in labels:
                    if lbl not in taxonomy_labels and not lbl.startswith("UNMAPPED::"):
                        issues.append(ValidationIssue("error", str(file_path), i, field, f"label not in taxonomy: {lbl}"))

            for field in entry.get("non_empty_columns", []):
                if not row.get(field, "").strip():
                    issues.append(ValidationIssue("error", str(file_path), i, field, "empty required cell"))

    report_path = paths.logs_dir / "schema_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "issues": [issue.__dict__ for issue in issues],
        "error_count": sum(1 for i in issues if i.severity == "error"),
        "warning_count": sum(1 for i in issues if i.severity == "warning"),
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    for issue in issues:
        print(f"[{issue.severity.upper()}] {issue.file}:{issue.row} {issue.field} - {issue.message}")

    errors = report_payload["error_count"]
    warnings = report_payload["warning_count"]
    print(f"Validation complete. errors={errors}, warnings={warnings}. report={report_path}")

    if errors > 0 or (args.strict and warnings > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
