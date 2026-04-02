from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def parse_expected_labels(manual_scenarios_txt: str) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}

    def _clean(raw: str) -> Set[str]:
        raw = re.sub(r"\s+with\s+[^,;]+\s+risk\s*$", "", raw.strip(), flags=re.IGNORECASE)
        return {x.strip().lower() for x in raw.split(",") if x.strip()}

    simple_re = re.compile(r"\*\*Simple\s*(\d)\s*\(([^\)]+)\):\*\*\s*Should\s*detect\s*([^\n]+)", re.IGNORECASE)
    advanced_re = re.compile(r"\*\*Advanced\s*(\d)\s*\(([^\)]+)\):\*\*\s*Should\s*detect\s*([^\n]+)", re.IGNORECASE)

    for m in simple_re.finditer(manual_scenarios_txt):
        idx, _topic, labels = m.groups()
        out[f"simple_{idx}"] = _clean(labels)
    for m in advanced_re.finditer(manual_scenarios_txt):
        idx, _topic, labels = m.groups()
        out[f"advanced_{idx}"] = _clean(labels)

    return out


def main() -> None:
    root = repo_root()
    package_root = root / "gold_standard_assessment" / "f1score_academic_evaluation"
    weak_path = root / "gold_standard_assessment" / "data" / "weak_labels_abstracts.csv"
    scenarios_path = root / "manual_test_scenarios.txt"

    template_path = package_root / "input" / "f1score_manual_gold_matrix.csv"
    prefill_path = package_root / "input" / "f1score_manual_gold_matrix_autofill.csv"
    notes_path = package_root / "input" / "f1score_manual_gold_matrix_autofill_audit.csv"

    weak_rows = read_csv(weak_path)
    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")

    template_rows = read_csv(template_path)
    if not template_rows:
        raise SystemExit("Template is empty.")

    header = list(template_rows[0].keys())
    scenario_ids = [c for c in header if c != "record_id"]

    expected = parse_expected_labels(scenarios_path.read_text(encoding="utf-8"))
    weak_by_id = {r["record_id"]: r for r in weak_rows}

    prefilled: List[Dict[str, str]] = []
    notes: List[Dict[str, str]] = []

    for row in template_rows:
        rid = row["record_id"].strip()
        weak = weak_by_id.get(rid, {})
        weak_labels = {x.strip().lower() for x in weak.get("weak_labels", "").split(";") if x.strip()}

        out_row = {"record_id": rid}
        for sid in scenario_ids:
            scenario_labels = expected.get(sid, set())
            hit = 1 if weak_labels.intersection(scenario_labels) else 0
            out_row[sid] = str(hit)
            notes.append(
                {
                    "record_id": rid,
                    "scenario_id": sid,
                    "autofill_value": str(hit),
                    "matched_labels": ";".join(sorted(weak_labels.intersection(scenario_labels))),
                    "scenario_expected_labels": ";".join(sorted(scenario_labels)),
                    "weak_labels": ";".join(sorted(weak_labels)),
                    "needs_manual_review": "YES",
                }
            )
        prefilled.append(out_row)

    write_csv(prefill_path, prefilled, ["record_id", *scenario_ids])
    write_csv(
        notes_path,
        notes,
        [
            "record_id",
            "scenario_id",
            "autofill_value",
            "matched_labels",
            "scenario_expected_labels",
            "weak_labels",
            "needs_manual_review",
        ],
    )

    print("Prefilled matrix generated:", prefill_path)
    print("Audit notes generated:", notes_path)
    print("Next: copy reviewed values into f1score_manual_gold_matrix.csv")


if __name__ == "__main__":
    main()
