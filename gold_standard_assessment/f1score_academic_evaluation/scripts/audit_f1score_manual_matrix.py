from __future__ import annotations

import csv
import re
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_expected_labels(scenarios_path: Path):
    text = scenarios_path.read_text(encoding="utf-8")
    out = {}

    def clean(raw: str):
        raw = re.sub(r"\s+with\s+[^,;]+\s+risk\s*$", "", raw.strip(), flags=re.I)
        return {x.strip().lower() for x in raw.split(",") if x.strip()}

    for m in re.finditer(
        r"\*\*Simple\s*(\d)\s*\([^\)]+\):\*\*\s*Should\s*detect\s*([^\n]+)", text, flags=re.I
    ):
        out[f"simple_{m.group(1)}"] = clean(m.group(2))

    for m in re.finditer(
        r"\*\*Advanced\s*(\d)\s*\([^\)]+\):\*\*\s*Should\s*detect\s*([^\n]+)", text, flags=re.I
    ):
        out[f"advanced_{m.group(1)}"] = clean(m.group(2))

    return out


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    manual_path = root / "gold_standard_assessment" / "f1score_academic_evaluation" / "input" / "f1score_manual_gold_matrix.csv"
    notes_path = root / "gold_standard_assessment" / "f1score_academic_evaluation" / "input" / "f1score_manual_gold_matrix_autofill_audit.csv"
    weak_path = root / "gold_standard_assessment" / "data" / "weak_labels_abstracts.csv"
    scenarios_path = root / "manual_test_scenarios.txt"
    abstracts_dir = root / "backend" / "abstracts"

    manual_rows = read_csv(manual_path)
    note_rows = read_csv(notes_path)
    weak_rows = read_csv(weak_path)
    expected = parse_expected_labels(scenarios_path)

    scenario_ids = [c for c in manual_rows[0].keys() if c != "record_id"]
    weak_by_id = {r["record_id"]: r for r in weak_rows}
    note_by_pair = {(r["record_id"], r["scenario_id"]): r for r in note_rows}

    errors = []
    warnings = []

    if len(manual_rows) != len(weak_rows):
        errors.append(f"Row count mismatch manual={len(manual_rows)} weak={len(weak_rows)}")

    manual_ids = [r["record_id"] for r in manual_rows]
    if len(manual_ids) != len(set(manual_ids)):
        errors.append("Duplicate record_id found in manual matrix")

    missing = sorted(set(weak_by_id) - set(manual_ids))
    extra = sorted(set(manual_ids) - set(weak_by_id))
    if missing:
        errors.append(f"Missing ids in manual matrix: {len(missing)} (first={missing[0]})")
    if extra:
        errors.append(f"Extra ids in manual matrix: {len(extra)} (first={extra[0]})")

    invalid_cells = 0
    formula_mismatches = 0

    for row in manual_rows:
        rid = row["record_id"]
        weak_labels = {
            x.strip().lower()
            for x in weak_by_id[rid].get("weak_labels", "").split(";")
            if x.strip()
        }

        for sid in scenario_ids:
            value = row.get(sid, "").strip()
            if value not in {"0", "1"}:
                invalid_cells += 1
                continue

            recomputed = "1" if (weak_labels & expected.get(sid, set())) else "0"
            if value != recomputed:
                formula_mismatches += 1

            note = note_by_pair.get((rid, sid))
            if note and note.get("autofill_value", "") != value:
                warnings.append(
                    f"Value differs from prefill note at {rid}/{sid}: matrix={value} note={note.get('autofill_value', '')}"
                )

    prevalence = {sid: 0 for sid in scenario_ids}
    for row in manual_rows:
        for sid in scenario_ids:
            if row[sid] == "1":
                prevalence[sid] += 1

    spot_checks = []
    for row in manual_rows:
        if row.get("advanced_2") == "1" or row.get("simple_2") == "1":
            rid = row["record_id"]
            aid = weak_by_id[rid].get("abstract_id", "").strip()
            if aid and (abstracts_dir / aid).exists():
                text = (abstracts_dir / aid).read_text(encoding="utf-8", errors="ignore").lower()
                key_hits = [
                    k for k in ["ransom", "encrypt", "phishing", "credential", "email", "malware"] if k in text
                ]
                spot_checks.append((rid, aid, row.get("simple_2"), row.get("advanced_2"), key_hits[:4]))
            if len(spot_checks) >= 8:
                break

    print("AUDIT_RESULT_START")
    print(f"manual_rows={len(manual_rows)} weak_rows={len(weak_rows)} scenarios={len(scenario_ids)}")
    print(f"invalid_cells={invalid_cells}")
    print(f"formula_mismatches_vs_prefill_logic={formula_mismatches}")
    print(f"warning_note_differences={len(warnings)}")
    for sid in scenario_ids:
        print(f"prevalence_{sid}={prevalence[sid]}/{len(manual_rows)}")
    if errors:
        print("ERRORS:")
        for e in errors:
            print("-", e)
    else:
        print("ERRORS: none")
    print("SPOT_CHECKS:")
    for s in spot_checks:
        print(s)
    print("AUDIT_RESULT_END")


if __name__ == "__main__":
    main()
