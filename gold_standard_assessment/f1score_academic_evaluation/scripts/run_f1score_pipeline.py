from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


LABEL_KEYWORDS: Dict[str, List[str]] = {
    "malware": ["malware", "malicious", "infected", "virus", "trojan", "executable", "antivirus"],
    "ransomware": ["ransomware", "ransom", "encrypted", "cryption", "bitcoin", "decrypt"],
    "phishing": ["phishing", "spoof", "credential", "fake login", "verify identity", "urgent"],
    "cyberattack": ["attack", "adversary", "compromise", "intrusion", "exploit"],
    "cyberthreat": ["threat", "indicator", "ioc", "risk"],
    "networksecurity": ["network", "traffic", "intrusion", "segmentation", "firewall", "smb"],
    "dataprotection": ["data", "exfiltration", "leak", "database", "dlp", "records"],
    "privacy": ["privacy", "pii", "confidential", "sensitive information", "personal data"],
    "cloudsecurity": ["cloud", "aws", "s3", "iam", "cloudtrail", "bucket"],
    "iotsecurity": ["iot", "device", "firmware", "sensor", "camera", "botnet"],
    "cybercrime": ["cybercrime", "criminal", "fraud", "insider"],
    "hacking": ["hack", "hacking", "unauthorized access", "privilege escalation"],
    "vulnerability": ["vulnerability", "cve", "weakness", "unpatched", "misconfiguration"],
    "cybersecurity": ["cybersecurity", "security operations", "soc", "incident response"],
    "digitalsecurity": ["digital security", "digital asset", "information security"],
    "cyberdefense": ["defense", "protection", "mitigation", "hardening"],
    "cyberawareness": ["awareness", "human factor", "security behavior"],
    "cybertraining": ["training", "education", "workshop", "learning"],
    "cyberworld": ["cyberspace", "digital environment", "ecosystem"],
    "threatintelligence": ["threat intelligence", "threat intel", "ioc feed"],
}


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    package_root: Path
    backend_root: Path
    abstracts_dir: Path
    manual_scenarios_path: Path
    weak_labels_path: Path
    taxonomy_path: Path
    input_dir: Path
    config_dir: Path
    outputs_dir: Path
    docs_dir: Path


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    scenario_name: str
    scenario_type: str
    scenario_text: str
    expected_labels: List[str]
    keywords: List[str]


def resolve_paths() -> Paths:
    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parents[1]
    backend_root = repo_root / "backend"
    return Paths(
        repo_root=repo_root,
        package_root=package_root,
        backend_root=backend_root,
        abstracts_dir=backend_root / "abstracts",
        manual_scenarios_path=repo_root / "manual_test_scenarios.txt",
        weak_labels_path=repo_root / "gold_standard_assessment" / "data" / "weak_labels_abstracts.csv",
        taxonomy_path=repo_root / "gold_standard_assessment" / "data" / "label_taxonomy.csv",
        input_dir=package_root / "input",
        config_dir=package_root / "config",
        outputs_dir=package_root / "outputs",
        docs_dir=package_root / "docs",
    )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_expected_labels(content: str) -> Dict[str, List[str]]:
    def _clean(raw: str) -> List[str]:
        raw = re.sub(r"\s+with\s+[^,;]+\s+risk\s*$", "", raw.strip(), flags=re.IGNORECASE)
        return [x.strip().lower() for x in raw.split(",") if x.strip()]

    expected: Dict[str, List[str]] = {}
    simple_re = re.compile(r"\*\*Simple\s*(\d)\s*\(([^\)]+)\):\*\*\s*Should\s*detect\s*([^\n]+)", re.IGNORECASE)
    advanced_re = re.compile(r"\*\*Advanced\s*(\d)\s*\(([^\)]+)\):\*\*\s*Should\s*detect\s*([^\n]+)", re.IGNORECASE)

    for m in simple_re.finditer(content):
        idx, _topic, labels = m.groups()
        expected[f"Simple {idx}:"] = _clean(labels)
    for m in advanced_re.finditer(content):
        idx, _topic, labels = m.groups()
        expected[f"Advanced {idx}:"] = _clean(labels)
    return expected


def parse_scenarios(content: str) -> List[Dict[str, str]]:
    pattern = re.compile(r"##\s*(Simple|Advanced)\s*(\d+):\s*([^\n]+)\n\n([\s\S]*?)\n---", re.MULTILINE)
    out: List[Dict[str, str]] = []
    for m in pattern.finditer(content):
        kind, idx, title, text = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
        out.append(
            {
                "scenario_id": f"{kind.lower()}_{idx}",
                "scenario_name": f"{kind} {idx}: {title}",
                "scenario_type": kind,
                "scenario_text": text,
            }
        )
    return out


def load_taxonomy_label_to_tactic(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    return {r["label"].strip().lower(): r.get("tactic", "").strip() for r in rows}


def build_scenario_registry(paths: Paths) -> List[Scenario]:
    content = paths.manual_scenarios_path.read_text(encoding="utf-8")
    parsed = parse_scenarios(content)
    expected = extract_expected_labels(content)

    scenarios: List[Scenario] = []
    for item in parsed:
        labels: List[str] = []
        for k, vals in expected.items():
            if item["scenario_name"].startswith(k):
                labels = sorted(set(vals))
                break

        keywords: Set[str] = set()
        for lbl in labels:
            keywords.update(LABEL_KEYWORDS.get(lbl, [lbl]))
        keywords.update(x.lower() for x in re.findall(r"[A-Za-z]{4,}", item["scenario_name"]))

        scenarios.append(
            Scenario(
                scenario_id=item["scenario_id"],
                scenario_name=item["scenario_name"],
                scenario_type=item["scenario_type"],
                scenario_text=item["scenario_text"],
                expected_labels=labels,
                keywords=sorted(k.strip() for k in keywords if k.strip()),
            )
        )
    return scenarios


def build_manual_template(paths: Paths, weak_rows: List[Dict[str, str]], scenarios: List[Scenario]) -> Path:
    template_path = paths.input_dir / "f1score_manual_gold_matrix.csv"
    fieldnames = ["record_id"] + [s.scenario_id for s in scenarios]
    rows = [{"record_id": r["record_id"], **{s.scenario_id: "" for s in scenarios}} for r in weak_rows]
    write_csv_rows(template_path, rows, fieldnames)
    return template_path


def validate_manual_gold(rows: List[Dict[str, str]], scenario_ids: List[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for i, row in enumerate(rows, start=2):
        rid = row.get("record_id", "").strip()
        if not rid:
            raise ValueError(f"manual gold row {i}: missing record_id")
        out[rid] = {}
        for sid in scenario_ids:
            val = row.get(sid, "").strip()
            if val not in {"0", "1"}:
                raise ValueError(f"manual gold row {i}, column {sid}: expected 0/1, got '{val}'")
            out[rid][sid] = int(val)
    return out


def safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def confusion(gold: Sequence[int], pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    tn = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "total": len(gold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    paths = resolve_paths()
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.docs_dir.mkdir(parents=True, exist_ok=True)

    weak_rows = read_csv_rows(paths.weak_labels_path)
    scenarios = build_scenario_registry(paths)
    scenario_ids = [s.scenario_id for s in scenarios]
    label_to_tactic = load_taxonomy_label_to_tactic(paths.taxonomy_path)

    manual_path = paths.input_dir / "f1score_manual_gold_matrix.csv"
    if not manual_path.exists():
        template = build_manual_template(paths, weak_rows, scenarios)
        raise SystemExit(
            "Academic workflow requires manual gold matrix. Template created at: "
            + str(template)
        )

    manual_rows = read_csv_rows(manual_path)
    manual_gold = validate_manual_gold(manual_rows, scenario_ids)

    weak_by_id = {r["record_id"]: r for r in weak_rows}
    missing_ids = sorted(set(weak_by_id.keys()) - set(manual_gold.keys()))
    if missing_ids:
        raise SystemExit(
            "Manual matrix missing record_ids from weak labels. First missing id: " + missing_ids[0]
        )

    # Gold matrix resolved (manual only)
    gold_rows: List[Dict[str, object]] = []
    for rid, row in manual_gold.items():
        out_row: Dict[str, object] = {"record_id": rid}
        for sid in scenario_ids:
            out_row[sid] = row[sid]
        gold_rows.append(out_row)
    write_csv_rows(paths.outputs_dir / "f1score_gold_matrix_resolved.csv", gold_rows, ["record_id", *scenario_ids])

    # Scenario-concept mapping
    mapping_rows: List[Dict[str, object]] = []
    for sc in scenarios:
        for lbl in sc.expected_labels:
            mapping_rows.append(
                {
                    "scenario_id": sc.scenario_id,
                    "scenario_name": sc.scenario_name,
                    "concept_type": "label",
                    "concept": lbl,
                }
            )
            tac = label_to_tactic.get(lbl, "")
            if tac:
                mapping_rows.append(
                    {
                        "scenario_id": sc.scenario_id,
                        "scenario_name": sc.scenario_name,
                        "concept_type": "d3fend_tactic",
                        "concept": tac,
                    }
                )
        for kw in sc.keywords:
            mapping_rows.append(
                {
                    "scenario_id": sc.scenario_id,
                    "scenario_name": sc.scenario_name,
                    "concept_type": "keyword",
                    "concept": kw,
                }
            )
    write_csv_rows(
        paths.outputs_dir / "f1score_scenario_concept_mapping.csv",
        mapping_rows,
        ["scenario_id", "scenario_name", "concept_type", "concept"],
    )

    # Load abstract texts
    text_by_id: Dict[str, str] = {}
    for wr in weak_rows:
        rid = wr["record_id"]
        aid = wr.get("abstract_id", "").strip()
        txt = ""
        if aid:
            fpath = paths.abstracts_dir / aid
            if fpath.exists():
                txt = normalize_text(fpath.read_text(encoding="utf-8", errors="ignore"))
        text_by_id[rid] = txt

    keyword_hits_dir = paths.outputs_dir / "f1score_keyword_hits"
    keyword_hits_dir.mkdir(parents=True, exist_ok=True)

    predictions_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []

    for sc in scenarios:
        kw_rows: List[Dict[str, object]] = []
        kw_fieldnames = ["record_id"] + [f"kw::{k}" for k in sc.keywords] + ["keyword_hit_count", "pred_keyword_search"]

        gold_vec: List[int] = []
        pred_kw_vec: List[int] = []
        pred_d3_vec: List[int] = []
        pred_hybrid_vec: List[int] = []

        expected_labels = set(sc.expected_labels)
        expected_tactics = {label_to_tactic.get(lbl, "") for lbl in sc.expected_labels if label_to_tactic.get(lbl, "")}

        for rid, gold_row in manual_gold.items():
            text = text_by_id.get(rid, "")
            kw_count = 0
            kw_row: Dict[str, object] = {"record_id": rid}
            for kw in sc.keywords:
                present = 1 if kw in text else 0
                kw_row[f"kw::{kw}"] = present
                kw_count += present
            pred_kw = 1 if kw_count > 0 else 0
            kw_row["keyword_hit_count"] = kw_count
            kw_row["pred_keyword_search"] = pred_kw
            kw_rows.append(kw_row)

            wr = weak_by_id[rid]
            row_labels = {x.strip().lower() for x in wr.get("weak_labels", "").split(";") if x.strip()}
            row_tactics = {x.strip() for x in wr.get("weak_tactics", "").split(";") if x.strip()}
            pred_d3 = 1 if (row_labels & expected_labels or row_tactics & expected_tactics) else 0
            pred_hybrid = 1 if (pred_kw or pred_d3) else 0

            gold = gold_row[sc.scenario_id]
            gold_vec.append(gold)
            pred_kw_vec.append(pred_kw)
            pred_d3_vec.append(pred_d3)
            pred_hybrid_vec.append(pred_hybrid)

            predictions_rows.append(
                {
                    "record_id": rid,
                    "scenario_id": sc.scenario_id,
                    "gold": gold,
                    "pred_keyword_search": pred_kw,
                    "pred_d3fend_without_abstracts": pred_d3,
                    "pred_d3fend_plus_abstracts": pred_hybrid,
                }
            )

        write_csv_rows(keyword_hits_dir / f"{sc.scenario_id}_keyword_hits.csv", kw_rows, kw_fieldnames)

        for method, pred in [
            ("keyword_search", pred_kw_vec),
            ("d3fend_without_abstracts", pred_d3_vec),
            ("d3fend_plus_abstracts", pred_hybrid_vec),
        ]:
            cm = confusion(gold_vec, pred)
            confusion_rows.append(
                {
                    "scenario_id": sc.scenario_id,
                    "scenario_name": sc.scenario_name,
                    "method": method,
                    "total": cm["total"],
                    "tp": cm["tp"],
                    "tn": cm["tn"],
                    "fp": cm["fp"],
                    "fn": cm["fn"],
                    "precision": f"{cm['precision']:.6f}",
                    "recall": f"{cm['recall']:.6f}",
                    "f1": f"{cm['f1']:.6f}",
                }
            )

    write_csv_rows(
        paths.outputs_dir / "f1score_predictions_by_method.csv",
        predictions_rows,
        [
            "record_id",
            "scenario_id",
            "gold",
            "pred_keyword_search",
            "pred_d3fend_without_abstracts",
            "pred_d3fend_plus_abstracts",
        ],
    )

    write_csv_rows(
        paths.outputs_dir / "f1score_confusion_by_scenario_method.csv",
        confusion_rows,
        ["scenario_id", "scenario_name", "method", "total", "tp", "tn", "fp", "fn", "precision", "recall", "f1"],
    )

    # Aggregate method summary
    summary_rows: List[Dict[str, object]] = []
    methods = ["keyword_search", "d3fend_without_abstracts", "d3fend_plus_abstracts"]
    for method in methods:
        rows = [r for r in confusion_rows if r["method"] == method]
        tp = sum(int(r["tp"]) for r in rows)
        tn = sum(int(r["tn"]) for r in rows)
        fp = sum(int(r["fp"]) for r in rows)
        fn = sum(int(r["fn"]) for r in rows)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        summary_rows.append(
            {
                "method": method,
                "total": sum(int(r["total"]) for r in rows),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "f1": f"{f1:.6f}",
            }
        )

    write_csv_rows(
        paths.outputs_dir / "f1score_method_metrics_summary.csv",
        summary_rows,
        ["method", "total", "tp", "tn", "fp", "fn", "precision", "recall", "f1"],
    )

    trace = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": "manual_gold_only",
        "manual_gold_input": str((paths.input_dir / "f1score_manual_gold_matrix.csv").relative_to(paths.repo_root)).replace("\\", "/"),
        "formulas": {
            "precision": "TP / (TP + FP)",
            "recall": "TP / (TP + FN)",
            "f1": "2 * precision * recall / (precision + recall)",
        },
        "artifacts": {
            "gold_standard_matrix": "gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_gold_matrix_resolved.csv",
            "scenario_mapping": "gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_scenario_concept_mapping.csv",
            "keyword_hits": "gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_keyword_hits/*.csv",
            "predictions": "gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_predictions_by_method.csv",
            "per_cell_confusion": "gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_confusion_by_scenario_method.csv",
            "method_summary": "gold_standard_assessment/f1score_academic_evaluation/outputs/f1score_method_metrics_summary.csv",
        },
    }
    (paths.outputs_dir / "f1score_traceability_manifest.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    data_dictionary = """# Data Dictionary

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
"""
    (paths.docs_dir / "f1score_data_dictionary.md").write_text(data_dictionary, encoding="utf-8")

    examiner_nav = """# Examiner Navigation

1. Start with `outputs/f1score_traceability_manifest.json`.
2. Verify manual ground truth in `outputs/f1score_gold_matrix_resolved.csv`.
3. Inspect raw search evidence in `outputs/f1score_keyword_hits/`.
4. Check record-level predictions in `outputs/f1score_predictions_by_method.csv`.
5. Validate metric derivation in `outputs/f1score_confusion_by_scenario_method.csv` and `outputs/f1score_method_metrics_summary.csv`.
6. Use formulas documented in `f1score_traceability_manifest.json` and `docs/f1score_data_dictionary.md`.
"""
    (paths.docs_dir / "f1score_examiner_guide.md").write_text(examiner_nav, encoding="utf-8")

    print("Section 4.7 academic package generated successfully.")
    print("Output folder:", paths.outputs_dir)


if __name__ == "__main__":
    main()
