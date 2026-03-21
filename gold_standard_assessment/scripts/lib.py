from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from rdflib import Graph, Namespace, RDF

D3F = Namespace("http://d3fend.mitre.org/ontologies/d3fend.owl#")


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    package_root: Path
    backend_root: Path
    ontology_path: Path
    manual_scenarios_path: Path
    data_dir: Path
    evaluation_dir: Path
    reports_dir: Path
    logs_dir: Path
    schemas_dir: Path


def resolve_paths() -> Paths:
    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parent
    backend_root = repo_root / "backend"
    return Paths(
        repo_root=repo_root,
        package_root=package_root,
        backend_root=backend_root,
        ontology_path=backend_root / "d3fend_output.owl",
        manual_scenarios_path=repo_root / "manual_test_scenarios.txt",
        data_dir=package_root / "data",
        evaluation_dir=package_root / "evaluation",
        reports_dir=package_root / "reports",
        logs_dir=package_root / "logs",
        schemas_dir=package_root / "schemas",
    )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def normalize_multilabel(value: str) -> List[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(";")]
    cleaned = [p for p in parts if p]
    return sorted(set(cleaned))


def labels_to_string(labels: Iterable[str]) -> str:
    uniq = sorted({lbl.strip() for lbl in labels if lbl and lbl.strip()})
    return "; ".join(uniq)


def load_taxonomy(taxonomy_path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(taxonomy_path)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        label = row["label"].strip().lower()
        out[label] = {
            "label": label,
            "tactic": row.get("tactic", "").strip(),
            "category": row.get("category", "").strip(),
            "severity": row.get("severity", "").strip(),
            "definition": row.get("definition", "").strip(),
            "category_definition": row.get("category_definition", "").strip(),
            "mapping_status": row.get("mapping_status", "").strip(),
            "notes": row.get("notes", "").strip(),
        }
    return out


def category_for_labels(labels: Sequence[str], taxonomy: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    categories = [taxonomy.get(lbl, {}).get("category", "") for lbl in labels]
    categories = [c for c in categories if c]
    if not categories:
        return "Uncategorized", "no_label_in_taxonomy"
    unique = sorted(set(categories))
    if len(unique) == 1:
        return unique[0], "single_category"
    # deterministic precedence to avoid nondeterministic assignment
    priority = ["Malware", "Network Attacks", "Data Breach", "System Vulnerability"]
    for p in priority:
        if p in unique:
            return p, "multi_category_priority_rule"
    return unique[0], "multi_category_fallback"


def severity_for_labels(labels: Sequence[str], taxonomy: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    severities = [taxonomy.get(lbl, {}).get("severity", "") for lbl in labels]
    severities = [s for s in severities if s]
    if not severities:
        return "Unknown", "no_label_in_taxonomy"
    rank = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    best = sorted(set(severities), key=lambda s: rank.get(s, 0), reverse=True)[0]
    return best, "max_severity_rule"


def parse_manual_scenarios(content: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    pattern = re.compile(r"##\s*(Simple|Advanced)\s*(\d+):\s*([^\n]+)\n\n([\s\S]*?)\n---", re.MULTILINE)
    for m in pattern.finditer(content):
        kind, idx, title, text = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
        blocks.append(
            {
                "scenario_id": f"{kind.lower()}_{idx}",
                "scenario_name": f"{kind} {idx}: {title}",
                "scenario_type": kind,
                "scenario_text": text,
            }
        )
    return blocks


def extract_expected_labels(content: str) -> Dict[str, List[str]]:
    def _clean(raw: str) -> List[str]:
        cleaned = re.sub(r"\s+with\s+[^,;]+\s+risk\s*$", "", raw.strip(), flags=re.IGNORECASE)
        return [s.strip().lower() for s in cleaned.split(",") if s.strip()]

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


def ontology_abstract_rows(ontology_path: Path, taxonomy: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    graph = Graph()
    graph.parse(ontology_path, format="xml")

    rows: List[Dict[str, object]] = []
    for abstract_uri in graph.subjects(RDF.type, D3F.Abstract):
        match = re.search(r"abstract(\d+)", str(abstract_uri))
        if not match:
            continue

        abstract_id = match.group(1)
        labels: Set[str] = set()
        tactics: Set[str] = set()
        unknown_labels: Set[str] = set()

        for _, _, sentence in graph.triples((abstract_uri, D3F.hasSentence, None)):
            for _, _, segment in graph.triples((sentence, D3F.hasSegment, None)):
                for _, _, label in graph.triples((segment, D3F.hasLabel, None)):
                    key = str(label).rsplit("/", 1)[-1].rsplit("#", 1)[-1].lower()
                    labels.add(key)
                    tax = taxonomy.get(key)
                    if tax:
                        if tax.get("tactic"):
                            tactics.add(tax["tactic"])
                    else:
                        unknown_labels.add(key)
                for _, _, tac in graph.triples((segment, D3F.mitigatedBy, None)):
                    tactics.add(str(tac).split("#")[-1])

        labels_sorted = sorted(labels)
        tactics_sorted = sorted(tactics)
        category, category_rule = category_for_labels(labels_sorted, taxonomy)
        severity, severity_rule = severity_for_labels(labels_sorted, taxonomy)

        rows.append(
            {
                "record_id": f"abstract_{abstract_id}",
                "source_type": "ontology_abstract",
                "abstract_id": abstract_id,
                "abstract_uri": str(abstract_uri),
                "weak_labels": labels_to_string(labels_sorted),
                "weak_tactics": labels_to_string(tactics_sorted),
                "derived_category": category,
                "derived_severity": severity,
                "category_rule": category_rule,
                "severity_rule": severity_rule,
                "unknown_labels": labels_to_string(unknown_labels),
                "label_count": len(labels_sorted),
                "tactic_count": len(tactics_sorted),
            }
        )

    rows.sort(key=lambda r: int(str(r["abstract_id"])))
    return rows
