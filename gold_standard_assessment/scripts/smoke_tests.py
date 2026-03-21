from __future__ import annotations

from lib import category_for_labels, labels_to_string, load_taxonomy, normalize_multilabel, resolve_paths, severity_for_labels


def main() -> None:
    paths = resolve_paths()
    taxonomy = load_taxonomy(paths.data_dir / "label_taxonomy.csv")

    assert normalize_multilabel("a; b; a") == ["a", "b"]
    assert labels_to_string(["b", "a", "a"]) == "a; b"

    cat, cat_rule = category_for_labels(["malware", "phishing"], taxonomy)
    assert cat == "Malware"
    assert cat_rule in {"multi_category_priority_rule", "single_category"}

    sev, sev_rule = severity_for_labels(["phishing", "malware"], taxonomy)
    assert sev == "Critical"
    assert sev_rule == "max_severity_rule"

    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
