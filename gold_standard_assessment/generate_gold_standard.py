from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    package_root = Path(__file__).resolve().parent
    scripts_dir = package_root / "scripts"

    subprocess.run([sys.executable, str(scripts_dir / "run_all.py")], check=True)

    # Backward-compatible filenames from the previous package version.
    compat_pairs = [
        (package_root / "data" / "weak_labels_abstracts.csv", package_root / "abstracts_gold_standard.csv"),
        (package_root / "data" / "weak_labels_scenarios.csv", package_root / "manual_scenarios_gold_standard.csv"),
        (package_root / "evaluation" / "f1_recompute_template.xlsx", package_root / "f1_recompute_template.xlsx"),
        (package_root / "reports" / "gold_standard_summary.json", package_root / "gold_standard_summary.json"),
    ]

    for src, dst in compat_pairs:
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"Wrote compatibility artifact: {dst}")


if __name__ == "__main__":
    main()
