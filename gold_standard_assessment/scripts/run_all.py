from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(script: Path, log_path: Path, args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(script)]
    if args:
        cmd.extend(args)
    line = "RUN: " + " ".join(cmd)
    print(line)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(line + "\n")
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    with log_path.open("a", encoding="utf-8") as logf:
        if proc.stdout:
            logf.write(proc.stdout + ("\n" if not proc.stdout.endswith("\n") else ""))
        if proc.stderr:
            logf.write("[stderr]\n" + proc.stderr + ("\n" if not proc.stderr.endswith("\n") else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command workflow for gold-standard assessment package.")
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-validate", action="store_true")
    parser.add_argument("--with-evaluation", action="store_true", help="Also run metric recomputation after generation.")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    package_root = scripts_dir.parent
    logs_dir = package_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "pipeline.log"
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{datetime.now(timezone.utc).isoformat()}] START run_all\n")

    _run(scripts_dir / "generate_weak_labels.py", log_path)
    _run(scripts_dir / "split_gold_subset.py", log_path, ["--sample-size", str(args.sample_size), "--seed", str(args.seed)])
    _run(scripts_dir / "build_evaluation_templates.py", log_path)

    validate_args = ["--strict"] if args.strict_validate else []
    _run(scripts_dir / "validate_schema.py", log_path, validate_args)
    _run(scripts_dir / "generate_reports.py", log_path)

    if args.with_evaluation:
        _run(scripts_dir / "recompute_metrics.py", log_path)

    done = "Pipeline completed successfully. Log: " + str(log_path)
    print(done)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"[{datetime.now(timezone.utc).isoformat()}] END run_all\n")


if __name__ == "__main__":
    main()
