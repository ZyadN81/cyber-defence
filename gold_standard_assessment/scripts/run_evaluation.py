from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    log_path = scripts_dir.parent / "logs" / "evaluation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{datetime.now(timezone.utc).isoformat()}] START run_evaluation\n")

    commands = [
        [sys.executable, str(scripts_dir / "validate_schema.py")],
        [sys.executable, str(scripts_dir / "recompute_metrics.py")],
    ]

    for cmd in commands:
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

    print(f"Evaluation workflow completed. Log: {log_path}")
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"[{datetime.now(timezone.utc).isoformat()}] END run_evaluation\n")


if __name__ == "__main__":
    main()
