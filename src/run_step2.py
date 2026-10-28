"""Step 2 in one command: tests, classical baselines, tables, significance, report.

Run: python src/run_step2.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
    [sys.executable, str(ROOT / "src" / "baselines" / "classic.py")],
    [sys.executable, str(ROOT / "scripts" / "make_tables.py")],
    [sys.executable, str(ROOT / "scripts" / "significance.py")],
    [sys.executable, str(ROOT / "scripts" / "make_step2_report.py")],
]


def main():
    for cmd in STEPS:
        print("\n=== " + " ".join(Path(c).name if i else c for i, c in enumerate(cmd)), flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            sys.exit(f"=== step failed with exit code {rc}")


if __name__ == "__main__":
    main()
