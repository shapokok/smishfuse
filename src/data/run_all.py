"""Runs the whole step-1 pipeline in order.

Run: python src/data/run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "load_mendeley.py",
    "load_moz_smishing.py",
    "load_super_sms.py",
    "load_uci_sms.py",
    "dedup.py",
    "make_super_sms_clean.py",
    "overlap.py",
    "split.py",
    "report.py",
]


def main():
    for step in STEPS:
        print(f"\n=== {step}")
        rc = subprocess.run([sys.executable, str(HERE / step)]).returncode
        if rc != 0:
            sys.exit(f"=== {step} failed with exit code {rc}")


if __name__ == "__main__":
    main()
