"""Step 5b: packs the useful part of the smishfuse-final-cpu output into one zip (no model weights).

The full output is slow to download; this CPU notebook reads it as a kernel source and writes
/kaggle/working/final_pack.zip with results/final (metrics, significance, tau, latency, figure data,
levels, environment, job times, errors, predictions) and results/runtime. Model files are skipped.
Run (Kaggle): python src/final/pack.py --queue '["pack"]' --worker pack
"""
from __future__ import annotations

import argparse
import glob
import sys
import zipfile
from pathlib import Path

SKIP = ("/models/", "/latency_retrained/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    ap.parse_args()
    hits = sorted(glob.glob("/kaggle/input/**/results/final/metrics.csv", recursive=True))
    if not hits:
        sys.exit("smishfuse-final-cpu output not found under /kaggle/input")
    results = Path(hits[0]).parents[1]
    out = Path("/kaggle/working/final_pack.zip")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("final", "runtime"):
            for f in sorted((results / sub).rglob("*")):
                if f.is_file() and not any(s in f.as_posix() for s in SKIP):
                    z.write(f, f"results/{f.relative_to(results)}")
                    n += 1
    print(f"[pack] {n} files from {results} -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
