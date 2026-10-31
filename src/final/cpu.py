"""Step 5b CPU notebook: scoring of the final predictions, then the speed measurement.

Jobs (--queue, JSON list, in order): assemble (src/final/assemble.py), latency (src/final/latency.py),
posthoc (src/final/posthoc.py, post-hoc significance after the final evaluation),
pack (one zip of results/final and results/runtime without model files, /kaggle/working/final_pack.zip:
the full output with every restored prediction is slow to download).
Run: python src/final/cpu.py --queue '["assemble", "latency"]' --worker cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

from fusion.envcheck import check_versions  # noqa: E402


def pack() -> None:
    import zipfile

    from final import assemble

    results = assemble.OUT.parent
    out = Path("/kaggle/working/final_pack.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in (assemble.OUT.name, "runtime"):
            for f in sorted((results / sub).rglob("*")):
                if f.is_file() and "/models/" not in f.as_posix():
                    z.write(f, f"results/{f.relative_to(results)}")
    print(f"[final-cpu] packed -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    from final import assemble, latency, posthoc

    check_versions(args.worker, assemble.OUT / "environment")
    failed = 0
    for job in json.loads(args.queue):
        try:
            {"assemble": assemble.main, "latency": latency.main, "pack": pack, "posthoc": posthoc.main}[job]()
        except Exception:  # noqa: BLE001 - record and continue
            failed += 1
            (assemble.OUT / "errors").mkdir(parents=True, exist_ok=True)
            (assemble.OUT / "errors" / f"{job}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[final-cpu] {job} FAILED:\n{traceback.format_exc()}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
