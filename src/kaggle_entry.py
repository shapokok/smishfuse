"""Entry point executed inside a Kaggle kernel (see scripts/make_kernels.py).

* locates the smishfuse-data input (directory that contains mendeley/train.csv);
* restores finished work of this kernel from the smishfuse-state input, if any
  (state/<kernel>/results/... -> /kaggle/working/results, existing files kept);
* runs the spec: kind "neural" (one worker process per GPU queue, pinned with
  CUDA_VISIBLE_DEVICES), "classic" (src/baselines/classic.py), "latency", "fusion" (step 4a) or "stress" (step 4b);
* writes results/runtime/<kernel>.json: wall time, hardware, library versions.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
WORK = Path("/kaggle/working")
RESULTS = WORK / "results"


def find_data() -> Path:
    hits = sorted(glob.glob("/kaggle/input/**/mendeley/train.csv", recursive=True))
    if not hits:
        # The dataset may arrive as archives (uploaded with --dir-mode zip): unpack and search again.
        tmp = Path("/tmp/smishfuse_data")
        for a in glob.glob("/kaggle/input/**/smishfuse-data/**/*.zip", recursive=True):
            zipfile.ZipFile(a).extractall(tmp / Path(a).stem)
        hits = sorted(glob.glob(str(tmp / "**" / "mendeley" / "train.csv"), recursive=True))
    if not hits:
        raise SystemExit("smishfuse-data not found under /kaggle/input")
    return Path(hits[0]).parents[1]


def restore_state(kernel: str) -> int:
    n = 0
    for root in sorted(glob.glob(f"/kaggle/input/**/state/{kernel}/results", recursive=True)):
        for src in Path(root).rglob("*"):
            if src.is_file():
                dst = RESULTS / src.relative_to(root)
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    n += 1
    return n


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "cpu_model": sh("lscpu | grep 'Model name' | sed 's/.*: *//'"), "cpu_count": os.cpu_count(),
           "gpus": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
           "packages": {}}
    for pkg in ["torch", "transformers", "bitsandbytes", "accelerate", "optuna", "scikit-learn", "numpy",
                "pandas", "xgboost", "scipy", "pyspellchecker"]:
        env["packages"][pkg] = sh(f"{sys.executable} -m pip show {pkg} 2>/dev/null | grep '^Version' | cut -d' ' -f2")
    return env


def run_neural(kernel: str, spec: dict, env_vars: dict) -> list[int]:
    procs = []
    for i, queue in enumerate(spec["queues"]):
        if not queue:
            continue
        env = {**os.environ, **env_vars, "CUDA_VISIBLE_DEVICES": str(i)}
        cmd = [sys.executable, str(CODE / "src" / "neural" / "run.py"), "--mode", spec["mode"],
               "--jobs", json.dumps(queue), "--worker", f"{kernel}_gpu{i}",
               "--overrides", json.dumps(spec.get("overrides", {}))]
        procs.append(subprocess.Popen(cmd, env=env))
    return [p.wait() for p in procs]


def run_stress(kernel: str, spec: dict, env_vars: dict) -> list[int]:
    """Steps 4b / 4c / 5a: one worker per GPU, each running its queue with spec["script"]
    (default src/fusion/stress.py: conditions; src/fusion/v2.py: models; src/step5a/run.py: jobs),
    then the optional spec["post"] queue once all workers have finished."""
    procs = []
    for i, conditions in enumerate(spec["queues"]):
        if not conditions:
            continue
        env = {**os.environ, **env_vars, "CUDA_VISIBLE_DEVICES": str(i)}
        cmd = [sys.executable, str(CODE / spec.get("script", "src/fusion/stress.py")), "--queue", json.dumps(conditions),
               "--worker", f"{kernel}_gpu{i}"]
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    if spec.get("post"):  # jobs that need the output of every queue (step 5a: comparison), run after them
        cmd = [sys.executable, str(CODE / spec["script"]), "--queue", json.dumps(spec["post"]),
               "--worker", f"{kernel}_post"]
        rcs.append(subprocess.run(cmd, env={**os.environ, **env_vars}).returncode)
    return rcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.loads(args.spec)
    start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    env_vars = {"SMISHFUSE_DATA": str(find_data()), "SMISHFUSE_RESULTS": str(RESULTS),
                "PYTHONPATH": str(CODE / "src"), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                # no new trial / seed / LLM batch after this point (session budget, default 8.5 h)
                "SMISHFUSE_DEADLINE": str(start + 3600 * float(spec.get("budget_hours", 8.5)))}
    # A continuation kernel (new name, so the first session's output with model weights stays
    # available as a kernel source) restores the state of the kernel named in state_from.
    sources = spec.get("state_from", args.kernel)
    restored = sum(restore_state(k) for k in ([sources] if isinstance(sources, str) else sources))
    print(f"[entry] kernel {args.kernel}, data {env_vars['SMISHFUSE_DATA']}, restored {restored} files", flush=True)
    info = {"kernel": args.kernel, "spec": spec, "restored_files": restored, "environment": environment(),
            "start_unix": start}
    if spec["kind"] == "neural":
        rcs = run_neural(args.kernel, spec, env_vars)
    elif spec["kind"] == "classic":
        cmd = [sys.executable, str(CODE / "src" / "baselines" / "classic.py"), *spec.get("tasks", []),
               f"--models={','.join(spec['models'])}"]
        rcs = [subprocess.run(cmd, env={**os.environ, **env_vars}).returncode]
    elif spec["kind"] == "stress":
        rcs = run_stress(args.kernel, spec, env_vars)
    elif spec["kind"] == "fusion":
        cmd = [sys.executable, str(CODE / "src" / "fusion" / "run.py")]
        rcs = [subprocess.run(cmd, env={**os.environ, **env_vars}).returncode]
    elif spec["kind"] == "latency":
        cmd = [sys.executable, str(CODE / "src" / "latency.py")]
        rcs = [subprocess.run(cmd, env={**os.environ, **env_vars}).returncode]
    else:
        raise SystemExit(f"unknown kind {spec['kind']}")
    info.update({"end_unix": time.time(), "wall_seconds": time.time() - start, "return_codes": rcs})
    out = RESULTS / "runtime" / f"{args.kernel}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {"sessions": []}
    prev["sessions"].append(info)
    out.write_text(json.dumps(prev, indent=2) + "\n")
    print(f"[entry] done in {info['wall_seconds'] / 3600:.2f} h, return codes {rcs}", flush=True)
    if any(rcs):
        sys.exit(1)


if __name__ == "__main__":
    main()
