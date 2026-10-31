"""Kaggle plumbing for SmishFuse: private datasets, kernel push, status and outputs.

  python scripts/kaggle_sync.py data            # create / new version of smishfuse-data (data/processed)
  python scripts/kaggle_sync.py code            # new version of smishfuse-code if src/configs/... changed
  python scripts/kaggle_sync.py state [KERNEL]  # upload collected results as resume state for kernels
  python scripts/kaggle_sync.py push KERNEL     # push (= run) kaggle/kernels/KERNEL
  python scripts/kaggle_sync.py status KERNEL
  python scripts/kaggle_sync.py output KERNEL   # -> results/kaggle_runs/KERNEL (model weights and code skipped)

Datasets are always created private. Credentials are read by the Kaggle CLI from
~/.kaggle; this script never reads or prints them.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
BUILD = ROOT / "kaggle" / "build"
CFG = yaml.safe_load((ROOT / "configs" / "kaggle_kernels.yaml").read_text(encoding="utf-8"))
OWNER = CFG["owner"]
# results/step4c, results/step4b/baseline (val predictions) and results/step5a/inputs are inputs of step 5a
# results/raw (clean test predictions of steps 2-3), results/latency (Qwen GPU row), results/step4b/*/subsets:
# inputs of step 5b
CODE_PARTS = ["src", "configs", "tests", "requirements.txt", "results/best_params", "results/step4c",
              "results/step4b/baseline", "results/step5a/inputs", "results/raw", "results/latency",
              "results/step4b/low_data/subsets", "results/step4b/transfer/subsets", "data/kz_cases"]
OUTPUT_PATTERN = r"^(?!.*(/models/|^models/|/code/|^code/)).*$"


def kaggle(*args, check=True, capture=False, timeout=None) -> str:
    r = subprocess.run([str(KAGGLE), *args], capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args[:2])} failed ({r.returncode}): {r.stderr if capture else ''}")
    return (r.stdout or "") if capture else ""


def metadata(slug: str, title: str) -> dict:
    return {"title": title, "id": f"{OWNER}/{slug}", "licenses": [{"name": "other"}]}


def exists(slug: str) -> bool:
    r = subprocess.run([str(KAGGLE), "datasets", "status", f"{OWNER}/{slug}"], capture_output=True, text=True)
    return r.returncode == 0 and any(w in r.stdout.lower() for w in ("ready", "pending"))


def wait_ready(slug: str, timeout: int = 900) -> None:
    start = time.time()
    while time.time() - start < timeout:
        out = kaggle("datasets", "status", f"{OWNER}/{slug}", check=False, capture=True).strip().lower()
        if out == "ready":
            print(f"[{slug}] ready")
            return
        time.sleep(15)
    raise SystemExit(f"[{slug}] not ready after {timeout}s")


def upload(slug: str, folder: Path, message: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps(metadata(slug, slug), indent=2) + "\n")
    if exists(slug):
        kaggle("datasets", "version", "-p", str(folder), "-m", message, "--dir-mode", "zip", "-q")
    else:
        kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "zip", "-q")  # private by default
    wait_ready(slug)


def fresh(folder: Path) -> Path:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


def tree_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for base in paths:
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for f in files:
            if "__pycache__" in f.parts or f.suffix == ".pyc":
                continue
            h.update(str(f.relative_to(ROOT)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def cmd_data():
    folder = fresh(BUILD / "smishfuse-data")
    shutil.copytree(ROOT / "data" / "processed", folder / "processed")
    upload("smishfuse-data", folder, "data/processed from step 1")


def cmd_code():
    parts = [ROOT / p for p in CODE_PARTS]
    digest = tree_hash(parts)
    stamp = BUILD / "smishfuse-code.sha256"
    if stamp.exists() and stamp.read_text().strip() == digest:
        print(f"[smishfuse-code] unchanged ({digest[:12]}), no new version")
        return
    folder = fresh(BUILD / "smishfuse-code")
    for p in parts:
        dst = folder / p.relative_to(ROOT)
        if p.is_dir():
            shutil.copytree(p, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    (folder / "CODE_SHA256").write_text(digest + "\n")
    upload("smishfuse-code", folder, f"code {digest[:12]}")
    stamp.write_text(digest + "\n")


def cmd_state(kernels: list[str]):
    folder = fresh(BUILD / "smishfuse-state")
    (folder / "README.md").write_text("Resume state for SmishFuse Kaggle kernels (collected outputs).\n")
    (folder / "state").mkdir()
    for k in kernels:
        src = ROOT / "results" / "kaggle_runs" / k / "results"
        if src.exists():
            shutil.copytree(src, folder / "state" / k / "results")
            print(f"[state] {k}: {sum(1 for _ in src.rglob('*') if _.is_file())} files")
    upload("smishfuse-state", folder, "resume state: " + (", ".join(kernels) or "empty"))


def cmd_push(name: str):
    meta = json.loads((ROOT / "kaggle" / "kernels" / name / "kernel-metadata.json").read_text())
    args = ["kernels", "push", "-p", str(ROOT / "kaggle" / "kernels" / name)]
    if meta.get("enable_gpu"):
        args += ["--accelerator", meta["machine_shape"]]
    print(kaggle(*args, capture=True))


def cmd_status(name: str):
    print(kaggle("kernels", "status", f"{OWNER}/{name}", capture=True).strip())


def cmd_output(name: str):
    dst = ROOT / "results" / "kaggle_runs" / name
    dst.mkdir(parents=True, exist_ok=True)
    # The output download sometimes hangs or drops the connection: bounded time, a few retries.
    for attempt in range(1, 6):
        try:
            kaggle("kernels", "output", f"{OWNER}/{name}", "-p", str(dst), "--file-pattern", OUTPUT_PATTERN,
                   "-o", "-q", timeout=900)
            break
        except (subprocess.TimeoutExpired, SystemExit) as e:
            print(f"[{name}] download attempt {attempt} failed: {type(e).__name__}")
            if attempt == 5:
                raise
            time.sleep(30)
    print(f"[{name}] outputs in {dst}: {sum(1 for p in dst.rglob('*') if p.is_file())} files")


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    {"data": lambda: cmd_data(), "code": lambda: cmd_code(), "state": lambda: cmd_state(rest),
     "push": lambda: cmd_push(rest[0]), "status": lambda: cmd_status(rest[0]),
     "output": lambda: cmd_output(rest[0])}[cmd]()


if __name__ == "__main__":
    main(sys.argv[1:])
