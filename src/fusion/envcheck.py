"""Pinned environment check for the Kaggle notebooks of steps 4b and 4c.

The installed versions must equal the `name==version` pins of requirements.txt (the local
label of a build such as torch 2.11.0+cu128 is ignored); otherwise the worker stops.
"""
from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

from neural.common import write_json

ROOT = Path(__file__).resolve().parents[2]


def requirement_pins() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==")
            pins[name.strip()] = ver.strip()
    return pins


def check_versions(worker: str, out_dir: Path) -> None:
    rows, bad = {}, []
    for name, ver in requirement_pins().items():
        if name == "pytest":
            continue
        try:
            inst = md.version(name)
        except md.PackageNotFoundError:
            inst = None
        rows[name] = {"pinned": ver, "installed": inst}
        if inst is None or inst.split("+")[0] != ver:  # torch 2.11.0+cu128 matches the pin 2.11.0
            bad.append(name)
    write_json({"packages": rows, "mismatch": bad}, out_dir / f"{worker}.json")
    if bad:
        raise SystemExit(f"installed versions differ from requirements.txt: {bad}")
