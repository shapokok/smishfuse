"""Shared utilities for SmishFuse step 1 (data loading, cleaning, splitting).

Everything here is deterministic: fixed seeds, fixed hash functions, sorted outputs.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
STATS = REPORTS / "stats"
REMOVED = REPORTS / "removed"
CONFLICTS = REPORTS / "conflicts"

SCHEMA = ["id", "text", "label", "dataset", "language"]
SEED = 42

# Extra columns added to clean.csv and the splits of some datasets.
# Mendeley: label_binary = 0 for ham, 1 for spam and smishing.
EXTRA_COLUMNS = {"mendeley": ["label_binary"]}


def add_extra_columns(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if name == "mendeley":
        df["label_binary"] = (df["label"] != "ham").astype(int)
    return df[SCHEMA + EXTRA_COLUMNS.get(name, [])]

# Near-duplicate parameters (step 4/5/7).
SHINGLE_K = 5
JACCARD_THRESHOLD = 0.9
NUM_PERM = 128
# LSH is only used to retrieve candidates; every candidate pair is then verified
# with the exact Jaccard similarity of the shingle sets. A candidate threshold
# below 0.9 keeps the LSH false-negative rate at J >= 0.9 negligible.
LSH_CANDIDATE_THRESHOLD = 0.7

# Pinned sources. sha256 values were recorded at first download (2026-10-28);
# a mismatch means the upstream file changed and the run stops.
SOURCES = {
    "mendeley": {
        "url": "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
        "edb361de-918d-469f-9106-e84823830665/file_downloaded",
        "path": RAW / "mendeley" / "Dataset_5971.zip",
        "sha256": "9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3",
    },
    "moz_smishing": {
        "url": "https://huggingface.co/datasets/MOZNLP/MOZ-Smishing/resolve/"
        "1092f9d9a545b29ae6be030ee9713b615fc2d987/test.csv",
        "path": RAW / "moz_smishing" / "test.csv",
        "sha256": "814a11d9b05741c4b47eb0d0784b1fd12a2a076f83a714a9908bdda594986ab8",
    },
    "super_sms": {
        "url": "https://raw.githubusercontent.com/smspamresearch/spstudy/"
        "a0889f36fb59e23a7f853a6932485d93d18b8afc/Data/super_sms_dataset.csv",
        "path": RAW / "super_sms" / "super_sms_dataset.csv",
        "sha256": "ba825b468b59fb229cc56c15a6c2d14d593a88051d54ea2f402166bf2ab894d3",
    },
    "uci_sms": {
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "path": RAW / "uci_sms" / "sms_spam_collection.zip",
        "sha256": "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> Path:
    """Download the pinned raw file if absent, then verify its checksum."""
    src = SOURCES[name]
    path: Path = src["path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] downloading {src['url']}")
        r = requests.get(src["url"], timeout=300)
        r.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.rename(path)
    digest = sha256_of(path)
    if digest != src["sha256"]:
        raise RuntimeError(
            f"[{name}] sha256 mismatch for {path}: expected {src['sha256']}, got {digest}. "
            "The upstream file differs from the one this pipeline was built on."
        )
    return path


class _FallbackCounter:
    count = 0


def _cp1252_fallback(err: UnicodeDecodeError):
    _FallbackCounter.count += err.end - err.start
    return err.object[err.start : err.end].decode("cp1252", errors="replace"), err.end


codecs.register_error("smishfuse_cp1252", _cp1252_fallback)


def decode_utf8_with_cp1252_fallback(data: bytes) -> tuple[str, int]:
    """Decode UTF-8; bytes that are not valid UTF-8 are decoded as cp1252.

    Returns the text and the number of bytes that needed the fallback.
    """
    _FallbackCounter.count = 0
    text = data.decode("utf-8", errors="smishfuse_cp1252")
    return text, _FallbackCounter.count


_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalization used for duplicate detection: lowercase + collapse whitespace."""
    return _WS.sub(" ", text.lower()).strip()


def _detect_one(text: str) -> str:
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    # langdetect seeds every Detector from the factory seed, so the result
    # depends only on the text, not on order or process.
    DetectorFactory.seed = 0
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def detect_languages(texts) -> list[str]:
    """langdetect with a fixed seed; 'unknown' when no language features exist."""
    from multiprocessing import Pool

    with Pool() as pool:
        return pool.map(_detect_one, texts, chunksize=256)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finalize_load(
    name: str,
    df: pd.DataFrame,
    invalid: pd.DataFrame,
    expected: dict,
    extra_stats: dict | None = None,
) -> None:
    """Common tail of every load_<name>.py script.

    df: valid rows with columns id, text, label (row order = raw file order).
    invalid: rows dropped at load time with columns id, text, label, reason.
    expected: {"total": int|None, "labels": {label: int}} from the dataset description.
    """
    df = df.copy()
    df["dataset"] = name
    print(f"[{name}] detecting language for {len(df)} messages ...")
    df["language"] = detect_languages(df["text"].tolist())
    df = df[SCHEMA]
    write_csv(df, PROCESSED / name / "unified.csv")

    invalid = invalid.copy()
    invalid["dataset"] = name
    write_csv(
        invalid.reindex(columns=["dataset", "id", "text", "label", "reason"]),
        REMOVED / f"{name}_load.csv",
    )

    raw_rows = len(df) + len(invalid)
    actual_labels = df["label"].value_counts().sort_index().to_dict()
    checks = []
    if expected.get("total") is not None:
        checks.append({"what": "total rows in raw file", "expected": expected["total"], "actual": raw_rows})
        checks.append({"what": "valid rows after load", "expected": expected["total"], "actual": len(df)})
    for lab, n in expected.get("labels", {}).items():
        checks.append({"what": f"label '{lab}'", "expected": n, "actual": int(actual_labels.get(lab, 0))})
    for c in checks:
        c["match"] = c["expected"] == c["actual"]

    stats = {
        "dataset": name,
        "raw_rows": raw_rows,
        "valid_rows": len(df),
        "invalid_rows": invalid["reason"].value_counts().to_dict() if len(invalid) else {},
        "labels": {k: int(v) for k, v in actual_labels.items()},
        "languages": {k: int(v) for k, v in df["language"].value_counts().to_dict().items()},
        "mean_len_chars": round(float(df["text"].str.len().mean()), 1),
        "median_len_chars": float(df["text"].str.len().median()),
        "checks": checks,
        **(extra_stats or {}),
    }
    write_json(stats, STATS / f"load_{name}.json")

    print(f"[{name}] valid rows: {len(df)}  dropped at load: {len(invalid)}")
    print(f"[{name}] labels: {stats['labels']}")
    for c in checks:
        flag = "OK " if c["match"] else "MISMATCH"
        print(f"[{name}] {flag} {c['what']}: expected {c['expected']}, actual {c['actual']}")
