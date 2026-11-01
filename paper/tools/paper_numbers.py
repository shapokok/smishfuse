"""Every number printed in the paper, computed from results/, reports/ and configs/ (no number is typed by hand).

Writes paper/numbers.tex: one \\val{key} per value. paper/check_numbers.py imports build() and compares.
Run: python paper/tools/paper_numbers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FINAL = ROOT / "results" / "final"
STATS = ROOT / "reports" / "stats"


def f4(x: float) -> str:
    return f"{x:.4f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def pct1(x: float) -> str:
    return f"{100 * x:.1f}"


def pvalue(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


class Registry:
    def __init__(self):
        self.v: dict[str, tuple[str, str]] = {}

    def add(self, key: str, value, source: str) -> None:
        if key in self.v:
            raise KeyError(f"duplicate key {key}")
        self.v[key] = (str(value), source)


def metrics() -> pd.DataFrame:
    return pd.read_csv(FINAL / "metrics.csv")


def seed_stats(m: pd.DataFrame, cond: str, test: str, model: str, col: str = "macro_f1") -> tuple[float, float]:
    x = m[(m.condition == cond) & (m.test_set == test) & (m.model == model)][col]
    if len(x) != 5:
        raise ValueError(f"{cond}/{test}/{model}: {len(x)} seeds")
    return float(x.mean()), float(x.std(ddof=1))


def sig_row(cond: str, test: str, comparison: str, other: str | None = None) -> pd.Series:
    s = pd.read_csv(FINAL / "significance.csv")
    r = s[(s.condition == cond) & (s.test_set == test) & (s.comparison == comparison)]
    if other is not None:
        r = r[r.other == other]
    if len(r) != 1:
        raise ValueError(f"{cond}/{test}/{comparison}/{other}: {len(r)} rows")
    return r.iloc[0]


def data_numbers(R: Registry) -> None:
    src = "reports/stats/"
    load = {d: json.loads((STATS / f"dedup_{d}.json").read_text()) for d in ("mendeley", "moz_smishing", "super_sms", "uci_sms")}
    ov = json.loads((STATS / "overlap.json").read_text())
    men_after = load["mendeley"]["rows_after"]
    R.add("men-raw", load["mendeley"]["rows_before"], src + "dedup_mendeley.json rows_before")
    R.add("men-clean", men_after, src + "dedup_mendeley.json rows_after")
    clean = next(r for r in ov["mendeley_uci"] if r["data"] == "clean")
    if clean["mendeley_rows"] != men_after:
        raise ValueError("overlap.json and dedup_mendeley.json disagree")
    hit = clean["mendeley_rows_in_uci_exact_or_near"]
    R.add("men-in-uci", hit, src + "overlap.json mendeley_uci[clean] exact_or_near")
    R.add("men-in-uci-pct", pct1(hit / men_after), src + "overlap.json / dedup_mendeley.json")
    R.add("men-removed", load["mendeley"]["rows_before"] - men_after, src + "dedup_mendeley.json rows_before - rows_after")
    R.add("men-conflict-groups", sum(load["mendeley"]["conflict_groups"].values()),
          src + "dedup_mendeley.json conflict_groups (exact + near)")


def main_numbers(R: Registry) -> None:
    m = metrics()
    src = "results/final/metrics.csv (mean over 5 seeds)"
    for key, model in (("cascade", "cascade"), ("svmchar", "svm_char"), ("svmcharaug", "svm_char_aug"),
                       ("distilaug", "distilbert_aug"), ("bert", "bert")):
        mu, sd = seed_stats(m, "mendeley_3class", "clean", model)
        R.add(f"{key}-clean-f1", f4(mu), src)
        R.add(f"{key}-clean-f1-sd", f4(sd), src)
    r = sig_row("mendeley_3class", "clean", "best_baseline")
    if r.other != "svm_char":
        raise ValueError("best baseline on the clean test is not svm_char")
    R.add("clean-best-p-boot-holm", pvalue(r.p_bootstrap_holm), "results/final/significance.csv")
    R.add("clean-best-p-mcn-holm", pvalue(r.p_mcnemar_holm), "results/final/significance.csv")
    shares = m[(m.condition == "mendeley_3class") & (m.model == "cascade")].groupby("test_set").routed_share.mean()
    R.add("routed-min-pct", pct1(shares.min()), "results/final/metrics.csv routed_share (mean over seeds)")
    R.add("routed-max-pct", pct1(shares.max()), "results/final/metrics.csv routed_share (mean over seeds)")
    two = m[m.condition.isin(["setting_mahmud", "transfer"]) & (m.model == "cascade")]
    two = two.groupby(["condition", "test_set"]).routed_share.mean()
    R.add("routed-binary-min-pct", f"{100 * two.min():.0f}", "results/final/metrics.csv routed_share, Mahmud and transfer")
    R.add("routed-binary-max-pct", f"{100 * two.max():.0f}", "results/final/metrics.csv routed_share, Mahmud and transfer")


def published_numbers(R: Registry) -> None:
    """Published settings at the strongest held-out obfuscation level (mean over 5 seeds)."""
    m = metrics()
    src = "results/final/metrics.csv (mean over 5 seeds)"
    pub = {"setting_mahmud": ["mahmud2024_cnn_bigru"],
           "setting_shim": ["shim2024_bert_base", "shim2024_distilbert_base", "shim2024_roberta_base"]}
    vals = [seed_stats(m, c, "heldout20", x)[0] for c, xs in pub.items() for x in xs]
    R.add("pub-ho20-min", f4(min(vals)), src)
    R.add("pub-ho20-max", f4(max(vals)), src)
    for c, k in (("setting_mahmud", "mahmud"), ("setting_shim", "shim")):
        R.add(f"cascade-{k}-ho20", f4(seed_stats(m, c, "heldout20", "cascade")[0]), src)
        R.add(f"svmchar-{k}-ho20", f4(seed_stats(m, c, "heldout20", "svm_char")[0]), src)


def latency_numbers(R: Registry) -> None:
    e2e = json.loads((FINAL / "cascade_end_to_end.json").read_text())
    lat = pd.read_csv(FINAL / "latency.csv").set_index("model")
    R.add("lat-cascade-p50", f2(e2e["p50_ms"]), "results/final/cascade_end_to_end.json")
    R.add("lat-cascade-p95", f2(e2e["p95_ms"]), "results/final/cascade_end_to_end.json")
    R.add("lat-cascade-thr", f"{e2e['throughput_msg_per_s']:.0f}", "results/final/cascade_end_to_end.json")
    R.add("lat-cascade-routed-pct", pct1(e2e["routed_share"]), "results/final/cascade_end_to_end.json")
    R.add("lat-distilaug-p50", f"{lat.loc['distilbert_aug', 'p50_ms']:.1f}", "results/final/latency.csv")
    R.add("lat-svmcharaug-p50", f2(lat.loc["svm_char_aug", "p50_ms"]), "results/final/latency.csv")


def config_numbers(R: Registry) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())
    R.add("n-seeds", len(cfg["seeds"]), "configs/final.yaml seeds")


def build() -> dict[str, tuple[str, str]]:
    R = Registry()
    for fn in (data_numbers, main_numbers, published_numbers, latency_numbers, config_numbers):
        fn(R)
    return R.v


def render(v: dict[str, tuple[str, str]]) -> str:
    out = ["% Generated by paper/tools/paper_numbers.py from results/, reports/ and configs/. Do not edit by hand.",
           "\\makeatletter"]
    for k, (val, src) in v.items():
        out.append(f"\\@namedef{{val@{k}}}{{{val}}}% {src}")
    out += ["\\makeatother",
            "\\newcommand{\\val}[1]{\\ifcsname val@#1\\endcsname\\csname val@#1\\endcsname"
            "\\else\\textbf{??#1??}\\fi}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    v = build()
    (PAPER / "numbers.tex").write_text(render(v), encoding="utf-8")
    print(f"wrote paper/numbers.tex ({len(v)} values)")
