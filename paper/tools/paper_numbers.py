"""Every number printed in the paper, computed from results/, reports/ and configs/ (no number is typed by hand).

Writes paper/numbers.tex: one \\val{key} per value. paper/check_numbers.py imports build() and compares.
Run: python paper/tools/paper_numbers.py
"""
from __future__ import annotations

import ast
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
    sp = json.loads((STATS / "split_mendeley.json").read_text())["labels"]["test"]
    R.add("men-test-n", sum(sp.values()), src + "split_mendeley.json")
    R.add("men-test-smishing", sp["smishing"], src + "split_mendeley.json")
    R.add("men-test-spam", sp["spam"], src + "split_mendeley.json")
    R.add("shim-test-n", sp["smishing"] + sp["spam"], src + "split_mendeley.json (smishing + spam)")
    ssc = json.loads((STATS / "super_sms_clean.json").read_text())
    R.add("ssc-removed", sum(ssc["removed"].values()), src + "super_sms_clean.json removed")
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


def const(path: str, name: str):
    """Value of a module-level constant NAME = <literal> in a source file (read, not imported)."""
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not in {path}")


def cfg(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())


def pct0(x: float) -> str:
    return f"{100 * x:.0f}"


def sci(x: float, digits: int = 2) -> str:
    m, e = f"{x:.{digits}e}".split("e")
    m = m.rstrip("0").rstrip(".") if "." in m else m
    return f"{m}\\times 10^{{{int(e)}}}" if int(e) else m


def config_numbers(R: Registry) -> None:
    fin, s2, s3, s4a, s4b, s4c, s5a = (cfg(n) for n in ("final", "step2", "step3", "step4a", "step4b", "step4c", "step5a"))
    R.add("n-seeds", len(fin["seeds"]), "configs/final.yaml seeds")
    R.add("seed-last", fin["seeds"][-1], "configs/final.yaml seeds")
    # data (src/data)
    R.add("shingle-k", const("src/data/common.py", "SHINGLE_K"), "src/data/common.py SHINGLE_K")
    R.add("jaccard-thr", const("src/data/common.py", "JACCARD_THRESHOLD"), "src/data/common.py JACCARD_THRESHOLD")
    R.add("num-perm", const("src/data/common.py", "NUM_PERM"), "src/data/common.py NUM_PERM")
    R.add("lsh-thr", const("src/data/common.py", "LSH_CANDIDATE_THRESHOLD"), "src/data/common.py")
    R.add("check-lsh-thr", const("src/data/split.py", "CHECK_LSH_THRESHOLD"), "src/data/split.py")
    fr = const("src/data/split.py", "FRACTIONS")
    for k in ("train", "val", "test"):
        R.add(f"split-{k}", pct0(fr[k]), "src/data/split.py FRACTIONS")
    R.add("split-seed", const("src/data/common.py", "SEED"), "src/data/common.py SEED")
    # metrics
    R.add("ece-bins", s2["protocol"]["ece_bins"], "configs/step2.yaml ece_bins")
    lv = s4c["eval_levels"]
    for i, x in enumerate(lv):
        R.add(f"lvl-{'abc'[i]}", pct0(x), "configs/step4c.yaml eval_levels")
    R.add("obf-seed", const("src/eval/obfuscation.py", "SEED"), "src/eval/obfuscation.py SEED")
    R.add("aug-p", s4c["augmentation"]["p"], "configs/step4c.yaml augmentation")
    R.add("aug-max", s4c["augmentation"]["max_level"], "configs/step4c.yaml augmentation")
    # character model
    ng = s4c["char_branch"]["ngram_range"]
    R.add("char-ng-lo", ng[0], "configs/step4c.yaml char_branch")
    R.add("char-ng-hi", ng[1], "configs/step4c.yaml char_branch")
    bp = json.loads((ROOT / "results" / "best_params" / "mendeley_3class" / "linear_svm.json").read_text())
    R.add("svm-c", f"{bp['model_params']['C']:.3f}", "results/best_params/mendeley_3class/linear_svm.json")
    R.add("calib-folds", s2["models"]["linear_svm"]["fixed"]["calibration_cv"], "configs/step2.yaml linear_svm")
    # stage 2
    t = s4a["text"]
    R.add("distil-lr", sci(t["lr"]), "configs/step4a.yaml text.lr")
    R.add("distil-batch", t["batch_size"], "configs/step4a.yaml text")
    R.add("distil-epochs", t["max_epochs"], "configs/step4a.yaml text")
    R.add("distil-patience", t["patience"], "configs/step4a.yaml text")
    R.add("max-len", t["max_len"], "configs/step4a.yaml text")
    R.add("fusion-d", s4a["fusion"]["d"], "configs/step4a.yaml fusion.d")
    R.add("aux-w", s4a["fusion"]["binary_loss_weight"], "configs/step4a.yaml fusion")
    R.add("hand-n", len(const("src/fusion/features.py", "FEATURES")), "src/fusion/features.py FEATURES")
    R.add("svd-k", s4c["char_branch"]["svd_components"], "configs/step4c.yaml char_branch")
    # cascade
    R.add("tau", f4(fin["method"]["tau"]), "configs/final.yaml method.tau")
    R.add("tau-noaug", f4(fin["ablations"]["cascade_noaug"]["tau"]), "configs/final.yaml ablations")
    seeds = [k for k in fin["seeds"] if (ROOT / "results" / "step4c" / "svm_char_aug" / f"clean_seed{k}.csv").exists()]
    if seeds != list(range(seeds[0], seeds[-1] + 1)):
        raise ValueError(f"tau seeds not contiguous: {seeds}")
    R.add("tau-seeds", f"{seeds[0]}--{seeds[-1]}", "scripts/freeze_tau.py: seeds with step-4c val probabilities")
    frozen = json.loads((FINAL / "tau.json").read_text())["mendeley_3class"]
    if frozen["cascade"] != fin["method"]["tau"] or frozen["cascade_noaug"] != fin["ablations"]["cascade_noaug"]["tau"]:
        raise ValueError("results/final/tau.json differs from configs/final.yaml")
    # cost and tests
    for i, r in enumerate(fin["metrics"]["cost_ratios"]):
        R.add(f"cost-r{i + 1}", r, "configs/final.yaml cost_ratios")
    sig = fin["significance"]
    R.add("boot-n", f"{sig['bootstrap']['n']:,}".replace(",", "\\,"), "configs/final.yaml significance")
    R.add("boot-seed", sig["bootstrap"]["seed"], "configs/final.yaml significance")
    R.add("alpha", sig["alpha"], "configs/final.yaml significance")
    import inspect
    sys_path = str(ROOT / "src")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from eval import stats
    R.add("mcn-exact", inspect.signature(stats.mcnemar).parameters["exact_threshold"].default, "src/eval/stats.py mcnemar")
    # protocol
    R.add("optuna-trials", s2["protocol"]["n_trials"], "configs/step2.yaml n_trials")
    if s3["protocol"]["n_trials"] != s2["protocol"]["n_trials"]:
        raise ValueError("steps 2 and 3 use different numbers of trials")
    q = s3["models"]["qwen_fewshot"]["fixed"]
    R.add("qwen-shots", q["n_shots"], "configs/step3.yaml qwen_fewshot")
    R.add("bert-epochs", s3["models"]["bert"]["fixed"]["max_epochs"], "configs/step3.yaml bert")
    for i, x in enumerate(fin["conditions"]["low_data"]["fractions"]):
        R.add(f"ld-{'abcd'[i]}", pct0(x), "configs/final.yaml low_data fractions")
    R.add("ld-min", s4b["low_data"]["min_per_class"], "configs/step4b.yaml low_data")
    R.add("tr-frac", pct0(s4b["transfer"]["finetune_fraction"]), "configs/step4b.yaml transfer")
    mh = s5a["published"]["mahmud2024_cnn_bigru"]
    R.add("mh-w2v", mh["word2vec"]["vector_size"], "configs/step5a.yaml mahmud")
    R.add("mh-filters", mh["conv"]["filters"], "configs/step5a.yaml mahmud")
    R.add("mh-kernel", mh["conv"]["kernel"], "configs/step5a.yaml mahmud")
    R.add("mh-pool", mh["pool"], "configs/step5a.yaml mahmud")
    R.add("mh-gru", mh["gru"]["units"], "configs/step5a.yaml mahmud")
    R.add("mh-drop", mh["gru"]["dropout"], "configs/step5a.yaml mahmud")
    R.add("mh-batch", mh["batch_size"], "configs/step5a.yaml mahmud")
    R.add("mh-epochs", mh["epochs"], "configs/step5a.yaml mahmud")
    R.add("mh-maxlen", mh["max_len"], "configs/step5a.yaml mahmud")
    sh = s5a["published"]["shim2024"]
    R.add("sh-batches", ", ".join(str(b) for b in sh["batch_sizes"]), "configs/step5a.yaml shim")
    R.add("sh-lrs", ", ".join(sci(x, 0) for x in sh["lrs"]), "configs/step5a.yaml shim")
    R.add("sh-epochs", sh["max_epochs"], "configs/step5a.yaml shim")
    R.add("sh-inner", pct0(sh["inner_val"]), "configs/step5a.yaml shim")
    lat = fin["latency"]["per_model"]
    R.add("lat-msgs", f"{lat['messages']:,}".replace(",", "\\,"), "configs/final.yaml latency")
    R.add("lat-warmup", lat["warmup"], "configs/final.yaml latency")


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
