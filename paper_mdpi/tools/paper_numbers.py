"""Every number printed in the paper, computed from results/, reports/ and configs/ (no number is typed by hand).

Writes paper_mdpi/numbers.tex: one \\val{key} per value. paper_mdpi/check_numbers.py imports build() and compares.
Run: python paper_mdpi/tools/paper_numbers.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_mdpi"
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


def signed(x: float, digits: int = 2) -> str:
    return f"$-${abs(x):.{digits}f}" if x < 0 else f"$+${x:.{digits}f}"


def m3(m, test, model, col="macro_f1"):
    return seed_stats(m, "mendeley_3class", test, model, col)[0]


def results_numbers(R: Registry) -> None:
    m = metrics()
    src = "results/final/metrics.csv (mean over 5 seeds)"
    sig = "results/final/significance.csv"
    # main comparison
    r = sig_row("mendeley_3class", "clean", "best_baseline")
    R.add("clean-diff-abs", f"{abs(100 * r.diff_macro_f1_seed1):.2f}", sig + " (seed 1, cascade below)")
    if r.diff_macro_f1_seed1 >= 0:
        raise ValueError("text says the cascade is below the char SVM on seed 1")
    R.add("clean-ci-lo", signed(100 * r.ci95_low), sig)
    R.add("clean-ci-hi", signed(100 * r.ci95_high), sig)
    R.add("qwen-clean-f1", f4(m3(m, "clean", "qwen_fewshot")), src)
    cm = pd.read_csv(FINAL / "confusion_cascade_seed1.csv").set_index("y_true")
    R.add("cm-spam-to-smish", cm.loc["spam", "smishing"], "results/final/confusion_cascade_seed1.csv")
    R.add("cm-smish-to-spam", cm.loc["smishing", "spam"], "results/final/confusion_cascade_seed1.csv")
    R.add("cm-ham-wrong", int(cm.loc["ham"].sum() - cm.loc["ham", "ham"]), "results/final/confusion_cascade_seed1.csv")
    R.add("cm-ham-n", int(cm.loc["ham"].sum()), "results/final/confusion_cascade_seed1.csv")
    for k, x in (("cascade", "cascade"), ("svmcharaug", "svm_char_aug"), ("distilaug", "distilbert_aug")):
        R.add(f"ece-{k}", f4(m3(m, "clean", x, "ece")), src)
    if not m3(m, "clean", "cascade", "ece") < min(m3(m, "clean", "svm_char_aug", "ece"), m3(m, "clean", "distilbert_aug", "ece")):
        raise ValueError("cascade ECE is not below both stages")
    ss = m[(m.condition == "super_sms") & (m.test_set == "clean") & (m.model != "qwen_fewshot")].groupby("model").macro_f1.mean()
    R.add("ss-min", f4(ss.min()), src + ", super_sms without the few-shot model")
    R.add("ss-max", f4(ss.max()), src + ", super_sms without the few-shot model")
    R.add("ss-cascade", f4(ss["cascade"]), src)
    R.add("ss-bert", f4(ss["bert"]), src)
    r = sig_row("super_sms", "clean", "best_baseline")
    if r.other != "bert":
        raise ValueError("Super SMS best baseline is not BERT")
    R.add("ss-p-boot", pvalue(r.p_bootstrap_holm), sig)
    R.add("ss-p-mcn", pvalue(r.p_mcnemar_holm), sig)
    # published settings
    claims = yaml.safe_load((ROOT / "results" / "step5a" / "published_claims.yaml").read_text())["on_mendeley"]
    R.add("claim-mh-acc", f4(claims["mahmud2024"]["reported_table7_dataset3"]["CNN-BiGRU"]["accuracy"]),
          "results/step5a/published_claims.yaml (from the paper)")
    f1s = [v["f1"] for v in claims["shim2024"]["reported_table1_original_data"].values()]
    R.add("claim-sh-min", f"{min(f1s):.1f}", "results/step5a/published_claims.yaml (from the paper, %)")
    R.add("claim-sh-max", f"{max(f1s):.1f}", "results/step5a/published_claims.yaml (from the paper, %)")
    R.add("mh-ho20", f4(seed_stats(m, "setting_mahmud", "heldout20", "mahmud2024_cnn_bigru")[0]), src)
    sh = [seed_stats(m, "setting_shim", "heldout20", x)[0] for x in
          ("shim2024_bert_base", "shim2024_distilbert_base", "shim2024_roberta_base")]
    R.add("sh-ho20-min", f4(min(sh)), src)
    R.add("sh-ho20-max", f4(max(sh)), src)
    # robustness
    lv = json.loads((FINAL / "levels" / "mendeley_3class_heldout20.json").read_text())
    R.add("realised-ho20", f"{100 * lv['realised_share']:.1f}", "results/final/levels/mendeley_3class_heldout20.json")
    neural = ["bert", "textcnn", "bilstm", "distilbert_step3", "smishfuse_v1"]
    v = [m3(m, "heldout20", x) for x in neural]
    R.add("rob-neural-ho20-min", f4(min(v)), src)
    R.add("rob-neural-ho20-max", f4(max(v)), src)
    for k, x in (("distilaug", "distilbert_aug"), ("distil", "distilbert"), ("svmchar", "svm_char"),
                 ("cascade", "cascade"), ("v1", "smishfuse_v1"), ("v2", "smishfuse_v2")):
        R.add(f"rob-{k}-ho20", f4(m3(m, "heldout20", x)), src)
    shares = m[(m.condition == "mendeley_3class") & (m.model == "cascade")].groupby("test_set").routed_share.mean()
    R.add("routed-clean-pct", pct1(shares["clean"]), src + " routed_share")
    R.add("tr05-p-boot", pvalue(sig_row("mendeley_3class", "train05", "best_baseline").p_bootstrap_holm), sig)
    # low data and transfer
    def ld(frac, x):
        return seed_stats(m, f"low_data_frac{frac}", "clean", x)[0]
    R.add("ld10-cascade", f4(ld("10", "cascade")), src)
    R.add("ld10-svmchar", f4(ld("10", "svm_char")), src)
    R.add("ld01-cascade", f4(ld("01", "cascade")), src)
    R.add("ld01-svmchar", f4(ld("01", "svm_char")), src)
    R.add("ld01-distilaug", f4(ld("01", "distilbert_aug")), src)
    R.add("ld01-share", pct1(seed_stats(m, "low_data_frac01", "clean", "cascade", "routed_share")[0]), src)
    r = sig_row("low_data_frac01", "clean", "best_baseline")
    R.add("ld01-b", int(r.mcnemar_b_cascade_only_correct), sig)
    R.add("ld01-c", int(r.mcnemar_c_other_only_correct), sig)
    abl = {"cascade", "cascade_noaug", "average", "smishfuse_v1", "smishfuse_v2", "smishfuse_v2_noaug"}
    for frac in ("01", "05", "10", "25"):
        g = m[(m.condition == f"low_data_frac{frac}") & (m.test_set == "clean")].groupby("model").macro_f1.mean()
        if g.drop([x for x in abl if x in g.index]).idxmax() != "svm_char":
            raise ValueError(f"char SVM is not the best baseline at {frac}")
    for frac in ("01", "05"):
        R.add(f"ld{frac}-average", f4(ld(frac, "average")), src)
    zs = m[(m.condition == "transfer") & (m.test_set == "zeroshot")].groupby("model").macro_f1.mean()
    R.add("tr-zs-best", f4(zs.max()), src)
    taus = json.loads((FINAL / "tau.json").read_text())
    R.add("tau-tr-zs", f4(taus["transfer"]["cascade:zeroshot"]["tau"]), "results/final/tau.json")
    R.add("tau-tr-ft", f4(taus["transfer"]["cascade:finetune10"]["tau"]), "results/final/tau.json")
    R.add("tau-mh", f4(taus["setting_mahmud"]["cascade:clean"]["tau"]), "results/final/tau.json")
    R.add("tr-zs-share", pct1(seed_stats(m, "transfer", "zeroshot", "cascade", "routed_share")[0]), src)
    R.add("tr-ft-share", pct1(seed_stats(m, "transfer", "finetune10", "cascade", "routed_share")[0]), src)
    for k, x in (("cascade", "cascade"), ("distilaug", "distilbert_aug"), ("average", "average")):
        R.add(f"tr-ft-{k}", f4(seed_stats(m, "transfer", "finetune10", x)[0]), src)
    sg = pd.read_csv(FINAL / "significance.csv")
    sg = sg[sg.comparison == "best_baseline"]
    win = sg[(sg.diff_macro_f1_seed1 > 0) & (sg.significant_bootstrap_holm | sg.significant_mcnemar_holm)]
    if sorted(win.condition) != ["low_data_frac01", "transfer"] or \
            (win.significant_bootstrap_holm & win.significant_mcnemar_holm).any():
        raise ValueError("text: the cascade beats the best baseline twice, each time by one test only")
    R.add("tr-ft-p-boot", pvalue(sig_row("transfer", "finetune10", "best_baseline").p_bootstrap_holm), sig)
    # cost
    for test, k in (("clean", "clean"), ("heldout20", "ho20")):
        allm = m[(m.condition == "mendeley_3class") & (m.test_set == test)]
        for i, r_ in enumerate((1, 5, 10)):
            g = allm.groupby("model")[f"cost_{r_}to1"].mean().dropna()
            if k == "ho20" and g.idxmin() != "cascade":
                raise ValueError(f"cascade is not the cheapest at HO20, r={r_}")
            if k == "clean":
                rest = g.drop([x for x in ("average", "cascade_noaug") if x in g.index])
                if g.idxmin() != "average" or rest.idxmin() != ("cascade" if r_ == 1 else "distilbert_aug"):
                    raise ValueError(f"unexpected cheapest models on clean, r={r_}: {g.idxmin()}, {rest.idxmin()}")
    R.add("cost-clean-r1-cascade", f4(m3(m, "clean", "cascade", "cost_1to1")), src)
    R.add("cost-clean-r3-cascade", f4(m3(m, "clean", "cascade", "cost_10to1")), src)
    R.add("cost-clean-r3-distilaug", f4(m3(m, "clean", "distilbert_aug", "cost_10to1")), src)
    R.add("cost-clean-r3-average", f4(m3(m, "clean", "average", "cost_10to1")), src)
    for x, k in (("cascade", "cascade"), ("svm_char_aug", "svmcharaug"), ("distilbert_aug", "distilaug")):
        R.add(f"cost-ho20-r2-{k}", f4(m3(m, "heldout20", x, "cost_5to1")), src)
    for x, k in (("cascade", "cascade"), ("svm_char_aug", "svmcharaug")):
        R.add(f"cost-mh-ho20-r2-{k}", f4(seed_stats(m, "setting_mahmud", "heldout20", x, "cost_5to1")[0]), src)
    # speed
    e2e = json.loads((FINAL / "cascade_end_to_end.json").read_text())
    lat = pd.read_csv(FINAL / "latency.csv").set_index("model")
    R.add("lat-cascade-mean", f2(e2e["mean_ms"]), "results/final/cascade_end_to_end.json")
    R.add("lat-svmcharaug-thr", f"{lat.loc['svm_char_aug', 'throughput_msg_per_s']:.0f}", "results/final/latency.csv")
    R.add("lat-bert-p50", f"{lat.loc['bert', 'p50_ms']:.1f}", "results/final/latency.csv")
    R.add("lat-qwen-p50", f"{lat.loc['qwen_fewshot', 'p50_ms']:.0f}", "results/final/latency.csv (GPU)")
    R.add("size-cascade", f"{e2e['size_mb']:.0f}", "results/final/cascade_end_to_end.json")
    R.add("v2-slowdown", f"{lat.loc['smishfuse_v2', 'p50_ms'] / e2e['p50_ms']:.0f}", "results/final/latency.csv / e2e p50")
    # analysis
    cv = pd.read_csv(FINAL / "cascade_curve.csv")
    g = cv[cv.test_set == "clean"].groupby("share").macro_f1.mean()
    R.add("curve-clean-0", f4(g.loc[0.0]), "results/final/cascade_curve.csv (mean over seeds)")
    R.add("curve-clean-peak", f4(g.max()), "results/final/cascade_curve.csv")
    R.add("curve-clean-peak-share", f"{100 * g.idxmax():.0f}", "results/final/cascade_curve.csv")
    R.add("curve-clean-100", f4(g.loc[1.0]), "results/final/cascade_curve.csv")
    tests = ["clean", "train05", "train10", "train20", "heldout05", "heldout10", "heldout20"]
    wins = [t for t in tests if m3(m, t, "cascade_noaug") > m3(m, t, "cascade")]
    R.add("noaug-wins", len(wins), src + " cascade_noaug > cascade")
    R.add("n-rob-sets", len(tests), "configs/final.yaml mendeley_3class test_sets")
    lost = [t for t in tests if t not in wins]
    if lost != ["heldout10"]:
        raise ValueError(f"text names the 10% held-out set as the only loss: {lost}")
    beyond = [t for t in wins if abs(m3(m, t, "cascade_noaug") - m3(m, t, "cascade")) >
              max(seed_stats(m, "mendeley_3class", t, "cascade_noaug")[1], seed_stats(m, "mendeley_3class", t, "cascade")[1])]
    R.add("noaug-beyond", len(beyond), src + " wins beyond the seed spread")
    for t in tests[1:]:
        if not m3(m, t, "average") < m3(m, t, "cascade"):
            raise ValueError(f"average not below the cascade at {t}")
    for t in tests:
        for x in ("smishfuse_v1", "smishfuse_v2", "smishfuse_v2_noaug"):
            if m3(m, t, x) >= m3(m, t, "svm_char"):
                raise ValueError(f"{x} not below char SVM at {t}")
    ov = json.loads((STATS / "overlap.json").read_text())
    clean = next(r for r in ov["mendeley_uci"] if r["data"] == "clean")
    ct = {(c["mendeley_label"], c["uci_label"]): c["n"] for c in clean["label_crosstab_mendeley_vs_best_uci_match"]}
    R.add("men-smish-uci-spam", ct[("smishing", "spam")], "reports/stats/overlap.json crosstab (clean)")
    R.add("men-uci-smish", json.loads((STATS / "dedup_mendeley.json").read_text())["labels_after"]["smishing"],
          "reports/stats/dedup_mendeley.json labels_after")
    sys.path.insert(0, str(PAPER / "tools"))
    from make_tables import inflation_rows
    rows = {r["model"]: r for r in inflation_rows()}
    mh = rows["mahmud2024_cnn_bigru"]
    R.add("infl-mh-acc", f2(mh["accuracy_diff"]), "results/step5a/inflation (dirty - clean, pp)")
    R.add("infl-mh-f1", f2(mh["macro_f1_diff"]), "results/step5a/inflation (dirty - clean, pp)")
    R.add("infl-mh-acc-dirty", f4(mh["accuracy_dirty"]), "results/step5a/inflation")
    drops = [-r["macro_f1_diff"] for r in rows.values() if r["task"] != "mendeley_binary"]
    if min(drops) <= 0:
        raise ValueError("a 3-class or smishing-vs-spam model is not lower on dirty data")
    R.add("infl-max-drop", f2(max(drops)), "results/step5a/inflation (largest macro F1 drop)")


def kz_numbers(R: Registry) -> None:
    """Kazakhstan case study (results/step6, step6b, step6c; data/kz_*), as in the step-6 reports."""
    import re
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    src6 = "results/step6/predictions.csv"
    R.add("kz-orig", int((cases.variant == "original").sum()), "data/kz_cases/cases.csv")
    R.add("kz-all", len(cases), "data/kz_cases/cases.csv")
    p6 = pd.read_csv(ROOT / "results" / "step6" / "predictions.csv")
    o = p6[p6.variant == "original"]
    fr = o[o.pair == "fraud"]
    def missed(inp, lang, model):
        x = fr[(fr.input == inp) & (fr.language == lang) & (fr.model == model)]
        return int((x.pred == "ham").sum()), len(x)
    for lang in ("ru", "kk"):
        a, n = missed("original", lang, "lite")
        if (a, n) != missed("original", lang, "full"):
            raise ValueError("text says Lite and the cascade miss the same number")
        R.add(f"kz-{lang}-missed-lite", a, src6)
        R.add(f"kz-{lang}-missed-bert", missed("original", lang, "bert")[0], src6)
        if lang == "ru":
            R.add("kz-fraud-lang", n, src6)
    tr_kk = [missed("translated_en", "kk", x)[0] for x in ("lite", "full", "bert")]
    if max(tr_kk) != 0:
        raise ValueError("text says no Kazakh fraud message is missed after translation")
    tr_ru = [missed("translated_en", "ru", x)[0] for x in ("lite", "full", "bert")]
    R.add("kz-ru-tr-missed-min", min(tr_ru), src6)
    R.add("kz-ru-tr-missed-max", max(tr_ru), src6)
    en = o[(o.language == "en") & (o.input == "original") & (o.model == "lite")]
    R.add("kz-en-correct-lite", int((en.pred == en.label).sum()), src6)
    R.add("kz-en-n", len(en), src6)
    link = re.compile(r"\S+\.[^\s.]*x[^\s.]*mp[^\s.]*|\S+\.example")   # scripts/make_step6b_report.py LINK
    tw = pd.read_csv(ROOT / "results" / "step6" / "translations.csv", keep_default_na=False).set_index("id")
    ts = pd.read_csv(ROOT / "results" / "step6b" / "mt_sentence" / "translations.csv", keep_default_na=False).set_index("id")
    text = cases.set_index("id").text
    with_link = [i for i in ts.index if link.search(text[i])]
    R.add("kz-link-n", len(with_link), "results/step6b/mt_sentence/translations.csv (RU and KK with a link)")
    R.add("kz-link-lost-word", sum(not link.search(tw.loc[i, "translation"]) for i in with_link), "results/step6/translations.csv")
    if any(not link.search(ts.loc[i, "translation"]) for i in with_link):
        raise ValueError("text says sentence-wise translation keeps every link")
    ps = pd.read_csv(ROOT / "results" / "step6b" / "mt_sentence" / "predictions.csv")
    for lang in ("ru", "kk"):
        w = o[(o.input == "translated_en") & (o.language == lang) & (o.model == "lite")]
        x = ps[(ps.variant == "original") & (ps.language == lang) & (ps.model == "lite")]
        R.add(f"kz-{lang}-sent-before", int((w.pred == w.label).sum()), src6)
        R.add(f"kz-{lang}-sent-after", int((x.pred == x.label).sum()), "results/step6b/mt_sentence/predictions.csv")
        if lang == "ru":
            R.add("kz-ru-orig-n", len(x), "results/step6b/mt_sentence/predictions.csv")
    l6 = json.loads((ROOT / "results" / "step6" / "latency.json").read_text())
    l6b = json.loads((ROOT / "results" / "step6b" / "mt_sentence" / "latency.json").read_text())
    R.add("kz-mt-p50", f"{l6['translation_nllb']['p50_ms'] / 1000:.1f}", "results/step6/latency.json")
    R.add("kz-mts-p50", f"{l6b['translation_nllb_sentencewise']['p50_ms'] / 1000:.1f}", "results/step6b/mt_sentence/latency.json")
    # adaptation
    adapt = pd.read_csv(ROOT / "data" / "kz_adapt" / "adapt.csv", dtype=str, keep_default_na=False)
    R.add("kz-adapt-n", len(adapt), "data/kz_adapt/adapt.csv")
    R.add("kz-adapt-ideas", adapt.id.str.rsplit("_", n=1).str[0].nunique(), "data/kz_adapt/adapt.csv (ids without language)")
    r6b = ROOT / "results" / "step6b" / "adapt"
    runs = pd.read_csv(r6b / "runs.csv")
    pa = pd.read_csv(r6b / "predictions.csv", keep_default_na=False)
    pa["ok"] = pa.pred == pa.label
    pc = pa[pa.set == "cases"].merge(cases[["id", "pair", "variant"]], on="id")
    orig = pc[pc.variant == "original"]
    full = {c: runs[(runs.condition == c) & (runs.subsample == 0)].run_id.iloc[0] for c in ("A", "B")}
    srcA = "results/step6b/adapt/predictions.csv"
    for key, rid in (("before", "before"), ("after-a", full["A"]), ("after-b", full["B"])):
        x = orig[orig.run_id == rid]
        R.add(f"kz-fraud-{key}", int(x[x.pair == "fraud"].ok.sum()), srcA)
        R.add(f"kz-acc-{key}", f2(x.ok.mean()), srcA)
    R.add("kz-fraud-n", int((orig[orig.run_id == "before"].pair == "fraud").sum()), srcA)
    gm = runs[runs.condition.isin(["A", "B"])].groupby(["condition", "size"]).val_macro_f1.mean()
    R.add("kz-val-min", f"{gm.min():.3f}", "results/step6b/adapt/runs.csv (mean per condition and size)")
    R.add("kz-val-max", f"{gm.max():.3f}", "results/step6b/adapt/runs.csv (mean per condition and size)")
    R.add("kz-val-before", f"{runs[runs.run_id == 'before'].val_macro_f1.iloc[0]:.3f}", "results/step6b/adapt/runs.csv")
    ho = pa[pa.set == "adapt_held_out"].merge(adapt[["id", "text"]], on="id")
    ho = ho[(ho.label != "smishing") & ho.text.str.contains(".example", regex=False)]
    accs = [ho[ho.run_id == full["A"]].ok.mean()]
    for cond in ("A", "B"):
        for _, g in runs[(runs.condition == cond) & (runs.subsample > 0)].groupby("size"):
            accs.append(ho[ho.run_id.isin(g.run_id)].groupby("run_id").ok.mean().mean())
    R.add("kz-legit-link-min", f2(min(accs)), srcA + " (legitimate held-out messages with a link)")
    R.add("kz-legit-link-max", f2(max(accs)), srcA + " (legitimate held-out messages with a link)")
    b = ho[ho.run_id == "before"]
    R.add("kz-legit-link-before", f2(b.ok.mean()), srcA)
    R.add("kz-legit-link-smish-before", pct0((b.pred == "smishing").mean()), srcA)
    adapted = ho[ho.run_id != "before"]
    R.add("kz-legit-link-smish-after", pct0((adapted.pred == "smishing").mean()), srcA + " (all adapted runs pooled)")
    k7 = pa[(pa.set == "cases") & pa.id.str.startswith("kz7_legit") & pa.run_id.isin(full.values())]
    if not (k7.pred == "smishing").all():
        raise ValueError("text says the adapted model flags the bank warning in every language")
    # real messages
    real = pd.read_csv(ROOT / "data" / "kz_real" / "public_smishing.csv", dtype=str, keep_default_na=False)
    pr = pd.read_csv(ROOT / "results" / "step6c" / "predictions.csv")
    R.add("kz-real-n", len(real), "data/kz_real/public_smishing.csv")
    aw = pr[pr.input == "as_written"]
    ad = aw[aw.model.isin(["lite_adapted_A", "lite_adapted_B"]) & (aw.pred == "smishing")]
    if ad.groupby("id").size().min() != 2:
        raise ValueError("A and B do not flag the same messages")
    R.add("kz-real-flagged", ad.id.nunique(), "results/step6c/predictions.csv (as written, adapted A and B)")
    R.add("kz-real-conf-min", f2(ad.confidence.min()), "results/step6c/predictions.csv")
    R.add("kz-real-conf-max", f2(ad.confidence.max()), "results/step6c/predictions.csv")
    if (aw[~aw.model.isin(["lite_adapted_A", "lite_adapted_B"])].pred == "smishing").any():
        raise ValueError("text says no model without adaptation flags a real message as written")
    never = set(pr.id) - set(pr[pr.pred == "smishing"].id)
    R.add("kz-real-never", len(never), "results/step6c/predictions.csv (no model, no input)")


SHIM = ("shim2024_bert_base", "shim2024_distilbert_base", "shim2024_roberta_base")


def lite_numbers(R: Registry) -> None:
    """Lite (final svm_char) against the re-implemented published transformers (MDPI framing)."""
    m = metrics()
    src = "results/final/metrics.csv (mean over 5 seeds)"
    clean = [seed_stats(m, "setting_shim", "clean", x)[0] for x in SHIM]
    R.add("sh-clean-min", f4(min(clean)), src)
    R.add("sh-clean-max", f4(max(clean)), src)
    R.add("svmchar-shim-clean", f4(seed_stats(m, "setting_shim", "clean", "svm_char")[0]), src)
    R.add("svmchar-mahmud-clean", f4(seed_stats(m, "setting_mahmud", "clean", "svm_char")[0]), src)
    R.add("mh-clean", f4(seed_stats(m, "setting_mahmud", "clean", "mahmud2024_cnn_bigru")[0]), src)
    # post-hoc tests of Lite against each published method (seed 1, Holm within the condition)
    ph = pd.read_csv(FINAL / "posthoc_significance.csv")
    ph = ph[ph.ours == "svm_char"]
    both = ph.significant_bootstrap_holm & ph.significant_mcnemar_holm
    for cond, tests in (("setting_shim", ("heldout10", "heldout20")),
                        ("setting_mahmud", ("heldout05", "heldout10", "heldout20"))):
        x = ph[(ph.condition == cond) & ph.test_set.isin(tests)]
        if len(x) != len(tests) * (3 if cond == "setting_shim" else 1) or not both[x.index].all() \
                or (x.diff_macro_f1_seed1 <= 0).any():
            raise ValueError(f"text: Lite is significantly better by both tests in {cond} at {tests}")
    x = ph[(ph.condition == "setting_shim") & (ph.test_set == "clean")]
    if sorted(x[both[x.index]].other) != ["shim2024_roberta_base"]:
        raise ValueError("text: on clean Shim data Lite is significantly better (both tests) only than RoBERTa")
    x = ph[(ph.condition == "setting_shim") & (ph.test_set == "heldout05")]
    if sorted(x[both[x.index]].other) != ["shim2024_roberta_base"]:
        raise ValueError("text: at HO 5% in the Shim setting Lite is significantly better only than RoBERTa")
    # latency: Lite against the published transformers (one CPU thread, batch 1)
    lat = pd.read_csv(FINAL / "latency.csv").set_index("model")
    lite = lat.loc["svm_char", "p50_ms"]
    sh = [lat.loc[x, "p50_ms"] for x in SHIM]
    R.add("lat-svmchar-p50", f2(lite), "results/final/latency.csv")
    R.add("lat-svmchar-thr", f"{lat.loc['svm_char', 'throughput_msg_per_s']:.0f}", "results/final/latency.csv")
    R.add("lat-sh-p50-min", f"{min(sh):.1f}", "results/final/latency.csv (Shim et al., setting_shim)")
    R.add("lat-sh-p50-max", f"{max(sh):.1f}", "results/final/latency.csv (Shim et al., setting_shim)")
    R.add("speed-ratio-min", f"{min(sh) / lite:.0f}", "results/final/latency.csv p50 Shim / p50 svm_char")
    R.add("speed-ratio-max", f"{max(sh) / lite:.0f}", "results/final/latency.csv p50 Shim / p50 svm_char")
    R.add("size-svmchar", f"{lat.loc['svm_char', 'size_mb']:.1f}", "results/final/latency.csv size_mb")


def build() -> dict[str, tuple[str, str]]:
    R = Registry()
    for fn in (data_numbers, main_numbers, published_numbers, latency_numbers, config_numbers, results_numbers,
               kz_numbers, lite_numbers):
        fn(R)
    return R.v


def render(v: dict[str, tuple[str, str]]) -> str:
    out = ["% Generated by paper_mdpi/tools/paper_numbers.py from results/, reports/ and configs/. Do not edit by hand.",
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
    print(f"wrote paper_mdpi/numbers.tex ({len(v)} values)")
