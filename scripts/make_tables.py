"""Step 2.3: per-task result tables, mean ± std over seeds (std with ddof = 1).

Best value per column in bold, second best underlined (ranking on the rounded
mean, so equal displayed values share a rank; lower is better for ECE).
Outputs: results/tables/<task>.md, results/tables/<task>.tex, results/tables/all.md
Run: python scripts/make_tables.py
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.registry import MODEL_NAMES, ordered  # noqa: E402
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
LOWER_IS_BETTER = {"ece"}
DECIMALS = 4
COLUMN_NAMES = {"macro_f1": "Macro F1", "accuracy": "Accuracy", "weighted_f1": "Weighted F1",
                "balanced_accuracy": "Balanced acc.", "pr_auc": "PR-AUC", "recall_at_fpr_1pct": "Recall@FPR=1%",
                "ece": "ECE"}


def task_columns(task: dict) -> list[str]:
    base = ["macro_f1", "accuracy", "weighted_f1", "balanced_accuracy", "pr_auc"]
    if task["positive"] is not None:
        return base + ["recall_at_fpr_1pct", "ece", f"recall_{task['positive']}"]
    return base + ["ece"] + [f"recall_{c}" for c in task["classes"]]


def col_name(c: str) -> str:
    if c in COLUMN_NAMES:
        return COLUMN_NAMES[c]
    return f"Recall {c[len('recall_'):]}" if c.startswith("recall_") else c


def summarize(metrics: pd.DataFrame, task_name: str, cols: list[str]) -> pd.DataFrame:
    df = metrics[metrics["task"] == task_name]
    g = df.groupby("model")[cols]
    out = pd.concat({"mean": g.mean(), "std": g.std(ddof=1), "n": g.count()}, axis=1)
    return out.reindex(ordered(out.index))


def ranks(means: pd.Series, lower_better: bool) -> dict:
    """model -> 1 (best), 2 (second) or 0, by rounded mean with ties sharing a rank."""
    r = means.round(DECIMALS)
    distinct = sorted(r.unique(), reverse=not lower_better)
    best = distinct[0]
    second = distinct[1] if len(distinct) > 1 else None
    return {m: 1 if v == best else (2 if v == second else 0) for m, v in r.items()}


def render(summary: pd.DataFrame, cols: list[str], fmt: str) -> list[list[str]]:
    rows = {m: [] for m in summary.index}
    for c in cols:
        rk = ranks(summary[("mean", c)], c in LOWER_IS_BETTER)
        for m in summary.index:
            mean, std = summary.loc[m, ("mean", c)], summary.loc[m, ("std", c)]
            if fmt == "md":
                cell = f"{mean:.{DECIMALS}f} ± {std:.{DECIMALS}f}"
                cell = f"**{cell}**" if rk[m] == 1 else (f"<u>{cell}</u>" if rk[m] == 2 else cell)
            else:
                cell = f"{mean:.{DECIMALS}f} $\\pm$ {std:.{DECIMALS}f}"
                cell = f"\\textbf{{{cell}}}" if rk[m] == 1 else (f"\\underline{{{cell}}}" if rk[m] == 2 else cell)
            rows[m].append(cell)
    return [[MODEL_NAMES[m]] + rows[m] for m in summary.index]


def markdown(summary, cols) -> str:
    header = ["Модель"] + [col_name(c) for c in cols]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in render(summary, cols, "md")]
    return "\n".join(lines) + "\n"


def latex(summary, cols, task_name: str, n_seeds: int) -> str:
    esc = lambda s: s.replace("%", "\\%").replace("_", "\\_")  # noqa: E731
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{{esc(task_name)}: mean $\\pm$ std over {n_seeds} seeds (test). "
        "Best in bold, second underlined.}",
        f"\\label{{tab:{task_name}}}",
        "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
        "Model & " + " & ".join(esc(col_name(c)) for c in cols) + " \\\\", "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in render(summary, cols, "tex")]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    TABLES.mkdir(parents=True, exist_ok=True)
    combined = []
    for task_name, task in cfg["tasks"].items():
        cols = task_columns(task)
        s = summarize(metrics, task_name, cols)
        n_seeds = int(s[("n", "macro_f1")].min())
        md = markdown(s, cols)
        (TABLES / f"{task_name}.md").write_text(md, encoding="utf-8")
        (TABLES / f"{task_name}.tex").write_text(latex(s, cols, task_name, n_seeds), encoding="utf-8")
        combined.append(f"### {task_name}\n\n{md}")
        print(f"[tables] {task_name}: {len(s)} models, {n_seeds} seeds")
    (TABLES / "all.md").write_text("\n".join(combined), encoding="utf-8")


if __name__ == "__main__":
    main()
