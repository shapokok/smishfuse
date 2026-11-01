"""Builds reports/step6c_kz_real.md from results/step6c/ and data/kz_real/public_smishing.csv (no model is run here).
Run: python scripts/make_step6c_report.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "step6c"
MODELS = {"lite": "Lite до", "lite_adapted_A": "Lite адапт. A", "lite_adapted_B": "Lite адапт. B", "full": "Full (каскад)",
          "bert": "BERT"}
INPUTS = {"as_written": "как написано", "mt_sentence": "перевод по предложениям"}
STATUS = {"full": "полный текст", "screenshot transcription, ending cut off": "расшифровка скриншота, окончание обрезано",
          "fragment": "фрагмент"}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    real = pd.read_csv(ROOT / "data" / "kz_real" / "public_smishing.csv", dtype=str, keep_default_na=False)
    p = pd.read_csv(R / "predictions.csv", keep_default_na=False)
    tr = pd.read_csv(R / "translations.csv", keep_default_na=False).set_index("id")
    check = json.loads((R / "check.json").read_text())
    runs = pd.read_csv(ROOT / "results" / "step6b" / "adapt" / "runs.csv")
    ref = {c: runs[(runs.condition == c) & (runs.subsample == 0)].val_macro_f1.iloc[0] for c in "AB"}

    md = ["# SmishFuse, шаг 6в: реальные публичные примеры из Казахстана\n",
          "Сгенерировано `scripts/make_step6c_report.py` из `results/step6c/` (ноутбук `smishfuse-step6c-real`, "
          "Kaggle CPU). **Пост-хок (решение 22 в README): после результатов шагов 6 и 6б.** Без обучения новых "
          "моделей; **4 сообщения — качественная иллюстрация, метрика (recall) не считается.** Набор зафиксирован "
          "коммитом 1596963 до запуска моделей.\n"]
    md.append("## 1. Сообщения\n")
    md.append(table(["id", "Язык", "Статус текста", "Текст", "Источник"],
                    [[r.id, r.language.upper(), STATUS.get(r.verbatim, r.verbatim), r.text,
                      f"[{r.source_url.split('/')[2]}]({r.source_url}), {r.published}"] for r in real.itertuples()]))
    md.append("\nСсылки в статьях не приводятся, поэтому в текстах их нет. В real03 цифры номера дела заменены "
              "случайными; отправитель (номер телефона) не сохранён.\n")

    md.append("## 2. Вердикты и уверенность\n")
    md.append("Предсказание (максимальная вероятность). Full: † — ответ дала вторая ступень (DistilBERT). Все "
              "сообщения — мошеннические; ✓ — помечено smishing.\n")
    rows = []
    for r in real.itertuples():
        for inp, iname in INPUTS.items():
            cells = []
            for m in MODELS:
                x = p[(p.id == r.id) & (p.input == inp) & (p.model == m)].iloc[0]
                dag = " †" if m == "full" and str(x.routed_to_distilbert) == "True" else ""
                cells.append(f"{x.pred} {x.confidence:.2f}{dag}{' ✓' if x.pred == 'smishing' else ''}")
            rows.append([r.id, iname] + cells)
    md.append(table(["id", "Вход"] + list(MODELS.values()), rows))
    md.append("\n**Переводы (NLLB по предложениям)**\n")
    md.append(table(["id", "Перевод"], [[i, t] for i, t in tr.translation.items()]))
    md.append(f"\nПроверка воспроизведения: Lite адаптированный обучен заново тем же кодом и seed, что в шаге 6б (веса "
              f"там не сохранялись). Macro F1 на Mendeley VAL: A {check['lite_adapted_A_val_macro_f1']:.4f} (в 6б "
              f"{ref['A']:.4f}), B {check['lite_adapted_B_val_macro_f1']:.4f} (в 6б {ref['B']:.4f}).\n")

    md.append("## 3. Ограничения\n")
    md += [f"- {s}" for s in LIMITATIONS] + [""] + SUMMARY
    (ROOT / "reports" / "step6c_kz_real.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote reports/step6c_kz_real.md")


LIMITATIONS = [
    "Реальных казахстанских SMS с дословным текстом в открытом доступе почти нет: СМИ и ведомства обычно "
    "пересказывают сообщения или показывают скриншоты. Из около 50 проверенных публикаций найдено 4 текста, из них "
    "один — расшифровка обрезанного скриншота и один — фрагмент.",
    "Все 4 сообщения мошеннические; оценка ложных тревог на реальных данных не проводилась.",
    "4 сообщения — иллюстрация, а не оценка: ни recall, ни сравнение моделей по ним делать нельзя.",
]
# Written after the run, checked against results/step6c/ (2026-11-01).
SUMMARY = [
    "## Итог (качественно)\n",
    "- **Как написано:** адаптированный Lite (A и B) помечает smishing 3 из 4 сообщений (real01–real03), но с "
    "низкой уверенностью (0.43–0.66). Lite до адаптации, каскад и BERT не помечают ни одного.",
    "- **После перевода по предложениям:** real01 (посылка со ссылкой) все пять моделей помечают smishing. "
    "real03 (штраф ПДД, казахский) — каскад (ответ DistilBERT) и BERT; все варианты Lite — spam.",
    "- **Не распознаётся ни одной моделью:** real04 — фрагмент «Статус вашего займа обновлен». real02 (второе "
    "сообщение рассылки без ссылки и призыва к действию) помечают smishing только адаптированные Lite A и B как "
    "написано, с низкой уверенностью (0.43 и 0.53); после перевода — ни одна модель. Сами по себе в этих текстах "
    "нет признаков обмана.",
    "- **Ошибка перевода:** «ҚР Жол полициясы» NLLB перевёл как «Road Police of the PRC».",
    "",
]


if __name__ == "__main__":
    main()
