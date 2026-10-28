"""Builds reports/step1_data_report.md from reports/stats/*.json.

Every number in the report comes from the stats files written by the other
scripts; the only hand-written content is the source metadata (SOURCES_META)
and the notes that cannot be derived from the data.
Run: python src/data/report.py
"""
from common import REPORTS, SOURCES, STATS, read_json

DATASETS = ["mendeley", "moz_smishing", "super_sms", "uci_sms"]
SPLIT_DATASETS = ["mendeley", "moz_smishing", "super_sms", "super_sms_clean"]
OVERLAP_DATASETS = ["mendeley", "moz_smishing", "super_sms", "super_sms_clean", "uci_sms"]

SOURCES_META = {
    "mendeley": {
        "title": "Mendeley SMS Phishing Dataset",
        "source": "https://data.mendeley.com/datasets/f45bkkt8pr/1 (DOI 10.17632/f45bkkt8pr.1), файл Dataset_5971.zip",
        "license": "CC BY 4.0 (поле data_licence в API Mendeley Data)",
        "expected_note": "5971: ham 4844, spam 489, smishing 638 (описание на Mendeley)",
    },
    "moz_smishing": {
        "title": "MOZ-Smishing",
        "source": "https://huggingface.co/datasets/MOZNLP/MOZ-Smishing (ссылка из PDF статьи "
        "aclanthology.org/2025.africanlp-1.23), commit 1092f9d9, файл test.csv",
        "license": "creativeml-openrail-m (тег на HF); в статье: «open license» без названия",
        "expected_note": "≈2561 (задание); 2009 legitimate / 552 smishing (карточка HF)",
    },
    "super_sms": {
        "title": "Super SMS Dataset",
        "source": "https://github.com/smspamresearch/spstudy (ссылка из текста статьи Salman, Ikram, Kaafar, "
        "IEEE Access 12, 2024, DOI 10.1109/ACCESS.2024.3364671), commit a0889f36, Data/super_sms_dataset.csv",
        "license": "не указана: в репозитории нет файла LICENSE, API GitHub возвращает license=null "
        "(CC BY-NC-ND 4.0 относится к статье, не к данным)",
        "expected_note": "67018 (текст статьи; в аннотации «more than 68K»); 40837 ham / 26181 spam "
        "(SpaLLM-Guard, arXiv:2501.04985, те же авторы)",
    },
    "uci_sms": {
        "title": "UCI SMS Spam Collection v.1 (только для проверки пересечений)",
        "source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection, файл sms+spam+collection.zip",
        "license": "CC BY 4.0 (страница UCI); readme в архиве: авторские права у Almeida и Gómez Hidalgo, "
        "использование бесплатно с просьбой ссылаться",
        "expected_note": "5574: ham 4827, spam 747 (readme в архиве)",
    },
}

# Literature figure for Mendeley ∩ UCI: Pritom et al., arXiv:2604.11429
# ("4,753 common data instances ... with 4,199 ham and 554 spam messages");
# the review does not describe how the overlap was counted.
LITERATURE_MENDELEY_UCI = 4753
LITERATURE_SOURCE = "обзор Pritom et al., arXiv:2604.11429: 4199 ham и 554 spam, метод подсчёта не описан"


def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "—"


def labels_str(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items())) or "—"


def lang_summary(langs: dict, top: int = 3) -> str:
    total = sum(langs.values())
    items = sorted(langs.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = [f"{k} {pct(v, total)}" for k, v in items[:top]]
    rest = total - sum(v for _, v in items[:top])
    if rest:
        shown.append(f"прочие {pct(rest, total)}")
    return ", ".join(shown)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    load = {d: read_json(STATS / f"load_{d}.json") for d in DATASETS}
    dedup = {d: read_json(STATS / f"dedup_{d}.json") for d in DATASETS}
    split = {d: read_json(STATS / f"split_{d}.json") for d in SPLIT_DATASETS}
    overlap = read_json(STATS / "overlap.json")
    ssc = read_json(STATS / "super_sms_clean.json")

    md = []
    md.append("# SmishFuse, шаг 1: отчёт по данным\n")
    md.append(
        "Сгенерировано скриптом `src/data/report.py` из `reports/stats/*.json`. Воспроизведение описано в "
        "`README.md`: `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt`, "
        "затем `.venv/bin/python src/data/run_all.py`. Сырые файлы скачиваются автоматически и сверяются по sha256. "
        "Модели не обучались.\n"
    )
    md.append(
        "Метод дубликатов везде один: текст → lowercase + схлопывание пробельных символов; точный дубликат = "
        "совпадение нормализованного текста; почти-дубликат = точный коэффициент Жаккара ≥ 0.9 на множествах "
        "символьных 5-грамм. MinHash LSH (datasketch, 128 перестановок, порог кандидатов 0.7) используется только "
        "для поиска кандидатов, каждая пара проверяется точным Жаккаром. Группы дубликатов = компоненты связности.\n"
    )

    # 1. Datasets
    md.append("## 1. Датасеты\n")
    rows = []
    for d in DATASETS:
        s, m = load[d], SOURCES_META[d]
        rows.append([m["title"], m["source"], m["license"], m["expected_note"],
                     f"{s['raw_rows']} строк в файле, {s['valid_rows']} валидных", labels_str(s["labels"]),
                     lang_summary(s["languages"]), s["mean_len_chars"]])
    md.append(table(["Датасет", "Источник", "Лицензия", "Ожидалось", "Фактически", "Классы (после загрузки)",
                     "Языки (langdetect)", "Ср. длина, симв."], rows))
    md.append("\nSmishing-4C исключён из проекта: доступ только по запросу.\n")
    md.append("sha256 сырых файлов (проверяются при каждом запуске):\n")
    for d, src in SOURCES.items():
        md.append(f"- `{src['path'].relative_to(REPORTS.parent)}`: `{src['sha256']}`")
    md.append("")

    # 2. Checks
    md.append("## 2. Проверка размеров и классов\n")
    rows = []
    for d in DATASETS:
        for c in load[d]["checks"]:
            rows.append([d, c["what"], c["expected"], c["actual"], "совпадает" if c["match"] else "**расхождение**"])
    md.append(table(["Датасет", "Что", "Ожидалось", "Фактически", "Итог"], rows))
    md.append("\nЗамечания по загрузке:\n")
    for d in DATASETS:
        s = load[d]
        bits = []
        if s.get("invalid_rows"):
            bits.append(f"отброшено при загрузке: {s['invalid_rows']} (см. `reports/removed/{d}_load.csv`)")
        if "raw_label_values" in s:
            bits.append(f"исходные значения меток: {s['raw_label_values']}")
        for k in ["rows_with_U+FFFD", "rows_with_leading_or_trailing_whitespace", "non_utf8_bytes_decoded_as_cp1252",
                  "rows_with_C1_control_chars", "rows_lowercase_alnum_only", "raw_source_values"]:
            if k in s:
                bits.append(f"{k}: {s[k]}")
        if bits:
            md.append(f"- **{d}**: " + "; ".join(str(b) for b in bits))
    md.append("")

    # 3. Languages
    md.append("## 3. Языки (langdetect 1.0.9, seed 0, до дедупликации)\n")
    rows = []
    for d in DATASETS:
        langs = load[d]["languages"]
        total = sum(langs.values())
        top = sorted(langs.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        rows.append([d, total, ", ".join(f"{k}: {v} ({pct(v, total)})" for k, v in top),
                     langs.get("unknown", 0)])
    md.append(table(["Датасет", "Сообщений", "Топ-8 языков", "unknown"], rows))
    md.append(
        "\nlangdetect ненадёжен на коротких SMS: метки вроде af/so/cy/no в английских датасетах почти наверняка "
        "ошибки детектора, а не реальные языки. Вручную это не проверялось. Поле `language` во всех CSV хранит "
        "результат детектора, а не заявленный язык датасета.\n"
    )

    # 4. Dedup
    md.append("## 4. Дубликаты внутри датасетов и конфликты меток\n")
    md.append(
        "Правило. В группе без конфликта остаётся сообщение с наименьшим id, остальные удаляются. В группе с "
        "разными метками проводится голосование большинством: остаётся сообщение с наименьшим id среди носителей "
        "метки большинства, остальные члены большинства удаляются как дубликаты, меньшинство — как "
        "`label_conflict_minority`. Если голоса разделились поровну, группа удаляется целиком "
        "(`label_conflict_tie`). Метки не переписываются: оставленное сообщение уже имеет метку большинства. "
        "Все удалённые строки лежат в `reports/removed/<name>_dedup.csv` (reason, group_id, kept_id, "
        "jaccard_to_kept), конфликтные группы целиком — в `reports/conflicts/<name>.csv` (колонка resolution).\n"
    )
    rows = []
    for d in DATASETS:
        s = dedup[d]
        r = s["removed"]
        cg, res = s["conflict_groups"], s["conflict_resolution"]
        rows.append([d, s["rows_before"], r.get("exact_duplicate", 0), r.get("near_duplicate", 0),
                     f"{sum(cg.values())} ({cg.get('exact', 0)} точн. / {cg.get('near', 0)} почти)",
                     res.get("majority", 0), res.get("tie", 0), r.get("label_conflict_minority", 0),
                     r.get("label_conflict_tie", 0), s["rows_after"], s["largest_group"]])
    md.append(table(["Датасет", "До", "Удалено точных дубл.", "Удалено почти-дубл.", "Групп с конфликтом",
                     "Решено большинством", "Ничьих", "Удалено (меньшинство)", "Удалено (ничья)", "После",
                     "Макс. размер группы"], rows))
    md.append("\n### Сравнение с прежним правилом «конфликтная группа удаляется целиком»\n")
    rows = []
    for d in DATASETS:
        s = dedup[d]
        rows.append([d, s["rows_after_drop_whole_group_rule"], s["rows_after"],
                     s["rows_saved_vs_drop_whole_group_rule"], labels_str(s["rows_saved_by_label"])])
    md.append(table(["Датасет", "После (прежнее правило)", "После (большинство)", "Сохранено строк",
                     "Сохранённые по меткам"], rows))
    md.append(
        "\nПочти все конфликтные группы состоят из двух сообщений с разными метками, то есть дают ничью и "
        "удаляются, как и раньше. Поэтому выигрыш от голосования минимальный.\n"
    )
    rows = []
    for d in DATASETS:
        s = dedup[d]
        rows.append([d, labels_str(s["labels_before"]), labels_str(s["labels_after"]),
                     labels_str(s["conflict_rows_by_label"])])
    md.append(table(["Датасет", "Классы до", "Классы после", "Строки в конфликтных группах по меткам"], rows))
    md.append("")

    # 5. Overlap
    md.append("## 5. Пересечения между датасетами (после дедупликации)\n")
    md.append("Ячейка [строка A, столбец B] = сколько сообщений A имеют точный или почти-дубликат в B "
              "(на диагонали размер A).\n")
    mat = overlap["matrix_exact_or_near"]
    md.append(table(["A \\ B"] + OVERLAP_DATASETS,
                    [[a] + [mat[a][b] for b in OVERLAP_DATASETS] for a in OVERLAP_DATASETS]))
    md.append("\nТолько точные совпадения после нормализации:\n")
    mat = overlap["matrix_exact"]
    md.append(table(["A \\ B"] + OVERLAP_DATASETS,
                    [[a] + [mat[a][b] for b in OVERLAP_DATASETS] for a in OVERLAP_DATASETS]))
    md.append("\n### Mendeley и UCI SMS Spam Collection\n")
    rows = []
    for s in overlap["mendeley_uci"]:
        rows.append([s["data"], s["mendeley_rows"], s["mendeley_rows_in_uci_exact"],
                     f"{s['mendeley_rows_in_uci_exact_or_near']} ({pct(s['mendeley_rows_in_uci_exact_or_near'], s['mendeley_rows'])})",
                     s["uci_rows"], s["uci_rows_in_mendeley_exact_or_near"],
                     "; ".join(f"{c['mendeley_label']}→{c['uci_label']}: {c['n']}"
                               for c in s["label_crosstab_mendeley_vs_best_uci_match"])])
    md.append(table(["Данные", "Mendeley", "из них в UCI (точно)", "в UCI (точно или почти)", "UCI",
                     "UCI в Mendeley (точно или почти)", "Метка Mendeley → метка лучшего совпадения в UCI"], rows))
    raw = overlap["mendeley_uci"][0]
    md.append(
        f"\nВ литературе общих сообщений {LITERATURE_MENDELEY_UCI} ({LITERATURE_SOURCE}). У нас до дедупликации "
        f"{raw['mendeley_rows_in_uci_exact']} сообщений Mendeley совпадают с UCI точно (после нормализации) и "
        f"{raw['mendeley_rows_in_uci_exact_or_near']} точно или почти. Поскольку метод в обзоре не описан, "
        "расхождение объяснить нельзя; в статье указываем обе цифры и наш метод (см. README, Decisions). "
        "Пары после дедупликации лежат в `reports/overlap_mendeley_uci_pairs.csv`.\n"
    )

    # 6. Super SMS clean
    md.append("## 6. super_sms_clean: Super SMS без пересечений с Mendeley и UCI\n")
    md.append(
        "Из Super SMS после дедупликации удалены все сообщения, которые точно или почти (Жаккар ≥ 0.9) совпадают "
        "хотя бы с одним сообщением Mendeley или UCI. Сравнение идёт с версиями до дедупликации (`unified.csv`), "
        "чтобы учесть и сообщения, удалённые там. Кандидаты LSH ищутся в обе стороны. Исходный "
        "`data/processed/super_sms/` не изменён. Лог удалённого с лучшим совпадением и его Жаккаром: "
        "`reports/removed/super_sms_clean_cross.csv`.\n"
    )
    rem = ssc["removed"]
    md.append(table(
        ["До", "Только в Mendeley", "Только в UCI", "И там, и там", "Всего удалено", "из них с J = 1",
         "Удалено по меткам", "После", "Классы после"],
        [[ssc["rows_before"], rem.get("match_mendeley", 0), rem.get("match_uci_sms", 0),
          rem.get("match_mendeley_and_uci_sms", 0), ssc["rows_before"] - ssc["rows_after"],
          ssc["removed_with_best_jaccard_1"], labels_str(ssc["removed_by_label"]), ssc["rows_after"],
          labels_str(ssc["labels_after"])]],
    ))
    om = overlap["matrix_exact_or_near"]
    md.append(
        f"\nКонтроль через матрицу раздела 5: super_sms_clean → mendeley = {om['super_sms_clean']['mendeley']}, "
        f"→ uci_sms = {om['super_sms_clean']['uci_sms']}; mendeley → super_sms_clean = "
        f"{om['mendeley']['super_sms_clean']}, uci_sms → super_sms_clean = {om['uci_sms']['super_sms_clean']}.\n"
    )

    # 7. Splits
    md.append("## 7. Разбиения (стратифицированно 70/10/20, seed 42, после дедупликации)\n")
    rows = []
    for d in SPLIT_DATASETS:
        s = split[d]
        for part in ["train", "val", "test"]:
            rows.append([d, part, s["sizes"][part], labels_str(s["labels"][part]),
                         labels_str(s["label_binary"][part]) if "label_binary" in s else "—"])
    md.append(table(["Датасет", "Сплит", "Размер", "Классы", "label_binary"], rows))
    md.append(
        "\nУ Mendeley есть колонка `label_binary`: ham = 0, spam и smishing = 1. Она есть в `clean.csv` и во "
        "всех сплитах. Процедура разбиения не менялась: стратификация по трёхклассовой `label`.\n"
    )
    md.append("Проверка утечек: повторный поиск по всем данным с более мягким порогом кандидатов LSH (0.5), "
              "Жаккар ≥ 0.9.\n")
    rows = []
    for d in SPLIT_DATASETS:
        lc = split[d]["leakage_check"]
        rows.append([d, lc["near_pairs_across_splits"], lc["near_pairs_within_split"],
                     lc["exact_duplicate_texts_remaining"]])
    md.append(table(["Датасет", "Почти-дубликаты между сплитами", "Почти-дубликаты внутри сплита",
                     "Оставшиеся точные дубликаты"], rows))
    md.append("\nUCI не разбивается: он используется только для проверки пересечений.\n")

    # 8. Problems
    md.append("## 8. Проблемы и что осталось сделать вручную\n")
    md.append("### Проблемы в данных\n")
    for p in problems(load, dedup, overlap, ssc):
        md.append(f"- {p}")
    md.append("\n### Требует ручного участия\n")
    for p in MANUAL:
        md.append(f"- {p}")
    md.append("")

    out = REPORTS / "step1_data_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")


def problems(load, dedup, overlap, ssc):
    m, z, s = load["mendeley"], load["moz_smishing"], load["super_sms"]
    om = overlap["matrix_exact_or_near"]
    dm, dz, ds = dedup["mendeley"], dedup["moz_smishing"], dedup["super_sms"]
    return [
        f"**Mendeley: метки в разном регистре** ({m['raw_label_values']}). Привёл к нижнему регистру; суммы "
        "сходятся с описанием.",
        f"**Mendeley: {m['rows_with_U+FFFD']} сообщений содержат символ замены U+FFFD**: кодировка была потеряна "
        "ещё в исходном файле, восстановить текст нельзя. Сообщения оставлены как есть.",
        f"**Mendeley: у {m['rows_with_leading_or_trailing_whitespace']} сообщений пробелы в начале или конце "
        f"текста, {m['rows_starting_with_tab']} начинаются с табуляции.** Похоже на артефакт OCR или конвертации. "
        "Текст в данных не менялся; при поиске дубликатов пробелы нормализуются.",
        "**Mendeley почти целиком пересекается с UCI**: "
        f"{om['mendeley']['uci_sms']} из {om['mendeley']['mendeley']} сообщений после дедупликации. Часть "
        "сообщений с меткой UCI `spam` в Mendeley размечена как `smishing` (см. таблицу в разделе 5), то есть "
        "граница spam/smishing задана разметкой Mendeley.",
        f"**Mendeley: все конфликтные группы ({sum(dm['conflict_groups'].values())}) — это spam против smishing** "
        f"на одинаковых или почти одинаковых текстах. Ничьих: {dm['conflict_resolution'].get('tie', 0)}, "
        f"решено большинством: {dm['conflict_resolution'].get('majority', 0)}. Граница между этими классами "
        "размечена непоследовательно; для `label_binary` эти конфликты несущественны (оба класса = 1), но "
        "группы с ничьей всё равно удалены.",
        f"**MOZ-Smishing: конфликты ham/smishing на одинаковых текстах** (групп: {sum(dz['conflict_groups'].values())}, "
        f"все с ничьей, удалены). На HF лежит только `test.csv`, официального train/test нет. Источники "
        f"сообщений: {z['raw_source_values']}.",
        "**MOZ-Smishing: лицензия creativeml-openrail-m** рассчитана на модели, а не на датасеты. Условия "
        "использования данных стоит уточнить.",
        f"**Super SMS: размер не совпадает с заявленным.** В файле {s['raw_rows']} строк (в статье 67018, в "
        f"аннотации «more than 68K»). Валидных {s['valid_rows']}: {s['invalid_rows']} строк отброшены "
        f"(пустая метка или пустой текст). Классы {s['labels']} против 40837/26181 в статье тех же авторов.",
        f"**Super SMS: кодировка смешанная.** {s['non_utf8_bytes_decoded_as_cp1252']} байта не из UTF-8 "
        f"прочитаны как cp1252. {s['rows_with_C1_control_chars']} сообщений содержат управляющие символы C1 "
        "(U+0080–U+009F), то есть исходный текст уже был испорчен на стороне источника (mojibake). Не исправлялось.",
        f"**Super SMS: предобработка неоднородная.** {s['rows_lowercase_alnum_only']} сообщений "
        f"({pct(s['rows_lowercase_alnum_only'], s['raw_rows'])}) состоят только из строчных латинских букв, цифр и "
        "пробелов: похоже, их уже привели к нижнему регистру и убрали пунктуацию (например, `httphikein`). "
        "Остальные выглядят как исходные. Модель может научиться отличать источник по стилю.",
        f"**Super SMS: группа шаблона hike** ({labels_str(ds['conflict_rows_by_label'])}) решена большинством: "
        "оставлено одно spam-сообщение, остальные spam удалены как почти-дубликаты, ham — как меньшинство.",
        "**Super SMS содержит UCI и Mendeley**: "
        f"{om['super_sms']['uci_sms']} сообщений Super SMS совпадают с UCI и {om['super_sms']['mendeley']} — "
        "с Mendeley. Для кросс-датасетных экспериментов с Mendeley сделан вариант `super_sms_clean` "
        f"({ssc['rows_after']} сообщений, раздел 6). Полный `super_sms` остаётся с этими пересечениями.",
        "**Super SMS: лицензия данных не указана.**",
        "**Цепочки в компонентах связности**: каждое звено имеет J ≥ 0.9, но крайние члены группы могут быть "
        "менее похожи друг на друга (минимальный J к оставленному представителю виден в `jaccard_to_kept`).",
        "**langdetect** на коротких SMS даёт шум (af, so, cy и т. п.); доли языков приблизительные.",
        f"**Mendeley ∩ UCI**: наша цифра не совпадает с литературной ({LITERATURE_MENDELEY_UCI}, "
        f"{LITERATURE_SOURCE}); указываем обе (см. раздел 5).",
    ]


MANUAL = [
    "Открытых вопросов по шагу 1 нет; принятые решения (только `super_sms_clean` в экспериментах, "
    "лицензии, цифра пересечения Mendeley и UCI, постановка задач на Mendeley) записаны в README, "
    "раздел Decisions.",
]


if __name__ == "__main__":
    main()
