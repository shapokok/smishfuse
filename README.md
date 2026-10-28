# SmishFuse

Исследовательский проект по детекции смишинга. Сейчас готов только шаг 1, данные: загрузка, проверка,
очистка и разбиение. Модели не обучаются.

Итоговый отчёт по данным: [reports/step1_data_report.md](reports/step1_data_report.md).

## Decisions

Решения по шагу 1, зафиксированы 2026-10-28.

1. **Super SMS.** Во всех экспериментах используется только `super_sms_clean`: Super SMS без сообщений,
   которые точно или почти (Жаккар ≥ 0.9) совпадают с Mendeley или UCI. Полный `super_sms` в экспериментах
   не участвует.
2. **Лицензии.** Сырые данные не распространяем, только скрипты загрузки с sha256-хэшами файлов. В статье
   указываем источники всех датасетов. У Super SMS лицензия не указана, у MOZ-Smishing стоит
   creativeml-openrail-m.
3. **Пересечение Mendeley и UCI.** Цифра 4753 общих сообщений (из них 4199 ham и 554 spam) взята из обзора
   Pritom et al., arXiv:2604.11429. Метод подсчёта в обзоре не описан. Наша оценка: 5025 сообщений Mendeley
   совпадают с UCI точно после нормализации (lowercase + схлопывание пробелов), 5143 — точно или почти
   (Жаккар ≥ 0.9 по символьным 5-граммам); обе цифры посчитаны до дедупликации. В статье указываем обе цифры
   и метод.
4. **Задачи на Mendeley.** Основная задача трёхклассовая (ham / spam / smishing), вторичная — бинарная
   (`label_binary`: ham = 0, spam и smishing = 1). 40 групп дубликатов с ничьей spam/smishing (80 сообщений,
   удалены при дедупликации) фиксируем как оценку шума разметки на границе spam/smishing.
5. **LinearSVM (шаг 2), правка 2026-10-28.** `CalibratedClassifierCV(ensemble=False)`: сигмоида
   калибруется по out-of-fold решениям 5 фолдов на train, итоговая модель — одна LinearSVC, обученная на всём
   train (раньше было `ensemble=True`, ансамбль из 5 SVM). LinearSVM перезапущен на всех 4 задачах на Kaggle
   CPU (ноутбук `smishfuse-svm-cpu`), таблицы и значимость пересчитаны.

## Воспроизведение данных

`data/raw/` и `data/processed/` не хранятся в репозитории (см. `.gitignore`), их полностью воссоздают скрипты.
Нужны Python 3.11 и [uv](https://docs.astral.sh/uv/); подойдёт и обычный `venv` + `pip`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/data/run_all.py
```

`run_all.py` скачивает сырые файлы по закреплённым ссылкам (коммит или версия) в `data/raw/<name>/` и
сверяет их sha256. Если файл на источнике изменился, запуск останавливается. Затем выполняются все шаги и
пишется отчёт. Полный прогон занимает около трёх минут. Результат детерминирован: повторные прогоны, в том
числе в чистом окружении, дают побайтно одинаковые файлы.

Каждый шаг можно запустить отдельно, в таком порядке:

| Скрипт | Что делает | Результат |
|---|---|---|
| `src/data/load_mendeley.py`, `load_moz_smishing.py`, `load_super_sms.py`, `load_uci_sms.py` | Скачивание, проверка sha256, приведение к схеме `id, text, label, dataset, language`, определение языка (langdetect), сверка размеров с ожидаемыми | `data/processed/<name>/unified.csv`, `reports/stats/load_<name>.json` |
| `src/data/dedup.py` | Точные дубликаты и почти-дубликаты внутри датасета (MinHash LSH, символьные 5-граммы, Жаккар ≥ 0.9), разрешение конфликтов меток большинством, при ничьей группа удаляется | `data/processed/<name>/clean.csv`, `reports/removed/<name>_dedup.csv`, `reports/conflicts/<name>.csv` |
| `src/data/make_super_sms_clean.py` | Super SMS без сообщений, совпадающих с Mendeley или UCI | `data/processed/super_sms_clean/clean.csv`, `reports/removed/super_sms_clean_cross.csv` |
| `src/data/overlap.py` | Матрица пересечений между датасетами, подробности по Mendeley и UCI | `reports/overlap_matrix*.csv`, `reports/overlap_mendeley_uci_pairs.csv` |
| `src/data/split.py` | Стратифицированное разбиение 70/10/20 (seed 42) и проверка утечек между сплитами | `data/processed/<name>/{train,val,test}.csv` |
| `src/data/report.py` | Отчёт из `reports/stats/*.json` | `reports/step1_data_report.md` |

Сплиты есть у `mendeley`, `moz_smishing`, `super_sms` и `super_sms_clean`. UCI используется только для
проверки пересечений. Колонки: `id, text, label, dataset, language`; у Mendeley добавлена `label_binary`
(ham = 0, spam и smishing = 1). Строки `super_sms_clean` сохраняют исходные `id` и `dataset = super_sms`.

## Шаг 2: модуль оценки и классические baseline

Нужны данные шага 1 (`data/processed/`). На macOS XGBoost требует `brew install libomp`.

```bash
.venv/bin/python src/run_step2.py
```

Скрипт по очереди запускает юнит-тесты (`tests/`; при падении шаг останавливается), подбор и финальную
оценку моделей (`src/baselines/classic.py`), таблицы (`scripts/make_tables.py`), тесты значимости
(`scripts/significance.py`) и отчёт [reports/step2_baselines_report.md](reports/step2_baselines_report.md).
Протокол и пространства поиска заданы в [configs/step2.yaml](configs/step2.yaml).

- `src/eval/metrics.py`: accuracy, macro и weighted F1, метрики по классам, balanced accuracy, PR-AUC,
  recall при FPR = 1%, ECE.
- `src/eval/stats.py`: парный бутстреп разницы macro F1, тест Макнемара, поправка Холма.
- `results/`: предсказания (`raw/`), метрики по запускам (`metrics.csv`), лучшие параметры, история подбора,
  таблицы (Markdown и LaTeX), `significance.csv`. Всё это побайтно воспроизводимо, кроме замеров времени:
  `efficiency.csv` и раздел 4 отчёта зависят от машины и её загрузки.

## Датасеты: источники и лицензии

| Датасет | Источник | Закреплённая версия | Лицензия |
|---|---|---|---|
| Mendeley SMS Phishing Dataset (Mishra, Soni) | https://data.mendeley.com/datasets/f45bkkt8pr/1, DOI 10.17632/f45bkkt8pr.1 | версия 1, `Dataset_5971.zip` | CC BY 4.0 |
| MOZ-Smishing (Ali et al., AfricaNLP 2025, https://aclanthology.org/2025.africanlp-1.23) | https://huggingface.co/datasets/MOZNLP/MOZ-Smishing | commit `1092f9d9`, `test.csv` | creativeml-openrail-m (тег на HF); в статье написано «open license» без названия |
| Super SMS Dataset (Salman, Ikram, Kaafar, IEEE Access 12, 2024, DOI 10.1109/ACCESS.2024.3364671) | https://github.com/smspamresearch/spstudy (ссылка из текста статьи) | commit `a0889f36`, `Data/super_sms_dataset.csv` | не указана: в репозитории нет файла LICENSE. CC BY-NC-ND 4.0 относится к статье, а не к данным |
| UCI SMS Spam Collection v.1 (Almeida, Gómez Hidalgo) | https://archive.ics.uci.edu/dataset/228/sms+spam+collection | `sms+spam+collection.zip` | CC BY 4.0 (страница UCI); readme в архиве: авторские права у авторов, использование бесплатно с просьбой ссылаться |

Smishing-4C исключён из проекта: доступ только по запросу.

При использовании данных ссылайтесь на исходные публикации. Перед публикацией производных данных уточните
условия использования Super SMS и MOZ-Smishing (см. раздел 8 отчёта).
