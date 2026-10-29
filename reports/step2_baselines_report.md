# SmishFuse, шаг 2: модуль оценки и классические baseline

Сгенерировано скриптом `scripts/make_step2_report.py` из `results/`. Запуск всего шага: `.venv/bin/python src/run_step2.py`. Нейросети не обучались.

## Протокол

- Задачи: `mendeley_3class`, `mendeley_binary`, `moz_binary`, `super_sms_clean_binary`. Данные — сплиты шага 1 (Super SMS только в варианте `super_sms_clean`).
- Модели: TF-IDF + LogReg, LinearSVM, RandomForest, XGBoost. Признаки TF-IDF (слова 1–2-граммы, символы 3–5-граммы или их объединение) выбираются подбором вместе с гиперпараметрами модели. Пространства поиска заданы в `configs/step2.yaml`.
- Подбор: Optuna TPE (seed 42), ровно 20 попыток на пару модель-задача. Каждая попытка обучается на train и оценивается по macro F1 на val. Test при подборе не читается.
- Финальная оценка: лучшие параметры, seed 1..5. Seed задаёт random_state модели и перемешивание порядка train. Обучение только на train, одно предсказание на test.
- Метрики: accuracy, macro и weighted F1, precision/recall/F1 по классам, balanced accuracy, PR-AUC (для бинарных задач по положительному классу, для трёх классов — среднее one-vs-rest), recall при FPR = 1% (бинарные задачи, положительный класс = malicious), ECE (15 бинов, top-label). Предсказание = argmax вероятностей.
- Значимость: лучшая по среднему macro F1 модель сравнивается с остальными на предсказаниях seed 1: парный бутстреп разницы macro F1 (10000 повторов, seed 42, 95% ДИ по перцентилям) и тест Макнемара (точный биномиальный при числе расхождений < 25, иначе χ² с поправкой на непрерывность). Поправка Холма применяется внутри каждой задачи, отдельно для каждого теста, α = 0.05.
- Юнит-тесты `tests/` (сверка с sklearn и ручным расчётом) запускаются первым шагом `run_step2.py`; при падении шаг останавливается.

## 1. Итоговые таблицы (test, среднее ± стандартное отклонение по 5 seed)

Жирным — лучшее значение в столбце, подчёркнуто — второе среди классических моделей (для ECE лучше меньшее). Таблицы со всеми моделями шагов 2 и 3: `results/tables/<task>.md|.tex`; все запуски: `results/metrics.csv`; предсказания: `results/raw/<task>/<model>/seed<k>.csv`.

### mendeley_3class

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | ECE | Recall ham | Recall smishing | Recall spam |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | <u>0.9181 ± 0.0012</u> | <u>0.9767 ± 0.0004</u> | <u>0.9760 ± 0.0004</u> | <u>0.9021 ± 0.0018</u> | <u>0.9616 ± 0.0000</u> | <u>0.0136 ± 0.0004</u> | **0.9990 ± 0.0000** | **0.9192 ± 0.0000** | 0.7881 ± 0.0053 |
| TF-IDF + LinearSVM | **0.9259 ± 0.0019** | **0.9791 ± 0.0006** | **0.9791 ± 0.0006** | **0.9255 ± 0.0017** | **0.9659 ± 0.0002** | 0.0187 ± 0.0011 | 0.9956 ± 0.0005 | <u>0.9071 ± 0.0045</u> | **0.8738 ± 0.0065** |
| TF-IDF + RandomForest | 0.9130 ± 0.0045 | 0.9732 ± 0.0013 | 0.9723 ± 0.0013 | 0.8902 ± 0.0032 | 0.9606 ± 0.0009 | 0.0875 ± 0.0011 | <u>0.9985 ± 0.0009</u> | 0.8768 ± 0.0085 | <u>0.7952 ± 0.0053</u> |
| TF-IDF + XGBoost | 0.8991 ± 0.0058 | 0.9667 ± 0.0019 | 0.9657 ± 0.0020 | 0.8768 ± 0.0072 | 0.9594 ± 0.0015 | **0.0110 ± 0.0019** | 0.9942 ± 0.0014 | 0.8646 ± 0.0115 | 0.7714 ± 0.0177 |

### mendeley_binary

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | Recall@FPR=1% | ECE | Recall malicious |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | <u>0.9788 ± 0.0000</u> | <u>0.9887 ± 0.0000</u> | <u>0.9887 ± 0.0000</u> | <u>0.9778 ± 0.0000</u> | **0.9955 ± 0.0000** | **0.9836 ± 0.0000** | <u>0.0165 ± 0.0000</u> | <u>0.9617 ± 0.0000</u> |
| TF-IDF + LinearSVM | **0.9883 ± 0.0007** | **0.9937 ± 0.0004** | **0.9937 ± 0.0004** | **0.9896 ± 0.0002** | <u>0.9954 ± 0.0000</u> | **0.9836 ± 0.0000** | **0.0063 ± 0.0006** | **0.9836 ± 0.0000** |
| TF-IDF + RandomForest | 0.9720 ± 0.0031 | 0.9852 ± 0.0016 | 0.9851 ± 0.0016 | 0.9664 ± 0.0038 | 0.9914 ± 0.0010 | <u>0.9781 ± 0.0067</u> | 0.0878 ± 0.0007 | 0.9388 ± 0.0071 |
| TF-IDF + XGBoost | 0.9582 ± 0.0019 | 0.9777 ± 0.0010 | 0.9776 ± 0.0010 | 0.9557 ± 0.0026 | 0.9850 ± 0.0002 | 0.9202 ± 0.0049 | 0.0212 ± 0.0018 | 0.9235 ± 0.0055 |

### moz_binary

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | Recall@FPR=1% | ECE | Recall malicious |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | <u>0.9797 ± 0.0000</u> | <u>0.9861 ± 0.0000</u> | <u>0.9863 ± 0.0000</u> | **0.9912 ± 0.0000** | **0.9811 ± 0.0000** | 0.8785 ± 0.0000 | 0.0136 ± 0.0000 | **1.0000 ± 0.0000** |
| TF-IDF + LinearSVM | 0.9774 ± 0.0013 | 0.9846 ± 0.0009 | 0.9847 ± 0.0009 | <u>0.9895 ± 0.0018</u> | <u>0.9793 ± 0.0000</u> | **0.9439 ± 0.0000** | <u>0.0099 ± 0.0018</u> | <u>0.9981 ± 0.0042</u> |
| TF-IDF + RandomForest | **0.9825 ± 0.0000** | **0.9881 ± 0.0000** | **0.9882 ± 0.0000** | 0.9890 ± 0.0000 | 0.9782 ± 0.0017 | <u>0.9346 ± 0.0229</u> | 0.0525 ± 0.0009 | 0.9907 ± 0.0000 |
| TF-IDF + XGBoost | 0.9762 ± 0.0024 | 0.9838 ± 0.0017 | 0.9839 ± 0.0016 | 0.9870 ± 0.0025 | 0.9700 ± 0.0027 | 0.8617 ± 0.0656 | **0.0086 ± 0.0028** | 0.9925 ± 0.0042 |

### super_sms_clean_binary

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | Recall@FPR=1% | ECE | Recall malicious |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | <u>0.9909 ± 0.0000</u> | <u>0.9913 ± 0.0000</u> | <u>0.9913 ± 0.0000</u> | <u>0.9902 ± 0.0000</u> | **0.9991 ± 0.0000** | <u>0.9933 ± 0.0000</u> | <u>0.0047 ± 0.0000</u> | <u>0.9847 ± 0.0000</u> |
| TF-IDF + LinearSVM | **0.9927 ± 0.0000** | **0.9930 ± 0.0000** | **0.9930 ± 0.0000** | **0.9923 ± 0.0000** | **0.9991 ± 0.0000** | **0.9941 ± 0.0000** | **0.0014 ± 0.0001** | **0.9892 ± 0.0001** |
| TF-IDF + RandomForest | 0.9872 ± 0.0003 | 0.9877 ± 0.0003 | 0.9877 ± 0.0003 | 0.9864 ± 0.0003 | 0.9980 ± 0.0001 | 0.9832 ± 0.0008 | 0.0553 ± 0.0005 | 0.9800 ± 0.0006 |
| TF-IDF + XGBoost | 0.9895 ± 0.0003 | 0.9899 ± 0.0003 | 0.9899 ± 0.0003 | 0.9890 ± 0.0003 | <u>0.9985 ± 0.0001</u> | 0.9885 ± 0.0004 | **0.0014 ± 0.0002** | 0.9845 ± 0.0007 |

## 2. Лучшие гиперпараметры (подбор на val)

| Задача | Модель | Признаки | max_features | min_df | Параметры модели | Macro F1 val | Лучшая попытка | Macro F1 test (среднее) |
|---|---|---|---|---|---|---|---|---|
| mendeley_3class | LogReg | word+char | 100000 | 2 | C=544.6, class_weight=None | 0.9267 | 10 | 0.9181 |
| mendeley_3class | LinearSVM | word+char | 20000 | 1 | C=0.5827, class_weight=balanced | 0.9167 | 16 | 0.9259 |
| mendeley_3class | RandomForest | char | 50000 | 2 | class_weight=balanced, max_depth=None, max_features=sqrt, min_samples_leaf=1, n_estimators=500 | 0.9160 | 18 | 0.9130 |
| mendeley_3class | XGBoost | char | 20000 | 2 | class_weight=balanced, colsample_bytree=0.9474, learning_rate=0.1244, max_depth=10, min_child_weight=1.292, n_estimators=100, reg_lambda=0.09476, subsample=0.699 | 0.9045 | 19 | 0.8991 |
| mendeley_binary | LogReg | char | 20000 | 2 | C=34.22, class_weight=balanced | 0.9801 | 6 | 0.9788 |
| mendeley_binary | LinearSVM | char | 100000 | 1 | C=2.695, class_weight=balanced | 0.9869 | 12 | 0.9883 |
| mendeley_binary | RandomForest | char | 50000 | 1 | class_weight=balanced, max_depth=None, max_features=sqrt, min_samples_leaf=2, n_estimators=300 | 0.9626 | 5 | 0.9720 |
| mendeley_binary | XGBoost | char | 20000 | 2 | class_weight=balanced, colsample_bytree=0.6681, learning_rate=0.02288, max_depth=5, min_child_weight=1.784, n_estimators=550, reg_lambda=0.2202, subsample=0.9717 | 0.9664 | 11 | 0.9582 |
| moz_binary | LogReg | char | 20000 | 2 | C=10.13, class_weight=None | 0.9704 | 0 | 0.9797 |
| moz_binary | LinearSVM | char | 50000 | 2 | C=72.9, class_weight=None | 0.9768 | 13 | 0.9774 |
| moz_binary | RandomForest | word+char | 100000 | 1 | class_weight=None, max_depth=100, max_features=sqrt, min_samples_leaf=1, n_estimators=300 | 0.9700 | 3 | 0.9825 |
| moz_binary | XGBoost | word+char | 100000 | 1 | class_weight=balanced, colsample_bytree=0.5104, learning_rate=0.2038, max_depth=4, min_child_weight=1.902, n_estimators=100, reg_lambda=9.396, subsample=0.6203 | 0.9765 | 10 | 0.9762 |
| super_sms_clean_binary | LogReg | word+char | 20000 | 1 | C=10.91, class_weight=None | 0.9910 | 2 | 0.9909 |
| super_sms_clean_binary | LinearSVM | word+char | 50000 | 2 | C=0.3431, class_weight=balanced | 0.9920 | 18 | 0.9927 |
| super_sms_clean_binary | RandomForest | char | 50000 | 1 | class_weight=balanced, max_depth=None, max_features=sqrt, min_samples_leaf=1, n_estimators=200 | 0.9865 | 7 | 0.9872 |
| super_sms_clean_binary | XGBoost | char | 50000 | 1 | class_weight=balanced, colsample_bytree=0.8847, learning_rate=0.08534, max_depth=8, min_child_weight=2.093, n_estimators=400, reg_lambda=0.03627, subsample=0.6361 | 0.9891 | 9 | 0.9895 |

Все попытки: `results/tuning/<task>/<model>.csv`; лучшие параметры: `results/best_params/<task>/<model>.json`.

## 3. Значимость различий (seed 1, лучшая модель против остальных)

| Задача | Лучшая | Против | Δ macro F1 [95% ДИ] | p бутстреп / Холм | Значимо (бутстреп) | Макнемар b / c | Метод | p Макнемар / Холм | Значимо (Макнемар) |
|---|---|---|---|---|---|---|---|---|---|
| mendeley_3class | LinearSVM | LogReg | +0.0109 [-0.0082; +0.0311] | 0.2616 / 0.4600 | нет | 9 / 5 | exact | 0.4240 / 0.4240 | нет |
| mendeley_3class | LinearSVM | RandomForest | +0.0145 [-0.0092; +0.0396] | 0.2300 / 0.4600 | нет | 12 / 5 | exact | 0.1435 / 0.2869 | нет |
| mendeley_3class | LinearSVM | XGBoost | +0.0290 [-0.0013; +0.0589] | 0.0614 / 0.1842 | нет | 22 / 7 | chi2_cc | 0.0093 / 0.0280 | да |
| mendeley_binary | LinearSVM | LogReg | +0.0098 [+0.0031; +0.0186] | 0.0040 / 0.0040 | да | 6 / 0 | exact | 0.0312 / 0.0312 | да |
| mendeley_binary | LinearSVM | RandomForest | +0.0167 [+0.0051; +0.0299] | 0.0020 / 0.0040 | да | 12 / 2 | exact | 0.0129 / 0.0259 | да |
| mendeley_binary | LinearSVM | XGBoost | +0.0313 [+0.0158; +0.0479] | 2.00e-04 / 6.00e-04 | да | 22 / 3 | chi2_cc | 3.18e-04 / 9.55e-04 | да |
| moz_binary | RandomForest | LogReg | +0.0027 [-0.0064; +0.0130] | 0.8055 / 1.0000 | нет | 2 / 1 | exact | 1.0000 / 1.0000 | нет |
| moz_binary | RandomForest | LinearSVM | +0.0027 [-0.0093; +0.0154] | 0.8375 / 1.0000 | нет | 3 / 2 | exact | 1.0000 / 1.0000 | нет |
| moz_binary | RandomForest | XGBoost | +0.0057 [+0.0000; +0.0148] | 0.2670 / 0.8009 | нет | 2 / 0 | exact | 0.5000 / 1.0000 | нет |
| super_sms_clean_binary | LinearSVM | LogReg | +0.0018 [+0.0008; +0.0029] | 6.00e-04 / 6.00e-04 | да | 32 / 11 | chi2_cc | 0.0023 / 0.0023 | да |
| super_sms_clean_binary | LinearSVM | RandomForest | +0.0054 [+0.0037; +0.0071] | 2.00e-04 / 6.00e-04 | да | 89 / 26 | chi2_cc | 7.40e-09 / 2.22e-08 | да |
| super_sms_clean_binary | LinearSVM | XGBoost | +0.0034 [+0.0017; +0.0051] | 2.00e-04 / 6.00e-04 | да | 71 / 31 | chi2_cc | 1.13e-04 / 2.25e-04 | да |

b — только лучшая модель права, c — только другая. Полная таблица: `results/significance_step2.csv` (сравнение всех моделей шагов 2 и 3 — `results/significance.csv`, отчёт шага 3).

## 4. Задержка и размер модели

Единый замер шага 3: одна сессия Kaggle для всех моделей шагов 2 и 3, Intel(R) Xeon(R) CPU @ 2.00GHz, 1 поток. Пайплайн заново обучен на train с лучшими параметрами и seed 1; задержка — медиана по 1000 предсказаниям с батчем 1 (TF-IDF + модель, после 20 прогревочных вызовов) на сообщениях test, по кругу, если их меньше; размер — сериализованный joblib-пайплайн в МБ (`results/latency/latency.csv`). Время обучения в итоговые таблицы не входит (решение 10 в README): модели обучались на разных машинах. Прежние локальные замеры остаются в `results/efficiency.csv`. Задержка зависит от машины и нагрузки и не воспроизводится побайтно.

| Задача | Модель | Train | Задержка, мс/сообщение | Размер, МБ |
|---|---|---|---|---|
| mendeley_3class | LogReg | 4017 | 1.926 | 1.78 |
| mendeley_3class | LinearSVM | 4017 | 2.817 | 2.26 |
| mendeley_3class | RandomForest | 4017 | 8.140 | 5.48 |
| mendeley_3class | XGBoost | 4017 | 1.538 | 2.86 |
| mendeley_binary | LogReg | 4017 | 0.912 | 0.69 |
| mendeley_binary | LinearSVM | 4017 | 1.756 | 3.85 |
| mendeley_binary | RandomForest | 4017 | 53.095 | 17.47 |
| mendeley_binary | XGBoost | 4017 | 1.367 | 1.45 |
| moz_binary | LogReg | 1765 | 1.012 | 0.69 |
| moz_binary | LinearSVM | 1765 | 1.728 | 1.26 |
| moz_binary | RandomForest | 1765 | 38.388 | 7.39 |
| moz_binary | XGBoost | 1765 | 2.590 | 3.33 |
| super_sms_clean_binary | LogReg | 42857 | 2.149 | 1.45 |
| super_sms_clean_binary | LinearSVM | 42857 | 2.899 | 4.06 |
| super_sms_clean_binary | RandomForest | 42857 | 20.941 | 41.58 |
| super_sms_clean_binary | XGBoost | 42857 | 1.432 | 2.41 |

## 5. Наблюдения

- **mendeley_3class**: лучшая модель по среднему macro F1 — LinearSVM (0.9259 ± 0.0019).
- mendeley_3class: классы, где средний по seed recall хотя бы одной модели ниже 0.90 (в скобках recall всех моделей): `smishing` (LogReg 0.919, LinearSVM 0.907, RandomForest 0.877, XGBoost 0.865); `spam` (LogReg 0.788, LinearSVM 0.874, RandomForest 0.795, XGBoost 0.771).
- mendeley_3class: macro F1 на val (лучшая попытка) → test (среднее): LogReg 0.9267 → 0.9181; LinearSVM 0.9167 → 0.9259; RandomForest 0.9160 → 0.9130; XGBoost 0.9045 → 0.8991.
- mendeley_3class: наибольший ECE у RandomForest (0.0875), наименьший у XGBoost (0.0110).
- mendeley_3class: после поправки Холма значимых различий с лучшей моделью: бутстреп 0 из 3, Макнемар 1 из 3.
- **mendeley_binary**: лучшая модель по среднему macro F1 — LinearSVM (0.9883 ± 0.0007).
- mendeley_binary: recall при FPR = 1% от 0.9202 (XGBoost) до 0.9836 (LinearSVM).
- mendeley_binary: macro F1 на val (лучшая попытка) → test (среднее): LogReg 0.9801 → 0.9788; LinearSVM 0.9869 → 0.9883; RandomForest 0.9626 → 0.9720; XGBoost 0.9664 → 0.9582.
- mendeley_binary: стандартное отклонение macro F1 по seed равно 0 у: LogReg.
- mendeley_binary: наибольший ECE у RandomForest (0.0878), наименьший у LinearSVM (0.0063).
- mendeley_binary: после поправки Холма значимых различий с лучшей моделью: бутстреп 3 из 3, Макнемар 3 из 3.
- **moz_binary**: лучшая модель по среднему macro F1 — RandomForest (0.9825 ± 0.0000).
- moz_binary: recall при FPR = 1% от 0.8617 (XGBoost) до 0.9439 (LinearSVM).
- moz_binary: macro F1 на val (лучшая попытка) → test (среднее): LogReg 0.9704 → 0.9797; LinearSVM 0.9768 → 0.9774; RandomForest 0.9700 → 0.9825; XGBoost 0.9765 → 0.9762.
- moz_binary: стандартное отклонение macro F1 по seed равно 0 у: LogReg, RandomForest.
- moz_binary: наибольший ECE у RandomForest (0.0525), наименьший у XGBoost (0.0086).
- moz_binary: после поправки Холма значимых различий с лучшей моделью: бутстреп 0 из 3, Макнемар 0 из 3.
- **super_sms_clean_binary**: лучшая модель по среднему macro F1 — LinearSVM (0.9927 ± 0.0000).
- super_sms_clean_binary: recall при FPR = 1% от 0.9832 (RandomForest) до 0.9941 (LinearSVM).
- super_sms_clean_binary: macro F1 на val (лучшая попытка) → test (среднее): LogReg 0.9910 → 0.9909; LinearSVM 0.9920 → 0.9927; RandomForest 0.9865 → 0.9872; XGBoost 0.9891 → 0.9895.
- super_sms_clean_binary: стандартное отклонение macro F1 по seed равно 0 у: LogReg.
- super_sms_clean_binary: наибольший ECE у RandomForest (0.0553), наименьший у LinearSVM (0.0014).
- super_sms_clean_binary: после поправки Холма значимых различий с лучшей моделью: бутстреп 3 из 3, Макнемар 3 из 3.

## 6. Проблемы и ограничения

- LinearSVC не выдаёт вероятностей, поэтому он обёрнут в `CalibratedClassifierCV(method='sigmoid', cv=5, ensemble=False)`: сигмоида обучается на out-of-fold решениях 5 фолдов train, итоговая модель — одна SVM на всём train. Предсказание — argmax калиброванных вероятностей; PR-AUC и ECE относятся к калиброванной модели.
- Все модели шага 2 обучены на Kaggle CPU с одним и тем же кодом и версиями: LinearSVM — ноутбук `smishfuse-svm-cpu`, LogReg, RandomForest и XGBoost — `smishfuse-classic-cpu` (шаг 4в, решение 14 в README). Прежние локальные результаты этих трёх моделей и сравнение с Kaggle: `results/step4c/step2_local_vs_kaggle.csv`.
- Задержка и размер (раздел 4) измерены в шаге 3 на моделях, заново обученных с лучшими параметрами локального подбора. Для LogReg, RandomForest и XGBoost лучшие параметры после пересчёта на Kaggle CPU изменились (кроме RandomForest на moz_binary, LogReg на mendeley_binary и XGBoost на super_sms_clean_binary), поэтому их задержка и размер относятся к прежним параметрам и не пересчитывались.
- LogReg (lbfgs) не содержит случайности: seed влияет только на порядок train, поэтому разброс по seed у неё нулевой или почти нулевой. Это не 5 независимых моделей.
- Пространство признаков помимо двух заданных вариантов (слова 1–2, символы 3–5) включает их объединение `word+char` как третий вариант (см. `configs/step2.yaml`).
- 20 попыток TPE — грубый поиск: первые 10 попыток TPE в Optuna по умолчанию случайные. Разница между моделями частично может объясняться бюджетом подбора.
- Значимость проверяется только на предсказаниях одного seed (seed 1), как требует протокол. Разброс между seed в тесты не входит.
- Размеры test по классам: mendeley_3class: ham 965, smishing 99, spam 84; mendeley_binary: benign 965, malicious 183; moz_binary: benign 398, malicious 107; super_sms_clean_binary: benign 7349, malicious 4896. На Mendeley и MOZ редкие классы насчитывают около сотни сообщений, поэтому доверительные интервалы там широкие, а разница в 1–3 ошибки заметно двигает метрики.
- RandomForest обучается в 8 потоков, а предсказывает в одном: параллельное суммирование вероятностей деревьев в sklearn меняет последние биты результата от запуска к запуску.
- Задержка измерена в одном потоке на Kaggle CPU (единый замер шага 3); при другой нагрузке или железе числа будут другими. У moz_binary в test 505 сообщений, поэтому для 1000 измерений сообщения повторяются.
- XGBoost на macOS требует OpenMP-рантайм (`brew install libomp`).
