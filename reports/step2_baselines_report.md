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

Жирным — лучшее значение в столбце, подчёркнуто — второе (для ECE лучше меньшее). LaTeX-версии: `results/tables/<task>.tex`; все запуски: `results/metrics.csv`; предсказания: `results/raw/<task>/<model>/seed<k>.csv`.

### mendeley_3class

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | ECE | Recall ham | Recall smishing | Recall spam |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | **0.9294 ± 0.0021** | **0.9794 ± 0.0005** | **0.9794 ± 0.0005** | **0.9293 ± 0.0022** | **0.9673 ± 0.0003** | <u>0.0287 ± 0.0004</u> | 0.9948 ± 0.0000 | **0.9192 ± 0.0000** | **0.8738 ± 0.0065** |
| TF-IDF + LinearSVM | <u>0.9259 ± 0.0019</u> | <u>0.9791 ± 0.0006</u> | <u>0.9791 ± 0.0006</u> | <u>0.9255 ± 0.0017</u> | <u>0.9659 ± 0.0002</u> | **0.0187 ± 0.0011** | <u>0.9956 ± 0.0005</u> | <u>0.9071 ± 0.0045</u> | **0.8738 ± 0.0065** |
| TF-IDF + RandomForest | 0.9019 ± 0.0062 | 0.9706 ± 0.0020 | 0.9697 ± 0.0022 | 0.8843 ± 0.0086 | 0.9556 ± 0.0025 | 0.0903 ± 0.0014 | **0.9967 ± 0.0009** | 0.8848 ± 0.0055 | <u>0.7714 ± 0.0258</u> |
| TF-IDF + XGBoost | 0.9001 ± 0.0052 | 0.9693 ± 0.0017 | 0.9686 ± 0.0017 | 0.8859 ± 0.0043 | 0.9564 ± 0.0011 | 0.0338 ± 0.0024 | 0.9944 ± 0.0012 | 0.8990 ± 0.0071 | 0.7643 ± 0.0100 |

### mendeley_binary

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | Recall@FPR=1% | ECE | Recall malicious |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | <u>0.9788 ± 0.0000</u> | <u>0.9887 ± 0.0000</u> | <u>0.9887 ± 0.0000</u> | <u>0.9778 ± 0.0000</u> | **0.9956 ± 0.0000** | <u>0.9836 ± 0.0000</u> | 0.0168 ± 0.0000 | <u>0.9617 ± 0.0000</u> |
| TF-IDF + LinearSVM | **0.9883 ± 0.0007** | **0.9937 ± 0.0004** | **0.9937 ± 0.0004** | **0.9896 ± 0.0002** | <u>0.9954 ± 0.0000</u> | <u>0.9836 ± 0.0000</u> | **0.0063 ± 0.0006** | **0.9836 ± 0.0000** |
| TF-IDF + RandomForest | 0.9756 ± 0.0018 | 0.9871 ± 0.0010 | 0.9870 ± 0.0010 | 0.9693 ± 0.0025 | 0.9919 ± 0.0008 | **0.9847 ± 0.0024** | 0.0933 ± 0.0010 | 0.9432 ± 0.0049 |
| TF-IDF + XGBoost | 0.9591 ± 0.0008 | 0.9784 ± 0.0004 | 0.9782 ± 0.0004 | 0.9522 ± 0.0012 | 0.9830 ± 0.0004 | 0.9158 ± 0.0049 | <u>0.0158 ± 0.0014</u> | 0.9137 ± 0.0024 |

### moz_binary

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | Recall@FPR=1% | ECE | Recall malicious |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.9769 ± 0.0000 | 0.9842 ± 0.0000 | 0.9844 ± 0.0000 | **0.9899 ± 0.0000** | **0.9812 ± 0.0000** | 0.8879 ± 0.0000 | <u>0.0125 ± 0.0000</u> | **1.0000 ± 0.0000** |
| TF-IDF + LinearSVM | <u>0.9774 ± 0.0013</u> | <u>0.9846 ± 0.0009</u> | <u>0.9847 ± 0.0009</u> | <u>0.9895 ± 0.0018</u> | <u>0.9793 ± 0.0000</u> | **0.9439 ± 0.0000** | **0.0099 ± 0.0018** | <u>0.9981 ± 0.0042</u> |
| TF-IDF + RandomForest | **0.9825 ± 0.0000** | **0.9881 ± 0.0000** | **0.9882 ± 0.0000** | 0.9890 ± 0.0000 | 0.9782 ± 0.0017 | <u>0.9346 ± 0.0229</u> | 0.0525 ± 0.0009 | 0.9907 ± 0.0000 |
| TF-IDF + XGBoost | 0.9745 ± 0.0013 | 0.9826 ± 0.0009 | 0.9828 ± 0.0009 | 0.9862 ± 0.0021 | 0.9736 ± 0.0032 | 0.7701 ± 0.0441 | 0.0179 ± 0.0028 | 0.9925 ± 0.0042 |

### super_sms_clean_binary

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | Recall@FPR=1% | ECE | Recall malicious |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | <u>0.9912 ± 0.0000</u> | <u>0.9916 ± 0.0000</u> | <u>0.9916 ± 0.0000</u> | <u>0.9906 ± 0.0000</u> | **0.9991 ± 0.0000** | <u>0.9924 ± 0.0000</u> | 0.0026 ± 0.0000 | <u>0.9859 ± 0.0000</u> |
| TF-IDF + LinearSVM | **0.9927 ± 0.0000** | **0.9930 ± 0.0000** | **0.9930 ± 0.0000** | **0.9923 ± 0.0000** | **0.9991 ± 0.0000** | **0.9941 ± 0.0000** | **0.0014 ± 0.0001** | **0.9892 ± 0.0001** |
| TF-IDF + RandomForest | 0.9861 ± 0.0006 | 0.9866 ± 0.0005 | 0.9866 ± 0.0005 | 0.9865 ± 0.0005 | 0.9981 ± 0.0001 | 0.9836 ± 0.0004 | 0.1451 ± 0.0010 | 0.9856 ± 0.0006 |
| TF-IDF + XGBoost | 0.9895 ± 0.0003 | 0.9899 ± 0.0003 | 0.9899 ± 0.0003 | 0.9890 ± 0.0003 | <u>0.9986 ± 0.0001</u> | 0.9883 ± 0.0008 | <u>0.0016 ± 0.0004</u> | 0.9845 ± 0.0004 |

## 2. Лучшие гиперпараметры (подбор на val)

| Задача | Модель | Признаки | max_features | min_df | Параметры модели | Macro F1 val | Лучшая попытка | Macro F1 test (среднее) |
|---|---|---|---|---|---|---|---|---|
| mendeley_3class | LogReg | word+char | 20000 | 1 | C=5.827, class_weight=balanced | 0.9207 | 16 | 0.9294 |
| mendeley_3class | LinearSVM | word+char | 20000 | 1 | C=0.5827, class_weight=balanced | 0.9167 | 16 | 0.9259 |
| mendeley_3class | RandomForest | char | 20000 | 1 | class_weight=balanced, max_depth=None, max_features=sqrt, min_samples_leaf=1, n_estimators=100 | 0.9196 | 12 | 0.9019 |
| mendeley_3class | XGBoost | char | 50000 | 1 | class_weight=balanced, colsample_bytree=0.5497, learning_rate=0.0417, max_depth=6, min_child_weight=1.91, n_estimators=250, reg_lambda=0.4247, subsample=0.9315 | 0.8958 | 3 | 0.9001 |
| mendeley_binary | LogReg | char | 20000 | 2 | C=34.22, class_weight=balanced | 0.9801 | 6 | 0.9788 |
| mendeley_binary | LinearSVM | char | 100000 | 1 | C=2.695, class_weight=balanced | 0.9869 | 12 | 0.9883 |
| mendeley_binary | RandomForest | word+char | 50000 | 1 | class_weight=balanced, max_depth=None, max_features=sqrt, min_samples_leaf=2, n_estimators=450 | 0.9626 | 12 | 0.9756 |
| mendeley_binary | XGBoost | char | 20000 | 2 | class_weight=balanced, colsample_bytree=0.8827, learning_rate=0.02115, max_depth=8, min_child_weight=1.631, n_estimators=400, reg_lambda=0.03511, subsample=0.988 | 0.9694 | 0 | 0.9591 |
| moz_binary | LogReg | char | 20000 | 2 | C=12.9, class_weight=None | 0.9704 | 13 | 0.9769 |
| moz_binary | LinearSVM | char | 50000 | 2 | C=72.9, class_weight=None | 0.9768 | 13 | 0.9774 |
| moz_binary | RandomForest | word+char | 100000 | 1 | class_weight=None, max_depth=100, max_features=sqrt, min_samples_leaf=1, n_estimators=300 | 0.9700 | 3 | 0.9825 |
| moz_binary | XGBoost | word+char | 100000 | 1 | class_weight=balanced, colsample_bytree=0.4322, learning_rate=0.06693, max_depth=3, min_child_weight=1.708, n_estimators=100, reg_lambda=9.267, subsample=0.8393 | 0.9704 | 15 | 0.9745 |
| super_sms_clean_binary | LogReg | word+char | 20000 | 1 | C=22.78, class_weight=balanced | 0.9908 | 14 | 0.9912 |
| super_sms_clean_binary | LinearSVM | word+char | 50000 | 2 | C=0.3431, class_weight=balanced | 0.9920 | 18 | 0.9927 |
| super_sms_clean_binary | RandomForest | char | 50000 | 1 | class_weight=balanced_subsample, max_depth=None, max_features=log2, min_samples_leaf=2, n_estimators=200 | 0.9871 | 15 | 0.9861 |
| super_sms_clean_binary | XGBoost | char | 50000 | 1 | class_weight=balanced, colsample_bytree=0.8847, learning_rate=0.08534, max_depth=8, min_child_weight=2.093, n_estimators=400, reg_lambda=0.03627, subsample=0.6361 | 0.9900 | 9 | 0.9895 |

Все попытки: `results/tuning/<task>/<model>.csv`; лучшие параметры: `results/best_params/<task>/<model>.json`.

## 3. Значимость различий (seed 1, лучшая модель против остальных)

| Задача | Лучшая | Против | Δ macro F1 [95% ДИ] | p бутстреп / Холм | Значимо (бутстреп) | Макнемар b / c | Метод | p Макнемар / Холм | Значимо (Макнемар) |
|---|---|---|---|---|---|---|---|---|---|
| mendeley_3class | LogReg | LinearSVM | +0.0032 [-0.0042; +0.0128] | 0.5639 / 0.5639 | нет | 1 / 1 | exact | 1.0000 / 1.0000 | нет |
| mendeley_3class | LogReg | RandomForest | +0.0329 [+0.0069; +0.0601] | 0.0118 / 0.0354 | да | 17 / 6 | exact | 0.0347 / 0.0694 | нет |
| mendeley_3class | LogReg | XGBoost | +0.0371 [+0.0079; +0.0677] | 0.0132 / 0.0354 | да | 18 / 5 | exact | 0.0106 / 0.0319 | да |
| mendeley_binary | LinearSVM | LogReg | +0.0098 [+0.0031; +0.0186] | 0.0040 / 0.0048 | да | 6 / 0 | exact | 0.0312 / 0.0312 | да |
| mendeley_binary | LinearSVM | RandomForest | +0.0150 [+0.0049; +0.0269] | 0.0024 / 0.0048 | да | 10 / 1 | exact | 0.0117 / 0.0234 | да |
| mendeley_binary | LinearSVM | XGBoost | +0.0282 [+0.0132; +0.0443] | 2.00e-04 / 6.00e-04 | да | 20 / 3 | exact | 4.88e-04 / 0.0015 | да |
| moz_binary | RandomForest | LogReg | +0.0055 [-0.0057; +0.0172] | 0.4542 / 0.9083 | нет | 3 / 1 | exact | 0.6250 / 1.0000 | нет |
| moz_binary | RandomForest | LinearSVM | +0.0027 [-0.0093; +0.0154] | 0.8375 / 0.9083 | нет | 3 / 2 | exact | 1.0000 / 1.0000 | нет |
| moz_binary | RandomForest | XGBoost | +0.0085 [+0.0000; +0.0199] | 0.0964 / 0.2892 | нет | 3 / 0 | exact | 0.2500 / 0.7500 | нет |
| super_sms_clean_binary | LinearSVM | LogReg | +0.0015 [+0.0005; +0.0025] | 0.0016 / 0.0016 | да | 25 / 8 | chi2_cc | 0.0053 / 0.0053 | да |
| super_sms_clean_binary | LinearSVM | RandomForest | +0.0064 [+0.0045; +0.0084] | 2.00e-04 / 6.00e-04 | да | 104 / 28 | chi2_cc | 6.67e-11 / 2.00e-10 | да |
| super_sms_clean_binary | LinearSVM | XGBoost | +0.0032 [+0.0015; +0.0049] | 2.00e-04 / 6.00e-04 | да | 69 / 32 | chi2_cc | 3.41e-04 / 6.82e-04 | да |

b — только лучшая модель права, c — только другая. Полная таблица: `results/significance.csv`.

## 4. Время обучения, задержка, размер модели

Медиана по 5 seed. Время обучения — wall-clock `fit` всего пайплайна (TF-IDF + модель, для LinearSVM вместе с калибровкой) на train. Задержка — медиана по 1000 предсказаниям с батчем 1 (TF-IDF + модель, один поток, после 20 прогревочных вызовов) на сообщениях test; если в test меньше 1000 сообщений, они повторяются по кругу. Размер — сериализованный joblib-пайплайн в МБ. Машина: macOS-26.6.2-arm64-arm-64bit, 15 логических CPU, Python 3.11.16. Эти числа зависят от машины и нагрузки и не воспроизводятся побайтно (`results/efficiency.csv`).

| Задача | Модель | Train | Обучение, с | Задержка, мс/сообщение | Размер, МБ | Машина |
|---|---|---|---|---|---|---|
| mendeley_3class | LogReg | 4017 | 0.52 | 0.413 | 1.78 | local_mac_m |
| mendeley_3class | LinearSVM | 4017 | 3.16 | 3.587 | 2.26 | kaggle_cpu_xeon_2.2ghz |
| mendeley_3class | RandomForest | 4017 | 0.32 | 1.773 | 5.45 | local_mac_m |
| mendeley_3class | XGBoost | 4017 | 16.18 | 0.293 | 2.86 | local_mac_m |
| mendeley_binary | LogReg | 4017 | 0.28 | 0.207 | 0.69 | local_mac_m |
| mendeley_binary | LinearSVM | 4017 | 2.75 | 1.966 | 3.85 | kaggle_cpu_xeon_2.2ghz |
| mendeley_binary | RandomForest | 4017 | 0.68 | 9.390 | 17.67 | local_mac_m |
| mendeley_binary | XGBoost | 4017 | 11.14 | 0.254 | 1.46 | local_mac_m |
| moz_binary | LogReg | 1765 | 0.14 | 0.212 | 0.69 | local_mac_m |
| moz_binary | LinearSVM | 1765 | 1.31 | 2.058 | 1.26 | kaggle_cpu_xeon_2.2ghz |
| moz_binary | RandomForest | 1765 | 0.32 | 6.651 | 7.42 | local_mac_m |
| moz_binary | XGBoost | 1765 | 0.86 | 0.492 | 3.33 | local_mac_m |
| super_sms_clean_binary | LogReg | 42857 | 3.48 | 0.441 | 1.45 | local_mac_m |
| super_sms_clean_binary | LinearSVM | 42857 | 22.22 | 3.504 | 4.06 | kaggle_cpu_xeon_2.2ghz |
| super_sms_clean_binary | RandomForest | 42857 | 3.46 | 3.864 | 41.70 | local_mac_m |
| super_sms_clean_binary | XGBoost | 42857 | 78.67 | 0.278 | 2.41 | local_mac_m |

LinearSVM после правки (`ensemble=False`) обучался и замерялся на Kaggle CPU, остальные модели — локально, поэтому эти числа между собой не сравнимы. Единый замер всех моделей шагов 2 и 3 на одном железе — в отчёте шага 3.

## 5. Наблюдения

- **mendeley_3class**: лучшая модель по среднему macro F1 — LogReg (0.9294 ± 0.0021).
- mendeley_3class: классы, где средний по seed recall хотя бы одной модели ниже 0.90 (в скобках recall всех моделей): `smishing` (LogReg 0.919, LinearSVM 0.907, RandomForest 0.885, XGBoost 0.899); `spam` (LogReg 0.874, LinearSVM 0.874, RandomForest 0.771, XGBoost 0.764).
- mendeley_3class: macro F1 на val (лучшая попытка) → test (среднее): LogReg 0.9207 → 0.9294; LinearSVM 0.9167 → 0.9259; RandomForest 0.9196 → 0.9019; XGBoost 0.8958 → 0.9001.
- mendeley_3class: наибольший ECE у RandomForest (0.0903), наименьший у LinearSVM (0.0187).
- mendeley_3class: после поправки Холма значимых различий с лучшей моделью: бутстреп 2 из 3, Макнемар 1 из 3.
- **mendeley_binary**: лучшая модель по среднему macro F1 — LinearSVM (0.9883 ± 0.0007).
- mendeley_binary: recall при FPR = 1% от 0.9158 (XGBoost) до 0.9847 (RandomForest).
- mendeley_binary: macro F1 на val (лучшая попытка) → test (среднее): LogReg 0.9801 → 0.9788; LinearSVM 0.9869 → 0.9883; RandomForest 0.9626 → 0.9756; XGBoost 0.9694 → 0.9591.
- mendeley_binary: стандартное отклонение macro F1 по seed равно 0 у: LogReg.
- mendeley_binary: наибольший ECE у RandomForest (0.0933), наименьший у LinearSVM (0.0063).
- mendeley_binary: после поправки Холма значимых различий с лучшей моделью: бутстреп 3 из 3, Макнемар 3 из 3.
- **moz_binary**: лучшая модель по среднему macro F1 — RandomForest (0.9825 ± 0.0000).
- moz_binary: recall при FPR = 1% от 0.7701 (XGBoost) до 0.9439 (LinearSVM).
- moz_binary: macro F1 на val (лучшая попытка) → test (среднее): LogReg 0.9704 → 0.9769; LinearSVM 0.9768 → 0.9774; RandomForest 0.9700 → 0.9825; XGBoost 0.9704 → 0.9745.
- moz_binary: стандартное отклонение macro F1 по seed равно 0 у: LogReg, RandomForest.
- moz_binary: наибольший ECE у RandomForest (0.0525), наименьший у LinearSVM (0.0099).
- moz_binary: после поправки Холма значимых различий с лучшей моделью: бутстреп 0 из 3, Макнемар 0 из 3.
- **super_sms_clean_binary**: лучшая модель по среднему macro F1 — LinearSVM (0.9927 ± 0.0000).
- super_sms_clean_binary: recall при FPR = 1% от 0.9836 (RandomForest) до 0.9941 (LinearSVM).
- super_sms_clean_binary: macro F1 на val (лучшая попытка) → test (среднее): LogReg 0.9908 → 0.9912; LinearSVM 0.9920 → 0.9927; RandomForest 0.9871 → 0.9861; XGBoost 0.9900 → 0.9895.
- super_sms_clean_binary: стандартное отклонение macro F1 по seed равно 0 у: LogReg.
- super_sms_clean_binary: наибольший ECE у RandomForest (0.1451), наименьший у LinearSVM (0.0014).
- super_sms_clean_binary: после поправки Холма значимых различий с лучшей моделью: бутстреп 3 из 3, Макнемар 3 из 3.

## 6. Проблемы и ограничения

- LinearSVC не выдаёт вероятностей, поэтому он обёрнут в `CalibratedClassifierCV(method='sigmoid', cv=5, ensemble=False)`: сигмоида обучается на out-of-fold решениях 5 фолдов train, итоговая модель — одна SVM на всём train. Предсказание — argmax калиброванных вероятностей; PR-AUC и ECE относятся к калиброванной модели. LinearSVM обучен на Kaggle CPU (`smishfuse-svm-cpu`), остальные модели шага 2 — локально.
- LogReg (lbfgs) не содержит случайности: seed влияет только на порядок train, поэтому разброс по seed у неё нулевой или почти нулевой. Это не 5 независимых моделей.
- Пространство признаков помимо двух заданных вариантов (слова 1–2, символы 3–5) включает их объединение `word+char` как третий вариант (см. `configs/step2.yaml`).
- 20 попыток TPE — грубый поиск: первые 10 попыток TPE в Optuna по умолчанию случайные. Разница между моделями частично может объясняться бюджетом подбора.
- Значимость проверяется только на предсказаниях одного seed (seed 1), как требует протокол. Разброс между seed в тесты не входит.
- Размеры test по классам: mendeley_3class: ham 965, smishing 99, spam 84; mendeley_binary: benign 965, malicious 183; moz_binary: benign 398, malicious 107; super_sms_clean_binary: benign 7349, malicious 4896. На Mendeley и MOZ редкие классы насчитывают около сотни сообщений, поэтому доверительные интервалы там широкие, а разница в 1–3 ошибки заметно двигает метрики.
- RandomForest обучается в 8 потоков, а предсказывает в одном: параллельное суммирование вероятностей деревьев в sklearn меняет последние биты результата от запуска к запуску.
- Задержка измерена в одном потоке на этой машине; при другой нагрузке или железе числа будут другими. У moz_binary в test 505 сообщений, поэтому для 1000 измерений сообщения повторяются.
- XGBoost на macOS требует OpenMP-рантайм (`brew install libomp`).
