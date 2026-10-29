# SmishFuse, шаг 3: нейросетевые baseline

Сгенерировано скриптом `scripts/make_step3_report.py` из `results/`. Все нейросетевые вычисления выполнены на Kaggle (2× Tesla T4 на сессию), локально — только код, отправка и сбор результатов. Метрики всех моделей пересчитаны одним кодом (`src/eval/metrics.py`) из сырых предсказаний `results/raw/`.

## Протокол

- Модели: TextCNN и BiLSTM (словные эмбеддинги с нуля), DistilBERT и BERT (`distilbert-base-uncased`, `bert-base-uncased`; для moz_binary — `distilbert-base-multilingual-cased`, `bert-base-multilingual-cased`), Qwen2.5-7B-Instruct в 4-bit, 8 примеров из train без дообучения.
- Подбор: Optuna TPE (seed 42), 20 попыток на пару модель-задача, критерий macro F1 на val, early stopping по macro F1 на val. Пространства поиска: `configs/step3.yaml`.
- Финальная оценка: seed 1..5 с лучшими параметрами, test только для финальной оценки. fp16 (autocast + GradScaler), `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, фиксированные seed; побайтная воспроизводимость не требуется.
- LLM: промпт фиксирован (`configs/llm_prompt.yaml`), подбора нет; 5 seed = 5 разных наборов из 8 примеров, сбалансированных по классам. Ответ вне допустимых слов — ошибка (`invalid`). Вероятности для PR-AUC и ECE — softmax по логитам первого токена допустимых слов.
- Устойчивость к обрывам: каждая попытка и каждый seed сохраняются сразу; воркер не начинает новую работу после 8.5 ч сессии; повторный запуск продолжает с датасета `smishfuse-state`.

## 1. Итоговые таблицы (test, среднее ± std по seed; все модели шагов 2 и 3)

Жирным — лучшее, подчёркнуто — второе (для ECE лучше меньшее). LaTeX: `results/tables/<task>.tex`.

### mendeley_3class

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | ECE | Recall ham | Recall smishing | Recall spam |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.9181 ± 0.0012 | 0.9767 ± 0.0004 | 0.9760 ± 0.0004 | 0.9021 ± 0.0018 | <u>0.9616 ± 0.0000</u> | 0.0136 ± 0.0004 | **0.9990 ± 0.0000** | 0.9192 ± 0.0000 | 0.7881 ± 0.0053 |
| TF-IDF + LinearSVM | **0.9259 ± 0.0019** | **0.9791 ± 0.0006** | **0.9791 ± 0.0006** | **0.9255 ± 0.0017** | **0.9659 ± 0.0002** | 0.0187 ± 0.0011 | 0.9956 ± 0.0005 | 0.9071 ± 0.0045 | **0.8738 ± 0.0065** |
| TF-IDF + RandomForest | 0.9130 ± 0.0045 | 0.9732 ± 0.0013 | 0.9723 ± 0.0013 | 0.8902 ± 0.0032 | 0.9606 ± 0.0009 | 0.0875 ± 0.0011 | <u>0.9985 ± 0.0009</u> | 0.8768 ± 0.0085 | 0.7952 ± 0.0053 |
| TF-IDF + XGBoost | 0.8991 ± 0.0058 | 0.9667 ± 0.0019 | 0.9657 ± 0.0020 | 0.8768 ± 0.0072 | 0.9594 ± 0.0015 | **0.0110 ± 0.0019** | 0.9942 ± 0.0014 | 0.8646 ± 0.0115 | 0.7714 ± 0.0177 |
| TextCNN | 0.9006 ± 0.0129 | 0.9706 ± 0.0043 | 0.9693 ± 0.0049 | 0.8780 ± 0.0195 | 0.9424 ± 0.0071 | 0.0183 ± 0.0104 | 0.9983 ± 0.0006 | 0.8929 ± 0.0233 | 0.7429 ± 0.0562 |
| BiLSTM | 0.9104 ± 0.0106 | 0.9742 ± 0.0029 | 0.9742 ± 0.0029 | 0.9105 ± 0.0134 | 0.9413 ± 0.0063 | <u>0.0130 ± 0.0063</u> | 0.9940 ± 0.0025 | 0.8828 ± 0.0332 | <u>0.8548 ± 0.0258</u> |
| DistilBERT | 0.9103 ± 0.0113 | 0.9756 ± 0.0025 | 0.9755 ± 0.0027 | 0.9095 ± 0.0120 | 0.9509 ± 0.0095 | 0.0171 ± 0.0034 | 0.9954 ± 0.0009 | <u>0.9212 ± 0.0111</u> | 0.8119 ± 0.0371 |
| BERT | <u>0.9195 ± 0.0039</u> | <u>0.9779 ± 0.0005</u> | <u>0.9778 ± 0.0005</u> | <u>0.9201 ± 0.0047</u> | 0.9598 ± 0.0055 | 0.0179 ± 0.0013 | 0.9952 ± 0.0012 | **0.9293 ± 0.0071** | 0.8357 ± 0.0155 |
| Qwen2.5-7B 8-shot | 0.8202 ± 0.0205 | 0.9455 ± 0.0061 | 0.9465 ± 0.0052 | 0.8356 ± 0.0187 | 0.8777 ± 0.0155 | 0.0407 ± 0.0067 | 0.9793 ± 0.0045 | 0.8061 ± 0.1268 | 0.7214 ± 0.1233 |

### mendeley_binary

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | Recall@FPR=1% | ECE | Recall malicious |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.9788 ± 0.0000 | 0.9887 ± 0.0000 | 0.9887 ± 0.0000 | 0.9778 ± 0.0000 | <u>0.9955 ± 0.0000</u> | **0.9836 ± 0.0000** | 0.0165 ± 0.0000 | 0.9617 ± 0.0000 |
| TF-IDF + LinearSVM | **0.9883 ± 0.0007** | **0.9937 ± 0.0004** | **0.9937 ± 0.0004** | **0.9896 ± 0.0002** | 0.9954 ± 0.0000 | **0.9836 ± 0.0000** | **0.0063 ± 0.0006** | **0.9836 ± 0.0000** |
| TF-IDF + RandomForest | 0.9720 ± 0.0031 | 0.9852 ± 0.0016 | 0.9851 ± 0.0016 | 0.9664 ± 0.0038 | 0.9914 ± 0.0010 | <u>0.9781 ± 0.0067</u> | 0.0878 ± 0.0007 | 0.9388 ± 0.0071 |
| TF-IDF + XGBoost | 0.9582 ± 0.0019 | 0.9777 ± 0.0010 | 0.9776 ± 0.0010 | 0.9557 ± 0.0026 | 0.9850 ± 0.0002 | 0.9202 ± 0.0049 | 0.0212 ± 0.0018 | 0.9235 ± 0.0055 |
| TextCNN | 0.9721 ± 0.0031 | 0.9852 ± 0.0017 | 0.9851 ± 0.0017 | 0.9673 ± 0.0066 | 0.9892 ± 0.0017 | 0.9530 ± 0.0062 | 0.0088 ± 0.0053 | 0.9410 ± 0.0166 |
| BiLSTM | 0.9818 ± 0.0025 | 0.9902 ± 0.0014 | 0.9902 ± 0.0014 | 0.9814 ± 0.0024 | 0.9869 ± 0.0028 | 0.9760 ± 0.0083 | 0.0070 ± 0.0017 | 0.9683 ± 0.0071 |
| DistilBERT | 0.9810 ± 0.0055 | 0.9897 ± 0.0030 | 0.9898 ± 0.0030 | 0.9841 ± 0.0050 | **0.9965 ± 0.0018** | **0.9836 ± 0.0077** | <u>0.0064 ± 0.0027</u> | <u>0.9760 ± 0.0099</u> |
| BERT | <u>0.9844 ± 0.0037</u> | <u>0.9916 ± 0.0020</u> | <u>0.9916 ± 0.0020</u> | <u>0.9853 ± 0.0039</u> | 0.9951 ± 0.0033 | **0.9836 ± 0.0102** | <u>0.0064 ± 0.0010</u> | <u>0.9760 ± 0.0083</u> |
| Qwen2.5-7B 8-shot | 0.9276 ± 0.0051 | 0.9618 ± 0.0033 | 0.9615 ± 0.0028 | 0.9175 ± 0.0295 | 0.9581 ± 0.0140 | 0.8525 ± 0.0631 | 0.0261 ± 0.0025 | 0.8525 ± 0.0755 |

### moz_binary

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | Recall@FPR=1% | ECE | Recall malicious |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.9797 ± 0.0000 | 0.9861 ± 0.0000 | 0.9863 ± 0.0000 | **0.9912 ± 0.0000** | **0.9811 ± 0.0000** | 0.8785 ± 0.0000 | 0.0136 ± 0.0000 | **1.0000 ± 0.0000** |
| TF-IDF + LinearSVM | 0.9774 ± 0.0013 | 0.9846 ± 0.0009 | 0.9847 ± 0.0009 | 0.9895 ± 0.0018 | <u>0.9793 ± 0.0000</u> | **0.9439 ± 0.0000** | 0.0099 ± 0.0018 | <u>0.9981 ± 0.0042</u> |
| TF-IDF + RandomForest | **0.9825 ± 0.0000** | **0.9881 ± 0.0000** | **0.9882 ± 0.0000** | 0.9890 ± 0.0000 | 0.9782 ± 0.0017 | <u>0.9346 ± 0.0229</u> | 0.0525 ± 0.0009 | 0.9907 ± 0.0000 |
| TF-IDF + XGBoost | 0.9762 ± 0.0024 | 0.9838 ± 0.0017 | 0.9839 ± 0.0016 | 0.9870 ± 0.0025 | 0.9700 ± 0.0027 | 0.8617 ± 0.0656 | <u>0.0086 ± 0.0028</u> | 0.9925 ± 0.0042 |
| TextCNN | 0.9677 ± 0.0042 | 0.9778 ± 0.0029 | 0.9781 ± 0.0029 | 0.9811 ± 0.0047 | 0.9700 ± 0.0065 | 0.8579 ± 0.1012 | 0.0172 ± 0.0019 | 0.9869 ± 0.0107 |
| BiLSTM | 0.9594 ± 0.0073 | 0.9727 ± 0.0049 | 0.9728 ± 0.0049 | 0.9628 ± 0.0105 | 0.9631 ± 0.0100 | 0.8579 ± 0.0440 | 0.0235 ± 0.0155 | 0.9458 ± 0.0233 |
| DistilBERT | 0.9809 ± 0.0069 | 0.9869 ± 0.0048 | 0.9871 ± 0.0047 | <u>0.9910 ± 0.0043</u> | 0.9522 ± 0.0053 | 0.7925 ± 0.1748 | **0.0072 ± 0.0017** | <u>0.9981 ± 0.0042</u> |
| BERT | <u>0.9819 ± 0.0024</u> | <u>0.9877 ± 0.0017</u> | <u>0.9878 ± 0.0016</u> | 0.9902 ± 0.0023 | 0.9736 ± 0.0108 | 0.9308 ± 0.0341 | 0.0103 ± 0.0024 | 0.9944 ± 0.0051 |
| Qwen2.5-7B 8-shot | 0.7954 ± 0.0341 | 0.8293 ± 0.0345 | 0.8432 ± 0.0304 | 0.8876 ± 0.0217 | 0.9193 ± 0.0477 | 0.5477 ± 0.1645 | 0.1385 ± 0.0420 | 0.9888 ± 0.0167 |

### super_sms_clean_binary

| Модель | Macro F1 | Accuracy | Weighted F1 | Balanced acc. | PR-AUC | Recall@FPR=1% | ECE | Recall malicious |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.9909 ± 0.0000 | 0.9913 ± 0.0000 | 0.9913 ± 0.0000 | 0.9902 ± 0.0000 | **0.9991 ± 0.0000** | 0.9933 ± 0.0000 | 0.0047 ± 0.0000 | 0.9847 ± 0.0000 |
| TF-IDF + LinearSVM | 0.9927 ± 0.0000 | 0.9930 ± 0.0000 | 0.9930 ± 0.0000 | 0.9923 ± 0.0000 | **0.9991 ± 0.0000** | 0.9941 ± 0.0000 | **0.0014 ± 0.0001** | 0.9892 ± 0.0001 |
| TF-IDF + RandomForest | 0.9872 ± 0.0003 | 0.9877 ± 0.0003 | 0.9877 ± 0.0003 | 0.9864 ± 0.0003 | 0.9980 ± 0.0001 | 0.9832 ± 0.0008 | 0.0553 ± 0.0005 | 0.9800 ± 0.0006 |
| TF-IDF + XGBoost | 0.9895 ± 0.0003 | 0.9899 ± 0.0003 | 0.9899 ± 0.0003 | 0.9890 ± 0.0003 | 0.9985 ± 0.0001 | 0.9885 ± 0.0004 | **0.0014 ± 0.0002** | 0.9845 ± 0.0007 |
| TextCNN | 0.9891 ± 0.0002 | 0.9895 ± 0.0002 | 0.9895 ± 0.0002 | 0.9891 ± 0.0004 | 0.9987 ± 0.0002 | 0.9877 ± 0.0008 | <u>0.0046 ± 0.0017</u> | 0.9870 ± 0.0018 |
| BiLSTM | 0.9897 ± 0.0005 | 0.9901 ± 0.0005 | 0.9901 ± 0.0005 | 0.9896 ± 0.0004 | 0.9986 ± 0.0002 | 0.9897 ± 0.0009 | 0.0057 ± 0.0022 | 0.9867 ± 0.0020 |
| DistilBERT | <u>0.9929 ± 0.0003</u> | <u>0.9932 ± 0.0003</u> | <u>0.9932 ± 0.0003</u> | <u>0.9928 ± 0.0004</u> | <u>0.9990 ± 0.0002</u> | <u>0.9944 ± 0.0006</u> | 0.0054 ± 0.0012 | <u>0.9910 ± 0.0014</u> |
| BERT | **0.9936 ± 0.0006** | **0.9939 ± 0.0006** | **0.9939 ± 0.0006** | **0.9935 ± 0.0006** | 0.9982 ± 0.0011 | **0.9951 ± 0.0005** | 0.0053 ± 0.0003 | **0.9916 ± 0.0013** |
| Qwen2.5-7B 8-shot | 0.8802 ± 0.0673 | 0.8928 ± 0.0558 | 0.8879 ± 0.0615 | 0.8695 ± 0.0716 | 0.9713 ± 0.0121 | 0.7423 ± 0.1006 | 0.0843 ± 0.0490 | 0.7533 ± 0.1506 |

## 2. Лучшие гиперпараметры нейросетей (подбор на val)

| Задача | Модель | Checkpoint | Параметры | Macro F1 val | Лучшая попытка | Попыток | Macro F1 test |
|---|---|---|---|---|---|---|---|
| mendeley_3class | TextCNN | — | batch_size=128, dropout=0.1787, embed_dim=300, kernel_sizes=3,4,5, lr=0.0003791, max_len=64, min_freq=2, num_filters=128 | 0.9095 | 11 | 20 | 0.9006 |
| mendeley_3class | BiLSTM | — | batch_size=64, dropout=0.2959, embed_dim=300, hidden_size=128, lr=0.0005129, max_len=128, min_freq=2, num_layers=2 | 0.9132 | 16 | 20 | 0.9104 |
| mendeley_3class | DistilBERT | distilbert-base-uncased | batch_size=32, lr=6.188e-05, warmup_ratio=0, weight_decay=0 | 0.9428 | 16 | 20 | 0.9103 |
| mendeley_3class | BERT | bert-base-uncased | batch_size=32, lr=5.915e-05, warmup_ratio=0.06, weight_decay=0.01 | 0.9331 | 19 | 20 | 0.9195 |
| mendeley_binary | TextCNN | — | batch_size=64, dropout=0.1888, embed_dim=300, kernel_sizes=2,3,4, lr=0.0006261, max_len=128, min_freq=1, num_filters=128 | 0.9729 | 7 | 20 | 0.9721 |
| mendeley_binary | BiLSTM | — | batch_size=32, dropout=0.433, embed_dim=100, hidden_size=64, lr=0.0003575, max_len=64, min_freq=2, num_layers=2 | 0.9772 | 0 | 20 | 0.9818 |
| mendeley_binary | DistilBERT | distilbert-base-uncased | batch_size=32, lr=7.773e-05, warmup_ratio=0, weight_decay=0.01 | 0.9935 | 14 | 20 | 0.9810 |
| mendeley_binary | BERT | bert-base-uncased | batch_size=32, lr=8.416e-05, warmup_ratio=0.1, weight_decay=0 | 0.9967 | 12 | 20 | 0.9844 |
| moz_binary | TextCNN | — | batch_size=32, dropout=0.2247, embed_dim=100, kernel_sizes=3,4,5, lr=0.002669, max_len=128, min_freq=2, num_filters=256 | 0.9765 | 2 | 20 | 0.9677 |
| moz_binary | BiLSTM | — | batch_size=128, dropout=0.1206, embed_dim=100, hidden_size=64, lr=0.0005839, max_len=128, min_freq=1, num_layers=1 | 0.9822 | 8 | 20 | 0.9594 |
| moz_binary | DistilBERT | distilbert-base-multilingual-cased | batch_size=16, lr=1.324e-05, warmup_ratio=0, weight_decay=0 | 0.9941 | 5 | 20 | 0.9809 |
| moz_binary | BERT | bert-base-multilingual-cased | batch_size=16, lr=3.856e-05, warmup_ratio=0.1, weight_decay=0.01 | 0.9882 | 12 | 20 | 0.9819 |
| super_sms_clean_binary | TextCNN | — | batch_size=32, dropout=0.1602, embed_dim=200, kernel_sizes=2,3,4, lr=0.0001028, max_len=64, min_freq=2, num_filters=128 | 0.9906 | 10 | 20 | 0.9891 |
| super_sms_clean_binary | BiLSTM | — | batch_size=64, dropout=0.4266, embed_dim=200, hidden_size=64, lr=0.001777, max_len=64, min_freq=2, num_layers=2 | 0.9917 | 14 | 20 | 0.9897 |
| super_sms_clean_binary | DistilBERT | distilbert-base-uncased | batch_size=32, lr=1.674e-05, warmup_ratio=0, weight_decay=0.01 | 0.9952 | 16 | 20 | 0.9929 |
| super_sms_clean_binary | BERT | bert-base-uncased | batch_size=32, lr=2.966e-05, warmup_ratio=0.06, weight_decay=0 | 0.9952 | 17 | 20 | 0.9936 |

Все попытки: `results/tuning/<task>/<model>/trial_*.json`. Параметры шага 2 — в отчёте шага 2.

## 3. Qwen2.5-7B 8-shot: невалидные ответы и KV-кэш

| Задача | Seed | Доля невалидных (среднее) | Максимум по seed | Macro F1 |
|---|---|---|---|---|
| mendeley_3class | 5 | 0.0000% | 0.0000% | 0.8202 ± 0.0205 |
| mendeley_binary | 5 | 0.0348% | 0.0871% | 0.9276 ± 0.0051 |
| moz_binary | 5 | 0.0000% | 0.0000% | 0.7954 ± 0.0341 |
| super_sms_clean_binary | 5 | 0.0016% | 0.0082% | 0.8802 ± 0.0673 |

Проверка эквивалентности префиксного KV-кэша (system + 8 примеров считаются один раз на seed): 200 случайных сообщений val (seed 0), примеры seed 1, ответы без кэша (батч 8) и с кэшем (батч 32).

| Задача | Сообщений | Совпавших ответов | 100% | max |Δp| | Время, с |
|---|---|---|---|---|---|
| mendeley_3class | 200 | 200 | да | 2.78e-02 | 92 → 37 |
| mendeley_binary | 200 | 200 | да | 3.62e-03 | 96 → 38 |
| moz_binary | 200 | 200 | да | 7.38e-03 | 74 → 34 |
| super_sms_clean_binary | 200 | 200 | да | 1.46e-02 | 88 → 37 |

В финальной оценке кэш используется (`prefix_cache` в `configs/step3.yaml`). Δp — расхождение вероятностей первого токена из-за fp16; на ответы оно не повлияло.

## 4. Значимость (seed 1, лучшая модель задачи против всех остальных, Холм внутри задачи)

| Задача | Лучшая | Против | Δ macro F1 [95% ДИ] | p бутстреп / Холм | Значимо | Макнемар b / c | Метод | p Макнемар / Холм | Значимо |
|---|---|---|---|---|---|---|---|---|---|
| mendeley_3class | LinearSVM | LogReg | +0.0109 [-0.0082; +0.0311] | 0.2616 / 1.0000 | нет | 9 / 5 | exact | 0.4240 / 1.0000 | нет |
| mendeley_3class | LinearSVM | RandomForest | +0.0145 [-0.0092; +0.0396] | 0.2300 / 1.0000 | нет | 12 / 5 | exact | 0.1435 / 0.8608 | нет |
| mendeley_3class | LinearSVM | XGBoost | +0.0290 [-0.0013; +0.0589] | 0.0614 / 0.4298 | нет | 22 / 7 | chi2_cc | 0.0093 / 0.0653 | нет |
| mendeley_3class | LinearSVM | TextCNN | +0.0157 [-0.0075; +0.0406] | 0.1902 / 1.0000 | нет | 12 / 6 | exact | 0.2379 / 1.0000 | нет |
| mendeley_3class | LinearSVM | BiLSTM | +0.0021 [-0.0254; +0.0297] | 0.8849 / 1.0000 | нет | 12 / 10 | exact | 0.8318 / 1.0000 | нет |
| mendeley_3class | LinearSVM | DistilBERT | +0.0118 [-0.0125; +0.0373] | 0.3398 / 1.0000 | нет | 11 / 7 | exact | 0.4807 / 1.0000 | нет |
| mendeley_3class | LinearSVM | BERT | +0.0112 [-0.0167; +0.0399] | 0.4366 / 1.0000 | нет | 13 / 10 | exact | 0.6776 / 1.0000 | нет |
| mendeley_3class | LinearSVM | Qwen2.5-7B 8-shot | +0.0801 [+0.0348; +0.1248] | 2.00e-04 / 0.0016 | да | 48 / 16 | chi2_cc | 1.07e-04 / 8.53e-04 | да |
| mendeley_binary | LinearSVM | LogReg | +0.0098 [+0.0031; +0.0186] | 0.0040 / 0.0200 | да | 6 / 0 | exact | 0.0312 / 0.1562 | нет |
| mendeley_binary | LinearSVM | RandomForest | +0.0167 [+0.0051; +0.0299] | 0.0020 / 0.0120 | да | 12 / 2 | exact | 0.0129 / 0.0776 | нет |
| mendeley_binary | LinearSVM | XGBoost | +0.0313 [+0.0158; +0.0479] | 2.00e-04 / 0.0016 | да | 22 / 3 | chi2_cc | 3.18e-04 / 0.0022 | да |
| mendeley_binary | LinearSVM | TextCNN | +0.0118 [+0.0004; +0.0243] | 0.0296 / 0.1184 | нет | 10 / 3 | exact | 0.0923 / 0.3691 | нет |
| mendeley_binary | LinearSVM | BiLSTM | +0.0034 [-0.0064; +0.0137] | 0.4250 / 0.8499 | нет | 6 / 4 | exact | 0.7539 / 1.0000 | нет |
| mendeley_binary | LinearSVM | DistilBERT | +0.0078 [-0.0032; +0.0193] | 0.2054 / 0.6161 | нет | 9 / 4 | exact | 0.2668 / 0.8005 | нет |
| mendeley_binary | LinearSVM | BERT | +0.0001 [-0.0089; +0.0091] | 0.9591 / 0.9591 | нет | 4 / 4 | exact | 1.0000 / 1.0000 | нет |
| mendeley_binary | LinearSVM | Qwen2.5-7B 8-shot | +0.0673 [+0.0452; +0.0905] | 2.00e-04 / 0.0016 | да | 45 / 5 | chi2_cc | 3.48e-08 / 2.78e-07 | да |
| moz_binary | RandomForest | LogReg | +0.0027 [-0.0064; +0.0130] | 0.8055 / 1.0000 | нет | 2 / 1 | exact | 1.0000 / 1.0000 | нет |
| moz_binary | RandomForest | LinearSVM | +0.0027 [-0.0093; +0.0154] | 0.8375 / 1.0000 | нет | 3 / 2 | exact | 1.0000 / 1.0000 | нет |
| moz_binary | RandomForest | XGBoost | +0.0057 [+0.0000; +0.0148] | 0.2670 / 1.0000 | нет | 2 / 0 | exact | 0.5000 / 1.0000 | нет |
| moz_binary | RandomForest | TextCNN | +0.0085 [+0.0000; +0.0196] | 0.0982 / 0.6873 | нет | 3 / 0 | exact | 0.2500 / 1.0000 | нет |
| moz_binary | RandomForest | BiLSTM | +0.0115 [-0.0033; +0.0279] | 0.1906 / 1.0000 | нет | 6 / 2 | exact | 0.2891 / 1.0000 | нет |
| moz_binary | RandomForest | DistilBERT | -0.0001 [-0.0118; +0.0113] | 0.9335 / 1.0000 | нет | 2 / 2 | exact | 1.0000 / 1.0000 | нет |
| moz_binary | RandomForest | BERT | -0.0030 [-0.0159; +0.0089] | 0.6383 / 1.0000 | нет | 2 / 3 | exact | 1.0000 / 1.0000 | нет |
| moz_binary | RandomForest | Qwen2.5-7B 8-shot | +0.2154 [+0.1777; +0.2546] | 2.00e-04 / 0.0016 | да | 96 / 1 | chi2_cc | 1.37e-21 / 1.10e-20 | да |
| super_sms_clean_binary | BERT | LogReg | +0.0030 [+0.0015; +0.0045] | 2.00e-04 / 0.0016 | да | 58 / 23 | chi2_cc | 1.58e-04 / 4.75e-04 | да |
| super_sms_clean_binary | BERT | LinearSVM | +0.0012 [-0.0002; +0.0026] | 0.0964 / 0.1928 | нет | 42 / 28 | chi2_cc | 0.1202 / 0.2405 | нет |
| super_sms_clean_binary | BERT | RandomForest | +0.0066 [+0.0048; +0.0084] | 2.00e-04 / 0.0016 | да | 101 / 24 | chi2_cc | 1.06e-11 / 7.44e-11 | да |
| super_sms_clean_binary | BERT | XGBoost | +0.0046 [+0.0030; +0.0063] | 2.00e-04 / 0.0016 | да | 80 / 26 | chi2_cc | 2.64e-07 / 1.32e-06 | да |
| super_sms_clean_binary | BERT | TextCNN | +0.0050 [+0.0033; +0.0067] | 2.00e-04 / 0.0016 | да | 83 / 24 | chi2_cc | 2.06e-08 / 1.23e-07 | да |
| super_sms_clean_binary | BERT | BiLSTM | +0.0042 [+0.0026; +0.0058] | 2.00e-04 / 0.0016 | да | 71 / 22 | chi2_cc | 6.45e-07 / 2.58e-06 | да |
| super_sms_clean_binary | BERT | DistilBERT | +0.0008 [-0.0003; +0.0020] | 0.1500 / 0.1928 | нет | 27 / 17 | chi2_cc | 0.1748 / 0.2405 | нет |
| super_sms_clean_binary | BERT | Qwen2.5-7B 8-shot | +0.0459 [+0.0419; +0.0501] | 2.00e-04 / 0.0016 | да | 572 / 37 | chi2_cc | 7.75e-104 / 6.20e-103 | да |

## 5. Время Kaggle по ноутбукам

Длительность сессии по `results/kaggle_runs/<kernel>/results/runtime/`. По наблюдению после smoke-теста квота GPU списывается по времени сессии (2× T4 = одна сессия).

| Ноутбук | Сессия | Часы | Тип | Коды возврата воркеров |
|---|---|---|---|---|
| smishfuse-classic-cpu | 1 | 4.80 | CPU | 0 |
| smishfuse-full-bert-distil | 1 | 8.04 | GPU | 0, 0 |
| smishfuse-full-c2 | 1 | 0.26 | GPU | 0, 0 |
| smishfuse-full-llm-bert | 1 | 2.60 | GPU | 1, 1 |
| smishfuse-full-rnn-cnn | 1 | 0.83 | GPU | 0, 0 |
| smishfuse-latency | 1 | 1.11 | GPU | 0 |
| smishfuse-smoke | 1 | 0.13 | GPU | 0, 0 |
| smishfuse-smoke2 | 1 | 0.05 | GPU | 0, 0 |
| smishfuse-smoke2 | 2 | 0.23 | GPU | 0, 0 |
| smishfuse-svm-cpu | 1 | 0.19 | CPU | 0 |
| **всего GPU** |  | **13.25** |  |  |

## 6. Задержка и размер моделей (единый замер)

Одна сессия Kaggle. CPU-модели: Intel(R) Xeon(R) CPU @ 2.00GHz, 1 поток, батч 1, медиана по 1000 сообщениям test (по кругу, если их меньше), 20 прогревочных вызовов, время включает предобработку. Qwen — на GPU, помечено отдельно.

| Задача | Модель | Устройство | Задержка, мс/сообщение | Размер, МБ | Источник |
|---|---|---|---|---|---|
| mendeley_3class | LogReg | cpu_1_thread | 1.926 | 1.8 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| mendeley_3class | LinearSVM | cpu_1_thread | 2.817 | 2.3 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| mendeley_3class | RandomForest | cpu_1_thread | 8.140 | 5.5 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| mendeley_3class | XGBoost | cpu_1_thread | 1.538 | 2.9 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| mendeley_3class | TextCNN | cpu_1_thread | 0.835 | 6.1 | веса seed 1 из `smishfuse-full-rnn-cnn` |
| mendeley_3class | BiLSTM | cpu_1_thread | 3.337 | 7.6 | веса seed 1 из `smishfuse-full-rnn-cnn` |
| mendeley_3class | DistilBERT | cpu_1_thread | 57.466 | 268.5 | веса seed 1 из `smishfuse-full-bert-distil` |
| mendeley_3class | BERT | cpu_1_thread | 112.348 | 438.7 | переобучен на GPU в этой сессии (seed 1, лучшие параметры) |
| mendeley_3class | Qwen2.5-7B 8-shot | gpu:Tesla T4 | 178.104 | 5443.3 | Qwen/Qwen2.5-7B-Instruct 4-bit, prefix_cache=True |
| mendeley_binary | LogReg | cpu_1_thread | 0.912 | 0.7 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| mendeley_binary | LinearSVM | cpu_1_thread | 1.756 | 3.9 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| mendeley_binary | RandomForest | cpu_1_thread | 53.095 | 17.5 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| mendeley_binary | XGBoost | cpu_1_thread | 1.367 | 1.5 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| mendeley_binary | TextCNN | cpu_1_thread | 0.755 | 10.9 | веса seed 1 из `smishfuse-full-rnn-cnn` |
| mendeley_binary | BiLSTM | cpu_1_thread | 2.637 | 2.2 | веса seed 1 из `smishfuse-full-rnn-cnn` |
| mendeley_binary | DistilBERT | cpu_1_thread | 60.346 | 268.5 | веса seed 1 из `smishfuse-full-bert-distil` |
| mendeley_binary | BERT | cpu_1_thread | 113.118 | 438.7 | переобучен на GPU в этой сессии (seed 1, лучшие параметры) |
| mendeley_binary | Qwen2.5-7B 8-shot | gpu:Tesla T4 | 241.072 | 5443.3 | Qwen/Qwen2.5-7B-Instruct 4-bit, prefix_cache=True |
| moz_binary | LogReg | cpu_1_thread | 1.012 | 0.7 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| moz_binary | LinearSVM | cpu_1_thread | 1.728 | 1.3 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| moz_binary | RandomForest | cpu_1_thread | 38.388 | 7.4 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| moz_binary | XGBoost | cpu_1_thread | 2.590 | 3.3 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| moz_binary | TextCNN | cpu_1_thread | 0.739 | 1.9 | веса seed 1 из `smishfuse-full-rnn-cnn` |
| moz_binary | BiLSTM | cpu_1_thread | 1.821 | 2.5 | веса seed 1 из `smishfuse-full-rnn-cnn` |
| moz_binary | DistilBERT | cpu_1_thread | 75.487 | 544.2 | веса seed 1 из `smishfuse-full-bert-distil` |
| moz_binary | BERT | cpu_1_thread | 127.374 | 714.4 | переобучен на GPU в этой сессии (seed 1, лучшие параметры) |
| moz_binary | Qwen2.5-7B 8-shot | gpu:Tesla T4 | 262.400 | 5443.3 | Qwen/Qwen2.5-7B-Instruct 4-bit, prefix_cache=True |
| super_sms_clean_binary | LogReg | cpu_1_thread | 2.149 | 1.4 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| super_sms_clean_binary | LinearSVM | cpu_1_thread | 2.899 | 4.1 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| super_sms_clean_binary | RandomForest | cpu_1_thread | 20.941 | 41.6 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| super_sms_clean_binary | XGBoost | cpu_1_thread | 1.432 | 2.4 | заново обучен в этой сессии (seed 1, лучшие параметры) |
| super_sms_clean_binary | TextCNN | cpu_1_thread | 0.629 | 16.2 | веса seed 1 из `smishfuse-full-rnn-cnn` |
| super_sms_clean_binary | BiLSTM | cpu_1_thread | 2.821 | 16.2 | веса seed 1 из `smishfuse-full-rnn-cnn` |
| super_sms_clean_binary | DistilBERT | cpu_1_thread | 59.611 | 268.5 | веса seed 1 из `smishfuse-full-bert-distil` |
| super_sms_clean_binary | BERT | cpu_1_thread | 116.581 | 438.7 | веса seed 1 из `smishfuse-full-bert-distil` |
| super_sms_clean_binary | Qwen2.5-7B 8-shot | gpu:Tesla T4 | 253.143 | 5443.3 | Qwen/Qwen2.5-7B-Instruct 4-bit, prefix_cache=True |

## 7. Наблюдения

- **mendeley_3class**: лучшая модель — LinearSVM (0.9259); лучшая нейросеть BERT 0.9195, лучший классический LinearSVM 0.9259.
- mendeley_3class: recall класса `smishing` ниже 0.90 у RandomForest 0.877, XGBoost 0.865, TextCNN 0.893, BiLSTM 0.883, Qwen2.5-7B 8-shot 0.806.
- mendeley_3class: recall класса `spam` ниже 0.90 у LogReg 0.788, LinearSVM 0.874, RandomForest 0.795, XGBoost 0.771, TextCNN 0.743, BiLSTM 0.855, DistilBERT 0.812, BERT 0.836, Qwen2.5-7B 8-shot 0.721.
- **mendeley_binary**: лучшая модель — LinearSVM (0.9883); лучшая нейросеть BERT 0.9844, лучший классический LinearSVM 0.9883.
- mendeley_binary: recall класса `malicious` ниже 0.90 у Qwen2.5-7B 8-shot 0.852.
- **moz_binary**: лучшая модель — RandomForest (0.9825); лучшая нейросеть BERT 0.9819, лучший классический RandomForest 0.9825.
- moz_binary: recall класса `benign` ниже 0.90 у Qwen2.5-7B 8-shot 0.786.
- **super_sms_clean_binary**: лучшая модель — BERT (0.9936); лучшая нейросеть BERT 0.9936, лучший классический LinearSVM 0.9927.
- super_sms_clean_binary: recall класса `malicious` ниже 0.90 у Qwen2.5-7B 8-shot 0.753.

## 8. Проблемы

- Ошибка в `mendeley_3class__qwen_fewshot` (первая сессия): `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.39 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.68 GiB is free. Including non-PyTorch` — исправлено: после правки все 5 seed получены в продолжении (`results/errors/mendeley_3class__qwen_fewshot.txt`).
- Ошибка в `moz_binary__qwen_fewshot` (первая сессия): `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.00 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.80 GiB is free. Including non-PyTorch` — исправлено: после правки все 5 seed получены в продолжении (`results/errors/moz_binary__qwen_fewshot.txt`).
- Работы, остановленные лимитом сессии и продолженные в следующей: super_sms_clean_binary__bert (1).
- Параллельная работа двух GPU в одной сессии замедляет друг друга: 4 ядра CPU общие (токенизация, коллация, загрузка моделей). В smoke-тесте попытка BERT шла 151–181 с при работающем на втором GPU LLM против 61 с без него.
- Ошибки нехватки памяти Qwen (mendeley_3class, moz_binary) вызывал расчёт префиксного KV-кэша: прямой проход считал логиты словаря для всех позиций префикса во всём батче (4–6 ГБ). Исправлено `logits_to_keep=1` (кэш ключей и значений от этого не меняется); проверка эквивалентности для этих двух задач повторена на исправленном коде (раздел 3).
- Ноутбук `smishfuse-full-llm-bert` завершился со статусом ERROR (из-за этих ошибок), поэтому Kaggle не принимает его как источник для замера задержки: веса BERT для mendeley_3class, mendeley_binary и moz_binary переобучены в сессии замера (seed 1, лучшие параметры), см. раздел 6. На задержку и размер это не влияет (та же архитектура и чекпойнт), но эти веса не побайтно те, что дали метрики seed 1.
- Первая версия ноутбука `smishfuse-latency` (с этим ноутбуком в источниках) была заменена второй; отменить запуск через CLI нельзя, поэтому неизвестно, успела ли первая версия потратить квоту. Её вывод перезаписан, в таблице раздела 5 только сессия второй версии.
- Задержка классических моделей на Kaggle (Xeon 2.0 ГГц) в 4.4–5.8 раза выше прежних локальных замеров (Apple M, LogReg / RandomForest / XGBoost); в отчётах шагов 2 и 3 теперь используется только единый замер, локальные числа остаются в `results/efficiency.csv`. Время обучения моделей шага 2 измерено на разных машинах и в итоговые таблицы не входит (решение 10 в README).
- Детерминированный режим включён с `warn_only=True`: операции без детерминированной реализации на GPU выполняются с предупреждением, поэтому повторный запуск может дать слегка другие метрики.
- Вероятности Qwen — по первому токену допустимых слов; они не учитывают невалидные ответы и не равны вероятности сгенерированного ответа.
- Версии библиотек на Kaggle (Python 3.13, torch 2.11, transformers 5.16, scikit-learn 1.6.1) отличаются от локальных; метрики пересчитаны локально (scikit-learn 1.9.1).

## 9. Итог

Сделано: 5 нейросетевых моделей × 4 задачи, подбор 20 попыток на val, 5 seed на test — полных пар модель-задача 36 из 36 (вместе с шагом 2); общие таблицы и значимость по всем моделям; единый замер задержки и размера; проверка KV-кэша; все вычисления шага 3 — на Kaggle, GPU 13.25 ч.

Требует решения:

- Квота первой версии `smishfuse-latency` (раздел 8) — проверить на странице квоты Kaggle.

Решено после шага 3 (README, решения 9–11):

- Qwen2.5-7B 8-shot остаётся с фиксированным промптом без подбора, по протоколу.
- Веса BERT для замера задержки на mendeley_3class, mendeley_binary и moz_binary переобучены в сессии замера (seed 1, те же параметры); локальные копии этих весов удалены.
- Время обучения убрано из итоговых таблиц; остаются задержка и размер.
- Низкий recall класса `spam` на mendeley_3class разобран в шаге 4а (`reports/step4a_report.md`).
- Веса моделей (seed 1) хранятся только в выводе ноутбуков Kaggle, в git их нет.
