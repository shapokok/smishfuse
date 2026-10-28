# SmishFuse, шаг 3: оценка GPU-времени по smoke-тесту

GPU-time estimate for the full step-3 run from the smoke test (results/kaggle_runs/smishfuse-smoke).

Method (stated in the output as well):
  * trainable models: per-sample training and validation time per epoch is taken
    from the super_sms_clean_binary smoke trial (the largest smoke sample, 2143
    train messages; the smaller tasks are dominated by fixed overhead); the fixed
    per-fit overhead (model init / download from cache) = trial time minus epochs and
    encoding. One fit = overhead + epochs * (n_train * t_train + n_val * t_val).
    Epoch counts: min = patience + 1, typical = epochs of the super smoke trial,
    max = max_epochs. 25 fits per (task, model): 20 trials + 5 seeds.
  * LLM: seconds per test message measured per task in the smoke run, times the
    full test size, times 5 seeds, plus one model load per worker.

| Задача | Модель | Train | batch (trial 0) | Эпоха, с | Накладные на fit, с | Часы: min | типично | max |
|---|---|---|---|---|---|---|---|---|
| mendeley_3class | textcnn | 4017.0 | 32.0 | 0.8 | 0.1 | 0.02 | 0.08 | 0.11 |
| mendeley_binary | textcnn | 4017.0 | 32.0 | 0.8 | 0.1 | 0.02 | 0.08 | 0.11 |
| moz_binary | textcnn | 1765.0 | 32.0 | 0.3 | 0.1 | 0.01 | 0.03 | 0.05 |
| super_sms_clean_binary | textcnn | 42857.0 | 32.0 | 8.4 | 0.1 | 0.23 | 0.82 | 1.16 |
| mendeley_3class | bilstm | 4017.0 | 32.0 | 4.8 | 0.2 | 0.14 | 0.24 | 0.67 |
| mendeley_binary | bilstm | 4017.0 | 32.0 | 4.8 | 0.2 | 0.14 | 0.24 | 0.67 |
| moz_binary | bilstm | 1765.0 | 32.0 | 2.1 | 0.2 | 0.06 | 0.10 | 0.30 |
| super_sms_clean_binary | bilstm | 42857.0 | 32.0 | 51.6 | 0.2 | 1.43 | 2.51 | 7.16 |
| mendeley_3class | distilbert | 4017.0 | 16.0 | 16.1 | 1.1 | 0.34 | 0.57 | 0.57 |
| mendeley_binary | distilbert | 4017.0 | 16.0 | 16.1 | 1.1 | 0.34 | 0.57 | 0.57 |
| moz_binary | distilbert | 1765.0 | 16.0 | 7.1 | 1.1 | 0.16 | 0.25 | 0.25 |
| super_sms_clean_binary | distilbert | 42857.0 | 16.0 | 171.9 | 1.1 | 3.59 | 5.97 | 5.97 |
| mendeley_3class | bert | 4017.0 | 16.0 | 28.6 | 1.4 | 0.61 | 0.80 | 1.00 |
| mendeley_binary | bert | 4017.0 | 16.0 | 28.6 | 1.4 | 0.61 | 0.80 | 1.00 |
| moz_binary | bert | 1765.0 | 16.0 | 12.6 | 1.4 | 0.27 | 0.36 | 0.45 |
| super_sms_clean_binary | bert | 42857.0 | 16.0 | 305.4 | 1.4 | 6.37 | 8.49 | 10.61 |

| Задача | LLM: test | с/сообщение (smoke) | Часы на 5 seed |
|---|---|---|---|
| mendeley_3class | 1148.0 | 0.376 | 0.60 |
| mendeley_binary | 1148.0 | 0.360 | 0.57 |
| moz_binary | 505.0 | 0.395 | 0.28 |
| super_sms_clean_binary | 12245.0 | 0.431 | 7.32 |

Загрузка Qwen (скачивание + 4-bit): 144 с на процесс.

| Модель | Часы GPU: min | типично | max |
|---|---|---|---|
| bert | 7.9 | 10.5 | 13.1 |
| bilstm | 1.8 | 3.1 | 8.8 |
| distilbert | 4.4 | 7.4 | 7.4 |
| qwen_fewshot | 8.8 | 8.8 | 8.8 |
| textcnn | 0.3 | 1.0 | 1.4 |
| **всего (один GPU)** | **23.1** | **30.7** | **39.4** |
