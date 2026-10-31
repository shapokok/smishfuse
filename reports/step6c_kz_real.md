# SmishFuse, шаг 6в: реальные публичные примеры из Казахстана

Сгенерировано `scripts/make_step6c_report.py` из `results/step6c/` (ноутбук `smishfuse-step6c-real`, Kaggle CPU). **Пост-хок (решение 22 в README): после результатов шагов 6 и 6б.** Без обучения новых моделей; **4 сообщения — качественная иллюстрация, метрика (recall) не считается.** Набор зафиксирован коммитом 1596963 до запуска моделей.

## 1. Сообщения

| id | Язык | Статус текста | Текст | Источник |
|---|---|---|---|---|
| real01 | RU | полный текст | Почта Казахстана: ваша посылка прибыла на склад, но не может быть доставлена из-за неверной или неполной информации об адресе доставки. Пожалуйста, подтвердите свой правильный адрес по ссылке ниже, в противном случае посылка будет возвращена через 24 часа | [www.zakon.kz](https://www.zakon.kz/obshestvo/6479122-pereydite-po-ssylke-chtoby-poluchit-posylku-kazakhstantsam-raskryli-kto-im-otpravlyaet-eto-SMS.html), 2025-05-29 |
| real02 | RU | полный текст | Мы отправим товар повторно в течение одного дня после подтверждения адреса. Желаю вам счастливой жизни | [www.zakon.kz](https://www.zakon.kz/obshestvo/6479122-pereydite-po-ssylke-chtoby-poluchit-posylku-kazakhstantsam-raskryli-kto-im-otpravlyaet-eto-SMS.html), 2025-05-29 |
| real03 | KK | расшифровка скриншота, окончание обрезано | [ҚР Жол полициясы департаменті] Бұл жол қозғалысы ережелерін бұзу ісі мәжбүрлеу процесіне жіберілді. Құрметті көлік иесі: Сіздің көлігіңіз Intelligent Traffic Systems (ITS) арқылы жол қозғалысы ережелерін бұзушы ретінде анықталды. Тиісті жол қозғалысы ережелеріне сәйкес, бұл орталық сізге жол қозғалысы ережелерін бұзу туралы алғашқы хабарлама (№: 2026-KZ-3KD71L9M-008) | [www.lada.kz](https://www.lada.kz/kazakhstan-news/153434-kazakhstantsev-predupredili-o-feikovykh-shtrafakh-moshenniki-rassylaiut-sms-ot-imeni-dorozhnoi-politsii.html), 2026-05-23 |
| real04 | RU | фрагмент | Статус вашего займа обновлен | [finance.kz](https://finance.kz/articles/smishing-sms-moshenniki-kak-zashchititsya), 2026-09-21 |

Ссылки в статьях не приводятся, поэтому в текстах их нет. В real03 цифры номера дела заменены случайными; отправитель (номер телефона) не сохранён.

## 2. Вердикты и уверенность

Предсказание (максимальная вероятность). Full: † — ответ дала вторая ступень (DistilBERT). Все сообщения — мошеннические; ✓ — помечено smishing.

| id | Вход | Lite до | Lite адапт. A | Lite адапт. B | Full (каскад) | BERT |
|---|---|---|---|---|---|---|
| real01 | как написано | ham 0.59 | smishing 0.65 ✓ | smishing 0.52 ✓ | spam 0.72 | ham 1.00 |
| real01 | перевод по предложениям | smishing 0.75 ✓ | smishing 0.88 ✓ | smishing 0.86 ✓ | smishing 0.73 ✓ | smishing 0.99 ✓ |
| real02 | как написано | ham 0.97 | smishing 0.43 ✓ | smishing 0.53 ✓ | ham 0.97 | ham 0.99 |
| real02 | перевод по предложениям | ham 0.99 | ham 0.99 | ham 0.99 | ham 0.99 | ham 1.00 |
| real03 | как написано | ham 0.84 | smishing 0.66 ✓ | smishing 0.58 ✓ | ham 0.80 | ham 0.92 |
| real03 | перевод по предложениям | spam 0.67 | spam 0.49 | spam 0.50 | smishing 1.00 † ✓ | smishing 0.99 ✓ |
| real04 | как написано | ham 0.98 | ham 0.73 | ham 0.78 | ham 0.99 | ham 1.00 |
| real04 | перевод по предложениям | ham 0.84 | ham 0.87 | ham 0.89 | ham 0.84 | ham 1.00 |

**Переводы (NLLB по предложениям)**

| id | Перевод |
|---|---|
| real01 | Kazakhstan Post: Your package has arrived at the warehouse, but cannot be delivered due to incorrect or incomplete delivery information. Please confirm your correct address by following the link below, otherwise the package will be returned within 24 hours |
| real02 | We will re-send the goods within one day of confirming the address. I wish you a happy life. |
| real03 | [Department of Road Police of the PRC] This case of violation of traffic rules was sent to a forced process. Dear Car Owner: Your vehicle has been identified as a traffic violation by Intelligent Traffic Systems (ITS). In accordance with the relevant traffic rules, this centre will provide you with the first notification of violation of traffic rules (No: 2026-KZ-3KD71L9M-008) |
| real04 | Your loan status is updated |

Проверка воспроизведения: Lite адаптированный обучен заново тем же кодом и seed, что в шаге 6б (веса там не сохранялись). Macro F1 на Mendeley VAL: A 0.8955 (в 6б 0.8955), B 0.9062 (в 6б 0.9062).

## 3. Ограничения

- Реальных казахстанских SMS с дословным текстом в открытом доступе почти нет: СМИ и ведомства обычно пересказывают сообщения или показывают скриншоты. Из около 50 проверенных публикаций найдено 4 текста, из них один — расшифровка обрезанного скриншота и один — фрагмент.
- Все 4 сообщения мошеннические; оценка ложных тревог на реальных данных не проводилась.
- 4 сообщения — иллюстрация, а не оценка: ни recall, ни сравнение моделей по ним делать нельзя.

## Итог (качественно)

- **Как написано:** адаптированный Lite (A и B) помечает smishing 3 из 4 сообщений (real01–real03), но с низкой уверенностью (0.43–0.66). Lite до адаптации, каскад и BERT не помечают ни одного.
- **После перевода по предложениям:** real01 (посылка со ссылкой) все пять моделей помечают smishing. real03 (штраф ПДД, казахский) — каскад (ответ DistilBERT) и BERT; все варианты Lite — spam.
- **Не распознаётся ни одной моделью:** real02 — второе сообщение рассылки без ссылки и призыва к действию, real04 — фрагмент «Статус вашего займа обновлен». Сами по себе в этих текстах нет признаков обмана.
- **Ошибка перевода:** «ҚР Жол полициясы» NLLB перевёл как «Road Police of the PRC».
