"""Export of Lite for the on-device demo (smishfuse-app): weights as JSON + a golden test, local CPU, seconds.

The weights of Lite are not kept in git (results/final/models/ lived only in the Kaggle kernel outputs), so both
models are re-trained here with exactly the frozen settings and checked against the frozen predictions:
  base     = the frozen final svm_char, seed 1 (src/final/run.py Classic): char TF-IDF 2-5-grams (configs/step4c.yaml
             char_branch), LinearSVC with the step-2 mendeley_3class C / class weight, CalibratedClassifierCV(sigmoid,
             cv 5, ensemble False), train order default_rng(1).permutation. Checked against
             results/final/preds/mendeley_3class/svm_char/val_seed1.csv (clean val; the test split is not read),
             the "before" rows of results/step6b/adapt/predictions.csv (63 case rows) and results/step8/model.json.
  adapted  = step 6b condition B (src/step6/adapt.py): the same model on Mendeley train + all of
             data/kz_adapt/adapt.csv, seed 1; checked against the B_n324_s0 rows of results/step6b/adapt/predictions.csv.
The run stops if a recomputed probability differs from its frozen value by more than CHECK_TOL.

Outputs (models/, not committed; copied into the app repo):
  lite_base.json, lite_adapted.json  classes, vectorizer settings, vocabulary (list: vocabulary[j] = n-gram of column j),
      idf (float32), one entry per calibrated classifier of CalibratedClassifierCV (coef, intercept, Platt a / b),
      tau of the cascade, the cost rule (`cost`: r not chosen, candidates of configs/final.yaml; the app ignores it,
      step 6d / smishfuse-app plan decision 5), provenance and check results.
  golden.json  300 messages with sklearn probabilities of both models. No dataset text is used: author-written KZ case
      rows (data/kz_cases), adaptation messages (data/kz_adapt), the templates below, TRAIN-type (TR) and HELD-OUT (HO)
      obfuscation of src/eval/obfuscation.py at 10% / 20%, and edge cases of the preprocessing (empty text, whitespace
      runs, surrogate pairs, special lower-casing).
On-device inference (reference implementation: reference_proba below, checked against predict_proba):
  t = text.lower(); t = re.sub(r"\\s\\s+", " ", t); n-grams of 2..5 code points; tf = 1 + ln(count) (float32);
  x = tf * idf (float32), L2-normalised; s_k = x . coef_k + intercept_k; p_k = expit(-(a_k s_k + b_k));
  p = p / sum(p) (uniform if the sum is 0); mean over the calibrated classifiers; values in (1, 1 + 1e-5] -> 1.
Run: python scripts/export_lite.py [--out models]
"""
from __future__ import annotations

import os

for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[var] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "baselines"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

import classic  # noqa: E402
from eval.obfuscation import HELDOUT_TYPES, TRAIN_TYPES, obfuscate  # noqa: E402
from neural.common import load_split, load_yaml, read_json  # noqa: E402

CLASSES = ["ham", "smishing", "spam"]
SEED = 1
CHECK_TOL = 1e-6
N_GOLDEN = 300

# ---------------------------------------------------------------- golden templates (written for this export)
TEMPLATES = {
    "en": {
        "ham": [
            "Hey, are we still meeting at 7 tonight?",
            "Mom, I'll be home late, don't wait for dinner.",
            "Your table for 4 at Riverside Cafe is confirmed for Friday 19:30.",
            "Can you send me the slides from today's meeting?",
            "Happy birthday! Hope you have an amazing day :)",
            "The plumber will come tomorrow between 10 and 12.",
            "Your verification code is 482913. Do not share it with anyone.",
            "Running 10 min late, sorry! Order me a coffee pls",
            "Your parcel 7781 was delivered to the pickup point on Abay St 12.",
            "Meeting moved to room 305, starts at 3pm.",
        ],
        "spam": [
            "MEGA SALE! Up to 70% off all sneakers this weekend only. Visit our store now!",
            "Congratulations! You have been selected for a FREE cruise. Call 0800 555 019 to claim.",
            "Get 50 GB of mobile data for just $5/month! Reply YES to activate.",
            "Hot singles in your area are waiting! Text DATE to 69900",
            "Lose 10 kg in 2 weeks with our natural formula! Order today: 1-800-SLIM-NOW",
            "URGENT! Your mobile number has won 2000 GBP in our weekly draw. Txt WIN to 85023 now.",
            "Exclusive offer: refinance your loan at 0% APR. Limited time. Call now!",
            "New pizza place opening! Free dessert with every order this Friday. Show this SMS.",
            "You are a WINNER! Claim your 500 GBP shopping voucher, reply CLAIM. Std txt rate. T&Cs apply",
            "Stop paying too much for insurance! Compare quotes free at bestquotes4u.com",
        ],
        "smishing": [
            "Your bank account has been suspended. Verify your identity at http://secure-bank-verify.com/login immediately.",
            "Your package is on hold due to an unpaid customs fee of $1.99. Pay now: https://parcel-redelivery.top/pay",
            "Unusual sign-in detected. Confirm your details within 24h or your account will be locked: http://id-confirm.info",
            "Your streaming payment failed. Update your billing info at billing-update-now.net to avoid suspension.",
            "We noticed a suspicious transaction of $849.00 on your card. If this wasn't you, call +1 (213) 555-0147 now.",
            "You are eligible for a tax refund of $312.40. Claim it here: http://tax-refund-claim.org/form",
            "Your wallet account is limited. Log in to restore access: https://account-restore.co/verify",
            "Delivery attempt failed. Reschedule and pay 2.50 EUR: http://parcel-track.xyz/r/8812",
            "Dear customer, your card ending 4471 is blocked. Reply with your PIN and CVV to unblock.",
            "Final notice: your electricity will be disconnected today. Pay the overdue bill at http://power-pay-now.com",
        ],
    },
    "ru": {
        "ham": [
            "Привет! Во сколько завтра встречаемся у метро?",
            "Мама, я доехала, всё хорошо. Позвоню вечером.",
            "Ваш код подтверждения: 551902. Никому его не сообщайте.",
            "Напоминаем: приём у стоматолога завтра в 15:30, кабинет 204.",
            "Заказ №48213 доставлен в пункт выдачи на ул. Абая, 15. Срок хранения 5 дней.",
            "Купи, пожалуйста, хлеб и молоко по дороге домой.",
            "Совещание перенесли на пятницу, 11:00.",
            "Поздравляю с днём рождения! Здоровья и счастья!",
        ],
        "spam": [
            "Только до воскресенья! Скидки до 60% на всю обувь в торговом центре. Ждём вас!",
            "Займ до 500 000 тг без справок и поручителей! Одобрение за 5 минут. Звоните 8 800 080 12 34",
            "Акция! Подключите безлимитный интернет за 1990 тг/мес. Ответьте ДА для подключения.",
            "Новая коллекция уже в продаже! Покажите это SMS и получите скидку 15%.",
            "Курсы английского со скидкой 50%! Первое занятие бесплатно. Запись: 8 707 123 45 67",
            "Распродажа техники! Телевизоры от 59 990 тг. Только в эти выходные.",
            "Похудей на 10 кг за месяц! Натуральный продукт, доставка бесплатно.",
            "Вы приглашены на бесплатный мастер-класс по инвестициям! Регистрация по ссылке invest-kurs.kz",
        ],
        "smishing": [
            "Ваша карта заблокирована. Для разблокировки перейдите по ссылке http://bank-unlock24.ru/card и введите данные карты.",
            "Служба безопасности банка: зафиксирована подозрительная операция на 185 000 тг. Сообщите код из SMS оператору.",
            "Поздравляем! Вы выиграли iPhone 15. Для получения оплатите доставку 2500 тг: http://priz-kz.top/get",
            "Ваша посылка задержана на таможне. Оплатите пошлину 990 тг по ссылке http://post-tamozhnya.site/pay",
            "На ваше имя оформлен кредит. Если это не вы, срочно позвоните +7 495 000 12 34.",
            "Ваш аккаунт будет удалён через 24 часа. Подтвердите вход: http://akkaunt-verify.com",
            "Начислена компенсация 45 000 тг. Для получения введите номер карты и CVV на сайте kompensaciya-gov.info",
            "Ваш номер будет отключён сегодня. Для продления продиктуйте код из SMS сотруднику оператора.",
        ],
    },
    "kk": {
        "ham": [
            "Сәлем! Ертең сағат нешеде кездесеміз?",
            "Апа, мен үйге жеттім, бәрі жақсы.",
            "Сіздің растау кодыңыз: 730164. Оны ешкімге айтпаңыз.",
            "Еске саламыз: ертең сағат 10:00-де дәрігердің қабылдауы бар.",
            "Тапсырыс №5521 Абай көшесі, 15 мекенжайындағы бекетке жеткізілді.",
            "Жолда нан мен сүт ала келші.",
            "Жиналыс жұмаға, сағат 11:00-ге ауыстырылды.",
            "Туған күніңізбен құттықтаймын! Денсаулық пен бақыт тілеймін!",
        ],
        "spam": [
            "Жексенбіге дейін ғана! Барлық аяқ киімге 60% дейін жеңілдік. Келіңіз!",
            "Анықтамасыз 500 000 теңгеге дейін несие! 5 минутта мақұлдау. Қоңырау шалыңыз: 8 800 080 12 34",
            "Акция! Шексіз интернетті айына 1990 теңгеге қосыңыз. Қосу үшін ИӘ деп жауап беріңіз.",
            "Жаңа коллекция сатылымда! Осы SMS-ті көрсетіп, 15% жеңілдік алыңыз.",
            "Ағылшын тілі курстары 50% жеңілдікпен! Алғашқы сабақ тегін.",
            "Техника сатылымы! Теледидарлар 59 990 теңгеден бастап. Тек осы демалыста.",
            "Бір айда 10 кг арықтаңыз! Табиғи өнім, жеткізу тегін.",
            "Инвестиция бойынша тегін шеберлік сыныбына шақырамыз! Тіркелу: invest-kurs.kz",
        ],
        "smishing": [
            "Сіздің картаңыз бұғатталды. Бұғаттан шығару үшін http://bank-unlock24.kz/card сілтемесіне өтіп, карта деректерін енгізіңіз.",
            "Банктің қауіпсіздік қызметі: 185 000 теңгеге күдікті операция тіркелді. SMS-тегі кодты операторға айтыңыз.",
            "Құттықтаймыз! Сіз iPhone 15 ұтып алдыңыз. Алу үшін жеткізу ақысын 2500 теңге төлеңіз: http://priz-kz.top/get",
            "Сіздің сәлемдемеңіз кеденде ұсталды. 990 теңге баж салығын төлеңіз: http://post-kz-pay.site/pay",
            "Сіздің атыңызға несие ресімделді. Бұл сіз болмасаңыз, +7 727 000 12 34 нөміріне шұғыл қоңырау шалыңыз.",
            "Аккаунтыңыз 24 сағаттан кейін жойылады. Кіруді растаңыз: http://akkaunt-verify.com",
            "Сізге 45 000 теңге өтемақы есептелді. Алу үшін kompensaciya-gov.info сайтына карта нөмірі мен CVV енгізіңіз.",
            "Нөміріңіз бүгін өшіріледі. Ұзарту үшін SMS-тегі кодты оператор қызметкеріне айтыңыз.",
        ],
    },
}
# per language: template indices (label, index) that also get the TR / HO obfuscated copies
OBF_SOURCES = [("smishing", 0), ("smishing", 2), ("smishing", 4), ("smishing", 6), ("spam", 0), ("spam", 1),
               ("ham", 0), ("ham", 2)]
OBF_LEVELS = [("tr10", TRAIN_TYPES, 0.10), ("tr20", TRAIN_TYPES, 0.20), ("ho10", HELDOUT_TYPES, 0.10),
              ("ho20", HELDOUT_TYPES, 0.20)]
EDGE = [
    ("empty", ""),
    ("one_char", "a"),
    ("two_chars", "OK"),
    ("whitespace_runs", "  Привет \t\t мир \n\n  код   1234  "),
    ("single_whitespace_kept", "Line1\nLine2\rLine3\tend"),
    ("emoji_surrogates", "\U0001F381 Вы выиграли iPhone 15! \U0001F449 http://prize-kz.top/win \U0001F525\U0001F525"),
    ("special_lowercase", "İSTANBUL ΟΔΟΣ Σ ΑΣ. ÆØÅ STRASSE Straße ǅemal"),
    ("unicode_spaces", "Kaspi  Bank　　код  1234 тг \x1c\x1d конец"),
    ("long", " ".join(["Your card is blocked! Verify now at http://secure-bank-verify.com/login or call +7 701 000 00 00."] * 6)),
]
N_ADAPT_PER_LANG_LABEL = 6


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- the frozen model, re-trained

def build_lite():
    """Exactly Classic('svm_char', 'mendeley_3class') of src/final/run.py (without importing its torch stack)."""
    cb = load_yaml("configs/step4c.yaml")["char_branch"]
    vec = TfidfVectorizer(analyzer="char", ngram_range=tuple(cb["ngram_range"]), lowercase=cb["lowercase"],
                          sublinear_tf=cb["sublinear_tf"], min_df=cb["min_df"], dtype=np.float32)
    best = read_json(ROOT / "results" / "best_params" / "mendeley_3class" / "linear_svm.json")
    fixed = load_yaml("configs/step2.yaml")["models"]["linear_svm"]["fixed"]
    model, bw = classic.build_model("linear_svm", best["model_params"], fixed, SEED, 1)
    return classic.TextClassifier(vec, model, bw), best["model_params"], fixed


def fit_lite(texts, y):
    clf, params, fixed = build_lite()
    texts = np.asarray(list(texts), dtype=object)
    perm = np.random.default_rng(SEED).permutation(len(y))  # train order of the final runs
    clf.fit(texts[perm], np.asarray(y)[perm])
    return clf, params, fixed


def check(name: str, got: np.ndarray, want: np.ndarray) -> float:
    diff = float(np.abs(got - want).max())
    same = float((got.argmax(1) == want.argmax(1)).mean())
    log(f"[export] check {name}: max |p - p_frozen| = {diff:.2e}, same class {same:.4f} (n={len(want)})")
    if diff > CHECK_TOL:
        raise SystemExit(f"{name}: re-trained model differs from the frozen predictions ({diff:.2e} > {CHECK_TOL})")
    return diff


def frozen_probs(df: pd.DataFrame, ids) -> np.ndarray:
    df = df.set_index("id").loc[list(ids)]
    return df[[f"prob_{c}" for c in CLASSES]].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------- export and reference inference

def f32(x: float) -> float:
    """float32 value as the shortest decimal that parses back to the same float32."""
    return float(np.format_float_positional(np.float32(x), unique=True, trim="-"))


def cost_rule(cost_ratios: list) -> dict:
    """The cost rule for the app (README decision 24): not chosen yet, the app ignores the field."""
    return {"rule": "alarm (smishing) if p(smishing) >= 1 / (1 + r): the cost rule of the paper (section 4.5) applied to "
                    "p(smishing); the paper states it for p(malicious) = p(smishing) + p(spam)",
            "probability": "smishing",
            "r": None,
            "status": "not chosen: the app uses argmax; r is chosen on the pilot data (smishfuse-app plan, decision 5); "
                      "the known-sender rule of the app uses r = 1; ignored by the app",
            "candidates": [{"r": r, "threshold": 1.0 / (1.0 + r)} for r in cost_ratios]}


def export(clf, name: str, description: str, params: dict, fixed: dict, train_info: dict, checks: dict, tau: float,
           cost: dict):
    vec, cal = clf.vectorizer, clf.model
    if list(cal.classes_) != [0, 1, 2]:
        raise SystemExit(f"unexpected classes {cal.classes_}")
    vocab = [None] * len(vec.vocabulary_)
    for g, j in vec.vocabulary_.items():
        vocab[j] = g
    ccs = []
    for cc in cal.calibrated_classifiers_:
        svm = cc.estimator
        if cc.method != "sigmoid" or list(svm.classes_) != [0, 1, 2] or svm.coef_.shape != (3, len(vocab)):
            raise SystemExit("unexpected calibrated classifier")
        ccs.append({"coef": svm.coef_.tolist(), "intercept": svm.intercept_.tolist(),
                    "platt_a": [float(c.a_) for c in cc.calibrators], "platt_b": [float(c.b_) for c in cc.calibrators]})
    return {
        "format": "smishfuse-lite/1",
        "name": name,
        "description": description,
        "classes": CLASSES,
        "vectorizer": {"analyzer": vec.analyzer, "ngram_range": list(vec.ngram_range), "lowercase": vec.lowercase,
                       "sublinear_tf": vec.sublinear_tf, "norm": vec.norm, "use_idf": vec.use_idf,
                       "smooth_idf": vec.smooth_idf, "min_df": vec.min_df, "max_df": vec.max_df,
                       "max_features": vec.max_features, "strip_accents": vec.strip_accents,
                       "dtype": "float32", "whitespace_regex": r"\s\s+", "n_features": len(vocab)},
        "vocabulary": vocab,
        "idf": [f32(v) for v in vec.idf_],
        "classifier": {"type": "LinearSVC one-vs-rest", "C": params["C"], "class_weight": params["class_weight"],
                       "max_iter": fixed["max_iter"], "random_state": SEED},
        "calibration": {"type": "CalibratedClassifierCV", "method": fixed["calibration_method"],
                        "cv": fixed["calibration_cv"], "ensemble": fixed["calibration_ensemble"],
                        "n_calibrated_classifiers": len(ccs),
                        "rule": "p_k = expit(-(a_k * s_k + b_k)); p /= sum(p) (uniform if 0); mean over classifiers"},
        "calibrated_classifiers": ccs,
        "tau": tau,
        "cost": cost,
        "provenance": {"sklearn": sklearn.__version__, "train": train_info, "checks": checks},
    }


WS = re.compile(r"\s\s+")


def reference_proba(model: dict, text: str) -> tuple[np.ndarray, np.ndarray]:
    """Independent re-implementation from the exported JSON only (the spec of the Kotlin port)."""
    v = model["vectorizer"]
    index = model["_index"]
    t = WS.sub(" ", text.lower() if v["lowercase"] else text)
    counts: dict[int, int] = {}
    lo, hi = v["ngram_range"]
    for n in range(lo, min(hi + 1, len(t) + 1)):
        for i in range(len(t) - n + 1):
            j = index.get(t[i:i + n])
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
    cols = sorted(counts)  # CSR column order of sklearn
    idf = model["_idf"]
    # float32 as in TfidfTransformer (dtype float32): tf = log(count) + 1, tf * idf
    x = [(np.float32(np.log(np.float32(counts[j]))) + np.float32(1.0)) * idf[j] for j in cols]
    # sklearn _inplace_csr_row_normalize_l2: float32 squares summed in double, division in double stored as float32
    sq = 0.0
    for v in x:
        sq += float(v * v)
    if sq != 0.0:
        norm = float(np.sqrt(sq))
        x = [np.float32(float(v) / norm) for v in x]
    proba = np.zeros(3)
    for cc in model["calibrated_classifiers"]:
        coef = cc["_coef"]
        s = np.zeros(3)
        for k in range(3):  # scipy csr @ dense: sequential double accumulation in column order
            acc = 0.0
            for j, v in zip(cols, x):
                acc += float(v) * coef[k, j]
            s[k] = acc
        s = s + np.asarray(cc["intercept"])
        p = expit(-(np.asarray(cc["platt_a"]) * s + np.asarray(cc["platt_b"])))
        tot = p.sum()
        proba += p / tot if tot != 0 else np.full(3, 1 / 3)
    proba /= len(model["calibrated_classifiers"])
    proba[(proba > 1.0) & (proba <= 1.0 + 1e-5)] = 1.0
    return proba, s


def prepare(model: dict) -> dict:
    model["_index"] = {g: j for j, g in enumerate(model["vocabulary"])}
    model["_idf"] = np.asarray(model["idf"], dtype=np.float32)
    for cc in model["calibrated_classifiers"]:
        cc["_coef"] = np.asarray(cc["coef"])
    return model


def golden_messages() -> list[dict]:
    rows = []
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    for _, r in cases.iterrows():
        rows.append({"id": f"case:{r['id']}", "lang": r["language"], "source": "kz_case", "variant": r["variant"],
                     "label": r["label"], "text": r["text"]})
    adapt = pd.read_csv(ROOT / "data" / "kz_adapt" / "adapt.csv", dtype=str, keep_default_na=False)
    for (lang, label), g in adapt.groupby(["language", "label"], sort=True):
        pick = g.sample(n=N_ADAPT_PER_LANG_LABEL, random_state=0).sort_values("id")
        for _, r in pick.iterrows():
            rows.append({"id": f"adapt:{r['id']}", "lang": lang, "source": "kz_adapt", "variant": "original",
                         "label": label, "text": r["text"]})
    for lang, by_label in TEMPLATES.items():
        for label, texts in by_label.items():
            for i, t in enumerate(texts):
                rows.append({"id": f"tpl:{lang}_{label}_{i:02d}", "lang": lang, "source": "template",
                             "variant": "original", "label": label, "text": t})
    for lang in TEMPLATES:
        for label, i in OBF_SOURCES:
            src = f"tpl:{lang}_{label}_{i:02d}"
            for variant, types, level in OBF_LEVELS:
                text, _ = obfuscate(TEMPLATES[lang][label][i], src, level, types)
                rows.append({"id": f"{src}|{variant}", "lang": lang, "source": "template", "variant": variant,
                             "label": label, "text": text})
    for name, text in EDGE:
        rows.append({"id": f"edge:{name}", "lang": "mixed", "source": "edge", "variant": "edge", "label": None,
                     "text": text})
    if len(rows) != N_GOLDEN:
        raise SystemExit(f"golden set has {len(rows)} messages, expected {N_GOLDEN}")
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate golden ids")
    return rows


def write(obj, path: Path) -> str:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(data)
    log(f"[export] wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} ({len(data) / 1e6:.1f} MB)")
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task = load_yaml("configs/step2.yaml")["tasks"]["mendeley_3class"]
    if task["classes"] != CLASSES:
        raise SystemExit(f"unexpected classes {task['classes']}")
    final = load_yaml("configs/final.yaml")
    tau = float(final["method"]["tau"])
    cost = cost_rule(list(final["metrics"]["cost_ratios"]))
    tr_ids, tr_texts, tr_y = load_split(task, "train")
    va_ids, va_texts, _ = load_split(task, "val")
    cases = pd.read_csv(ROOT / "data" / "kz_cases" / "cases.csv", dtype=str, keep_default_na=False)
    adapt = pd.read_csv(ROOT / "data" / "kz_adapt" / "adapt.csv", dtype=str, keep_default_na=False)
    step6b = pd.read_csv(ROOT / "results" / "step6b" / "adapt" / "predictions.csv", dtype={"id": str})
    step6b = step6b[step6b.set == "cases"]
    case_texts = cases["text"].tolist()

    # base
    base, params, fixed = fit_lite(tr_texts, tr_y)
    checks_base = {
        "val_vs_final_val_seed1": check("base / clean val", base.predict_proba(va_texts), frozen_probs(
            pd.read_csv(ROOT / "results/final/preds/mendeley_3class/svm_char/val_seed1.csv", dtype={"id": str}), va_ids)),
        "cases_vs_step6b_before": check("base / 63 KZ cases", base.predict_proba(case_texts),
                                        frozen_probs(step6b[step6b.run_id == "before"], cases["id"])),
    }
    s8 = read_json(ROOT / "results" / "step8" / "model.json")
    cc = base.model.calibrated_classifiers_[0]
    d8 = max(float(np.abs(cc.estimator.intercept_ - np.asarray(s8["intercept"])).max()),
             float(np.abs(np.array([c.a_ for c in cc.calibrators]) - np.asarray(s8["platt_a"])).max()),
             float(np.abs(np.array([c.b_ for c in cc.calibrators]) - np.asarray(s8["platt_b"])).max()))
    log(f"[export] check base / step8 model.json: n_features {len(base.vectorizer.vocabulary_)} vs {s8['n_features']}, "
        f"max |intercept, Platt a, b - step8| = {d8:.2e}")
    if len(base.vectorizer.vocabulary_) != s8["n_features"] or d8 > CHECK_TOL:
        raise SystemExit("base model differs from results/step8/model.json")
    checks_base["params_vs_step8_model_json"] = d8

    # adapted (condition B: all adaptation messages)
    ad_y = adapt["label"].map(CLASSES.index).to_numpy()
    adapted, _, _ = fit_lite(np.concatenate([np.asarray(tr_texts, dtype=object), adapt["text"].to_numpy(dtype=object)]),
                             np.concatenate([np.asarray(tr_y), ad_y]))
    run_b = f"B_n{len(adapt)}_s0"
    checks_adapted = {"cases_vs_step6b_" + run_b: check(
        f"adapted / 63 KZ cases ({run_b})", adapted.predict_proba(case_texts),
        frozen_probs(step6b[step6b.run_id == run_b], cases["id"]))}

    counts = lambda y: {c: int((np.asarray(y) == k).sum()) for k, c in enumerate(CLASSES)}  # noqa: E731
    models = {
        "base": (base, "Lite base: frozen final svm_char, seed 1 (Mendeley 3-class train)",
                 {"data": "mendeley_3class train", "n": len(tr_y), "per_class": counts(tr_y)}, checks_base),
        "adapted": (adapted, "Lite adapted to RU/KZ: step 6b condition B (Mendeley train + 324 adaptation messages), seed 1",
                    {"data": "mendeley_3class train + data/kz_adapt/adapt.csv (all)", "n": len(tr_y) + len(adapt),
                     "per_class": counts(np.concatenate([tr_y, ad_y]))}, checks_adapted),
    }
    exported, sha = {}, {}
    for name, (clf, desc, info, chk) in models.items():
        exported[name] = export(clf, name, desc, params, fixed, info, chk, tau, cost)
        sha[name] = write(exported[name], out / f"lite_{name}.json")
        prepare(exported[name])

    # golden
    rows = golden_messages()
    texts = [r["text"] for r in rows]
    worst = 0.0
    for name, (clf, *_rest) in models.items():
        P = clf.predict_proba(texts)
        S = clf.model.calibrated_classifiers_[0].estimator.decision_function(clf.vectorizer.transform(texts))
        for r, p, s, t in zip(rows, P, S, texts):
            p_ref, s_ref = reference_proba(exported[name], t)
            worst = max(worst, float(np.abs(p_ref - p).max()), float(np.abs(s_ref - s).max()))
            r[name] = {"proba": [float(v) for v in p], "pred": CLASSES[int(p.argmax())], "scores": [float(v) for v in s],
                       "confidence": float(p.max()), "uncertain": bool(p.max() < tau)}
    log(f"[export] reference re-implementation from the JSON vs sklearn on the golden set: max diff {worst:.2e}")
    if worst > 1e-9:
        raise SystemExit("the exported JSON does not reproduce predict_proba")
    golden = {"format": "smishfuse-golden/1", "classes": CLASSES, "tolerance": 1e-4, "tau": tau,
              "models": {n: {"file": f"lite_{n}.json", "sha256": sha[n]} for n in models},
              "reference_max_diff": worst, "n": len(rows), "messages": rows}
    write(golden, out / "golden.json")
    by = pd.DataFrame([{k: r[k] for k in ("lang", "variant")} for r in rows])
    log(f"[export] golden: {len(rows)} messages; by language {by.lang.value_counts().to_dict()}; "
        f"by variant {by.variant.value_counts().to_dict()}")
    agree = np.mean([r["base"]["pred"] == r["adapted"]["pred"] for r in rows])
    log(f"[export] golden: base and adapted agree on {agree:.3f} of the messages")


if __name__ == "__main__":
    main()
