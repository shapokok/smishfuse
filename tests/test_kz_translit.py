"""Step 6d (scripts/make_kz_translit.py): the transliteration tables and the mixing operator on hand-written strings."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_kz_translit import (  # noqa: E402
    CYR, KK_DECREE, KK_LATIN2018, KK_OUR_RULE, KK_SMS, RU_ICAO, RU_SMS, mix, partial, rng_for, transliterate,
)

RU_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
KK_LETTERS = RU_LETTERS + "әғқңөұүһі"


def test_tables_cover_the_alphabets():
    assert set(RU_ICAO) == set(RU_LETTERS) and set(RU_SMS) == set(RU_LETTERS)
    assert set(KK_LATIN2018) == set(KK_LETTERS) and set(KK_SMS) == set(KK_LETTERS)
    # the decree table has the 34 letters of Decree No. 637 (2018); the other 8 are our rule
    assert len(KK_DECREE) == 34 and set(KK_OUR_RULE) == set("яюёцщэъь")
    assert not set(KK_DECREE) & set(KK_OUR_RULE)


def test_icao_9303():
    assert transliterate("Щука, объявление, Юлия, Цой, ёж", RU_ICAO) == "Shchuka, obieiavlenie, Iuliia, Tsoi, ezh"
    assert transliterate("ЖКХ", RU_ICAO) == "ZHKKH"


def test_decree_637():
    assert transliterate("Қазақстан Әділет ұялы Шымкент", KK_LATIN2018) == "Qazaqstan Ádilet uıaly Shymkent"
    assert transliterate("үй, ғылым, өнер, бұғатталды, Ірі", KK_LATIN2018) == "úı, ǵylym, óner, buǵattaldy, Iri"
    assert transliterate("ШЫҒЫС", KK_LATIN2018) == "SHYǴYS"


def test_only_cyrillic_changes():
    text = "Kaspi Bank: код 4821, kaspi-verify.example, +7 701 000 00 00!"
    out = transliterate(text, RU_ICAO)
    assert out == "Kaspi Bank: kod 4821, kaspi-verify.example, +7 701 000 00 00!"
    assert not CYR.search(transliterate("Ваш код: 1234. Никому не говорите.", RU_SMS, rng_for("x", 1)))


def test_sms_transliteration_is_seeded_and_uses_the_table():
    text = "Ваша карта заблокирована, срочно позвоните в службу безопасности"
    a = transliterate(text, RU_SMS, rng_for("id_1", 1))
    assert a == transliterate(text, RU_SMS, rng_for("id_1", 1))
    outs = {transliterate(text, RU_SMS, rng_for(f"id_{i}", 1)) for i in range(20)}
    assert len(outs) > 1  # several spellings are drawn
    for o in outs:
        assert o.startswith("Va") and ("sh" in o[:5] or "w" in o[:5])


def test_partial_changes_about_half_of_the_words():
    text = " ".join(["слово"] * 200)
    out, done, total = partial(text, RU_SMS, rng_for("id_p", 3))
    assert total == 200 and 70 < done < 130
    assert sum(w == "слово" for w in out.split()) == total - done


def test_mix_takes_sentences_in_turn():
    ru = "Первое предложение. Второе предложение! Третье?"
    kk = "Бірінші сөйлем. Екінші сөйлем! Үшінші?"
    assert mix(ru, kk) == "Первое предложение. Екінші сөйлем! Третье?"
    assert mix(kk, ru) == "Бірінші сөйлем. Второе предложение! Үшінші?"
    assert mix("Одно предложение.", "Бір сөйлем.") is None
    assert mix("Раз. Два.", "Бір.") is None
