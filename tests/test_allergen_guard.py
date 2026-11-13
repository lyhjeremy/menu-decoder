import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from allergen_guard import guard_levels, open_allergens, second_opinion, validate_reply
from allergen_lexicon import lexicon_calls, raise_levels


def test_translated_shellfish_names_map_to_molluscs():
    assert lexicon_calls("Dish: Moules Marinieres.")["molluscs"] == "contains"
    assert lexicon_calls("Dish: Spaghetti alle Vongole.")["molluscs"] == "contains"
    assert lexicon_calls("Dish: 蚝油芥兰.")["molluscs"] == "contains"


def test_accents_and_case_are_folded():
    assert lexicon_calls("SAUMON AU BEURRE BLANC")["fish"] == "contains"
    assert lexicon_calls("Pâte with crème fraîche")["milk"] == "contains"


def test_wine_and_celery_salt():
    calls = lexicon_calls("Braised in red wine, with a celery-salt rim")
    assert calls["sulphites"] == "contains"
    assert calls["celery"] == "contains"


def test_whole_words_only():
    # "cod" inside "avocado" and "egg" inside "eggplant" must not match
    calls = lexicon_calls("Avocado and eggplant salad")
    assert "fish" not in calls and "eggs" not in calls


def test_plural_suffix_matches():
    assert lexicon_calls("three anchovies and two eggs")["fish"] == "contains"


def test_raise_never_lowers():
    out = raise_levels({"milk": "contains", "soy": "may"}, {"milk": "may", "soy": "contains"})
    assert out == {"milk": "contains", "soy": "contains"}


def test_guard_levels_keeps_lora_and_adds_lexicon():
    lora = {a: "not_indicated" for a in ["gluten", "fish", "milk"]}
    lora["gluten"] = "contains"
    out = guard_levels("Dish: Sole Meuniere.", lora)
    assert out["gluten"] == "contains" and out["fish"] == "contains" and out["milk"] == "contains"
    assert out["lupin"] == "not_indicated"


def test_open_allergens_respects_user_selection():
    levels = {"fish": "contains", "soy": "not_indicated", "milk": "not_indicated"}
    assert open_allergens(levels, only=["fish", "soy"]) == ["soy"]


BATCH = [{"id": "d1", "text": "Dish: Hummus.", "open": ["sesame", "milk"]},
         {"id": "d2", "text": "Dish: Ceviche.", "open": ["fish"]}]


def test_validate_reply_rejects_missing_dish_and_foreign_allergen():
    with pytest.raises(ValueError):
        validate_reply(json.dumps({"dishes": [{"id": "d1", "flags": []}]}), BATCH)
    with pytest.raises(ValueError):
        validate_reply(json.dumps({"dishes": [{"id": "d1", "flags": [{"allergen": "fish", "level": "may"}]},
                                              {"id": "d2", "flags": []}]}), BATCH)


def test_second_opinion_retries_then_succeeds():
    replies = iter(["not json", json.dumps({"dishes": [
        {"id": "d1", "flags": [{"allergen": "sesame", "level": "contains", "why": "tahini"}]},
        {"id": "d2", "flags": [{"allergen": "fish", "level": "contains", "why": "raw fish"}]}]})])

    def fake(prompt, system, model):
        return {"result": next(replies), "total_cost_usd": 0.001, "duration_ms": 900}

    out = second_opinion(BATCH, call=fake)
    assert out["fallback"] is False and len(out["attempts"]) == 2
    assert out["calls"]["d1"] == {"sesame": "contains"}


def test_second_opinion_fails_toward_warning():
    def fake(prompt, system, model):
        return {"result": "{}", "total_cost_usd": 0.0, "duration_ms": 1}

    out = second_opinion(BATCH, call=fake)
    assert out["fallback"] is True
    assert out["calls"]["d1"] == {"sesame": "may", "milk": "may"}
