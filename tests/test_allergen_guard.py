import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from allergen_guard import guard_levels, guard_menu, open_allergens, second_opinion, validate_reply
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


BATCH_SLUGS = [{"id": "hummus", "text": "Dish: Hummus.", "open": ["sesame", "milk"]},
               {"id": "ceviche", "text": "Dish: Ceviche.", "open": ["fish"]}]


def test_validate_reply_rejects_missing_dish_and_drops_foreign_allergen():
    with pytest.raises(ValueError):
        validate_reply(json.dumps({"dishes": [{"id": "d1", "flags": []}]}), BATCH_SLUGS)
    reply, dropped = validate_reply(json.dumps({"dishes": [
        {"id": "d1", "flags": [{"allergen": "fish", "level": "may"}, {"allergen": "sesame", "level": "contains"}]},
        {"id": "d2", "flags": []}]}), BATCH_SLUGS)
    assert dropped == 1
    assert [d.id for d in reply.dishes] == ["hummus", "ceviche"]
    assert [f.allergen for f in reply.dishes[0].flags] == ["sesame"]


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


def test_guard_menu_raises_user_allergens_only_via_second_opinion():
    from schemas import AllergenCall, Dish, DishTags

    dishes = [Dish(id="a", original="Hummus"), Dish(id="b", original="Moules marinieres")]
    base = [DishTags(dish_id=d.id, cuisine="x", spice=0, vegetarian="unclear", vegan="unclear",
                     allergen_calls=[AllergenCall(allergen=a, level="not_indicated") for a in
                                     ["gluten", "sesame", "molluscs", "milk"]]) for d in dishes]
    seen = []

    def fake(prompt, system, model):
        seen.append(prompt)
        return {"result": json.dumps({"dishes": [{"id": "d1", "flags": [{"allergen": "sesame", "level": "contains"}]},
                                         {"id": "d2", "flags": []}]}),
                "total_cost_usd": 0.0002, "duration_ms": 500}

    out, notes = guard_menu(dishes, base, ["sesame", "molluscs"], call=fake)
    levels = {t.dish_id: {c.allergen: c.level for c in t.allergen_calls} for t in out}
    assert levels["b"]["molluscs"] == "contains"      # lexicon: moules
    assert levels["a"]["sesame"] == "contains"        # second opinion: tahini
    assert levels["a"]["gluten"] == "not_indicated"   # not asked: not one of the user's allergens
    assert levels["b"]["sesame"] == "not_indicated"
    assert notes["second_opinion_calls"] == 1 and [ln for ln in seen[0].splitlines() if ln.startswith("Not indicated so far")] == [
        "Not indicated so far: sesame, molluscs", "Not indicated so far: sesame"]
