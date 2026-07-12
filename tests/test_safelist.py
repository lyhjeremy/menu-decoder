import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import AllergenCall, DietBrief, Dish, DishTags
from safelist import filter_safe_dishes, is_safe_for_brief, render_dish_badges


def make_tags(dish_id="d1", allergen_calls=None, cuisine="test", spice=1,
              vegetarian="yes", vegan="yes"):
    return DishTags(
        dish_id=dish_id, cuisine=cuisine, allergen_calls=allergen_calls or [],
        spice=spice, vegetarian=vegetarian, vegan=vegan,
    )


def test_contains_is_unsafe():
    tags = make_tags(allergen_calls=[AllergenCall(allergen="peanuts", level="contains")])
    brief = DietBrief(allergens=["peanuts"])
    assert is_safe_for_brief(tags, brief) is False


def test_may_is_unsafe_the_asymmetric_default():
    # 'may' (uncertain) must be treated as UNSAFE, not a soft pass -- this is
    # the whole point of the fail-safe design.
    tags = make_tags(allergen_calls=[AllergenCall(allergen="peanuts", level="may")])
    brief = DietBrief(allergens=["peanuts"])
    assert is_safe_for_brief(tags, brief) is False


def test_not_indicated_is_safe():
    tags = make_tags(allergen_calls=[AllergenCall(allergen="peanuts", level="not_indicated")])
    brief = DietBrief(allergens=["peanuts"])
    assert is_safe_for_brief(tags, brief) is True


def test_missing_allergen_call_defaults_to_not_indicated_and_safe():
    # if the tagger simply didn't call out an allergen the user cares about,
    # treat it as not_indicated -- same as an explicit 'not_indicated' call.
    tags = make_tags(allergen_calls=[])
    brief = DietBrief(allergens=["peanuts"])
    assert is_safe_for_brief(tags, brief) is True


def test_unrelated_allergen_does_not_block():
    tags = make_tags(allergen_calls=[AllergenCall(allergen="milk", level="contains")])
    brief = DietBrief(allergens=["peanuts"])  # user doesn't care about milk
    assert is_safe_for_brief(tags, brief) is True


def test_vegetarian_constraint():
    tags = make_tags(vegetarian="no")
    brief = DietBrief(diets=["vegetarian"])
    assert is_safe_for_brief(tags, brief) is False

    tags_unclear = make_tags(vegetarian="unclear")
    assert is_safe_for_brief(tags_unclear, brief) is False  # unclear also fails safe


def test_spice_max_constraint():
    tags = make_tags(spice=3)
    brief = DietBrief(spice_max=1)
    assert is_safe_for_brief(tags, brief) is False


def test_no_green_checkmark_ever_in_badge_map():
    from safelist import ALLERGEN_BADGE
    assert "✓" not in ALLERGEN_BADGE.values()
    assert all("✗" in v or "⚠" in v or "○" in v for v in ALLERGEN_BADGE.values())


def test_filter_safe_dishes_excludes_unsafe():
    dish1 = Dish(id="d1", original="Pad Thai")
    dish2 = Dish(id="d2", original="Pad Thai with peanuts")
    tags1 = make_tags(dish_id="d1", allergen_calls=[AllergenCall(allergen="peanuts", level="not_indicated")])
    tags2 = make_tags(dish_id="d2", allergen_calls=[AllergenCall(allergen="peanuts", level="contains")])
    brief = DietBrief(allergens=["peanuts"])

    safe = filter_safe_dishes([(dish1, tags1), (dish2, tags2)], brief)
    assert [d.id for d in safe] == ["d1"]


def test_render_badges_only_shows_user_relevant_allergens():
    tags = make_tags(allergen_calls=[
        AllergenCall(allergen="peanuts", level="contains"),
        AllergenCall(allergen="milk", level="contains"),
    ])
    badges = render_dish_badges(tags, user_allergens=["peanuts"])
    assert len(badges) == 1
    assert "peanuts" in badges[0]
