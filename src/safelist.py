"""The allergen fail-safe: 3 enforced layers, asymmetric by design.
See MENU_DECODER_SPEC.md §2.

Layer 1 (prompt rubric) lives in the tagging prompt, not here.
Layer 2 (code-level rendering) + Layer 3 (deterministic safe-list filter)
are both here -- neither can be overridden by the model, by construction.
"""
from __future__ import annotations

from schemas import Dish, DishTags, DietBrief

# Layer 2: rendering. Deliberately no green checkmark for allergens, ever --
# "not_indicated" reads neutral (a circle), never a safety claim.
ALLERGEN_BADGE = {"contains": "✗", "may": "⚠ ask the staff", "not_indicated": "○ not indicated"}


def render_dish_badges(tags: DishTags, user_allergens: list[str]) -> list[str]:
    badges = []
    for call in tags.allergen_calls:
        if call.allergen in user_allergens:
            badges.append(f"{call.allergen}: {ALLERGEN_BADGE[call.level]}")
    return badges


def is_safe_for_brief(tags: DishTags, brief: DietBrief) -> bool:
    """Layer 3: deterministic hard filter. A dish is safe ONLY if every
    user-relevant allergen is 'not_indicated' AND diet constraints are met.
    'may' is treated as unsafe -- the asymmetric fail-safe design.
    """
    calls_by_allergen = {c.allergen: c.level for c in tags.allergen_calls}
    for allergen in brief.allergens:
        level = calls_by_allergen.get(allergen, "not_indicated")
        if level in ("contains", "may"):
            return False

    if "vegetarian" in brief.diets and tags.vegetarian != "yes":
        return False
    if "vegan" in brief.diets and tags.vegan != "yes":
        return False
    if brief.spice_max is not None and tags.spice > brief.spice_max:
        return False

    return True


def filter_safe_dishes(dishes_with_tags: list[tuple[Dish, DishTags]], brief: DietBrief) -> list[Dish]:
    """The dishes the LLM is ALLOWED to recommend from -- computed in code,
    before any generation call. The model never sees or can override the
    excluded dishes' existence in the recommendation step.
    """
    return [dish for dish, tags in dishes_with_tags if is_safe_for_brief(tags, brief)]
