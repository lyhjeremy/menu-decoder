"""Prompt assembly for Menu Decoder. See MENU_DECODER_SPEC.md §3-4."""
from __future__ import annotations

from context import ContextBudgeter, Section
from schemas import Dish, DishTags, Menu

TAGGING_SYSTEM = """You are tagging dishes from a menu for allergen/dietary information.
Be ASYMMETRIC and cautious by design: a missed allergen is dangerous, a false
warning is just an inconvenience. Call "contains" when the allergen is a
defining/named ingredient, "may" when it's plausibly present in a typical
preparation even if not named (e.g. sesame oil in many Asian dishes, gluten
in most sauces), and "not_indicated" ONLY when it would be unusual for this
dish to contain it. When uncertain, prefer "may" over "not_indicated"."""

RECOMMEND_SYSTEM = """You are picking 2-3 dishes for a diner from a pre-filtered
SAFE list (every dish given to you already passed their allergen/diet
filters in code -- you do not need to re-check safety, only pick good
matches for their stated preferences). Ground every reason in the dish's
actual description. Never recommend a dish not in the list given to you."""


def build_tagging_prompt(dishes: list[Dish], *, total_budget: int = 2000) -> tuple[str, ContextBudgeter]:
    budgeter = ContextBudgeter(total_budget=total_budget)
    budgeter.add(Section(name="system", items=[TAGGING_SYSTEM], priority=0, min_tokens=300, max_tokens=350))
    dish_lines = [f"[{d.id}] {d.translation or d.original}" + (f" -- {d.description}" if d.description else "")
                  for d in dishes]
    budgeter.add(Section(name="dishes", items=dish_lines, priority=1, max_tokens=1400))
    budgeter.add(Section(
        name="output_format",
        items=["Respond with ONLY JSON: {\"tags\": [...]}, one entry per dish given, "
               "each: dish_id, cuisine, allergen_calls (list of {allergen, level} covering "
               "ALL 14 allergens), spice (0-3), vegetarian, vegan."],
        priority=0, min_tokens=150,
    ))
    packed = budgeter.pack()
    return packed.prompt, budgeter


def build_recommend_prompt(safe_dishes: list[Dish], brief_desc: str, *, total_budget: int = 1200) -> str:
    budgeter = ContextBudgeter(total_budget=total_budget)
    budgeter.add(Section(name="system", items=[RECOMMEND_SYSTEM], priority=0, min_tokens=250, max_tokens=300))
    budgeter.add(Section(name="brief", items=[f"Diner preferences: {brief_desc}"], priority=0, min_tokens=50))
    dish_lines = [f"[{d.id}] {d.translation or d.original}" + (f" -- {d.description}" if d.description else "")
                  for d in safe_dishes]
    budgeter.add(Section(name="safe_dishes", items=dish_lines, priority=1, max_tokens=700))
    packed = budgeter.pack()
    return packed.prompt
