"""Overnight resumable synthetic corpus generator for the dish tagger.
See MENU_DECODER_SPEC.md §5. ~15 cuisines x ~150 canonical dishes x variants
-> ~3.5-5k rows. Teacher = claude -p with the asymmetric allergen rubric.

Run under caffeinate:
  caffeinate -i python scripts/gen_dishes.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gen_data import DatasetGenerator
from schemas import Allergen

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "dishes.jsonl"

CUISINES = [
    "Italian", "Japanese", "Thai", "Mexican", "Indian", "French",
    "Chinese-Sichuan", "Chinese-Cantonese", "Korean", "Vietnamese",
    "Spanish", "Greek", "Middle-Eastern", "American-diner", "Ethiopian",
]

VARIANTS = ["name_only", "name_with_description", "original_script_with_translation"]

ALLERGENS: list[Allergen] = [
    "gluten", "crustaceans", "eggs", "fish", "peanuts", "soy", "milk",
    "nuts", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs",
]

RUBRIC = f"""You are tagging a dish for allergen/dietary information. Be ASYMMETRIC and
cautious by design: a missed allergen is dangerous, a false warning is just an
inconvenience. For each of these 14 allergens: {', '.join(ALLERGENS)}
-- call "contains" if the allergen is a defining/named ingredient, "may" if it's
plausibly present in a typical preparation (e.g. sesame oil in many Asian dishes,
gluten in most sauces/breading, peanuts in Southeast Asian dishes) even if not
named, and "not_indicated" ONLY when it would be unusual for this dish to contain
it. When uncertain, prefer "may" over "not_indicated"."""


def _stable_id(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def build_items() -> list[dict]:
    items = []
    for cuisine in CUISINES:
        for i in range(150):
            for variant in VARIANTS:
                items.append({
                    "id": _stable_id(cuisine, i, variant),
                    "cuisine": cuisine,
                    "seq": i,
                    "variant": variant,
                })
    return items


def build_prompt(item: dict) -> str:
    variant_instruction = {
        "name_only": "Invent one authentic, real dish name from this cuisine (no description).",
        "name_with_description": "Invent one authentic dish name from this cuisine PLUS a 1-sentence menu description.",
        "original_script_with_translation": "Invent one authentic dish, giving its name in the original script/language AND an English translation.",
    }[item["variant"]]

    return f"""{variant_instruction} Cuisine: {item['cuisine']}.

{RUBRIC}

Respond with ONLY JSON: {{"dish_name": "...", "description": "... or null",
"cuisine": "{item['cuisine']}", "allergen_calls": [{{"allergen": "...", "level": "contains|may|not_indicated"}}, ...one entry per allergen listed above...],
"spice": 0-3, "vegetarian": "yes|no|unclear", "vegan": "yes|no|unclear"}}"""


def parse(raw: str) -> dict:
    return json.loads(raw)


def validate(parsed: dict) -> list[str]:
    violations = []
    if not parsed.get("dish_name"):
        violations.append("missing dish_name")
    calls = parsed.get("allergen_calls", [])
    if len(calls) != len(ALLERGENS):
        violations.append(f"expected {len(ALLERGENS)} allergen_calls, got {len(calls)}")
    seen_allergens = {c.get("allergen") for c in calls}
    if seen_allergens != set(ALLERGENS):
        violations.append(f"allergen_calls must cover exactly {ALLERGENS}, got {sorted(seen_allergens)}")
    for c in calls:
        if c.get("level") not in ("contains", "may", "not_indicated"):
            violations.append(f"invalid level '{c.get('level')}' for {c.get('allergen')}")
    if parsed.get("vegetarian") not in ("yes", "no", "unclear"):
        violations.append("invalid vegetarian value")
    if parsed.get("vegan") not in ("yes", "no", "unclear"):
        violations.append("invalid vegan value")
    if not isinstance(parsed.get("spice"), int) or not (0 <= parsed.get("spice", -1) <= 3):
        violations.append("spice must be an int 0-3")
    return violations


def main():
    items = build_items()
    print(f"Total items to generate: {len(items)}")

    def generate_fn(prompt: str) -> str:
        import llm
        return llm.generate(prompt, tier="smart", json_only=True, max_tokens=1200).text

    gen = DatasetGenerator(
        name="dishes", out_path=OUT_PATH, items=items,
        build_prompt=build_prompt, parse=parse, validate=validate, generate_fn=generate_fn,
    )
    gen.run(max_consecutive_failures=3, sleep_between=1.0)


if __name__ == "__main__":
    main()
