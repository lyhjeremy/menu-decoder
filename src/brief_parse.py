"""Turn the free-text diet and allergen boxes into a DietBrief.

People type "shellfish", "dairy" or "tree nuts", not the 14 EU allergen names the
tagger uses. Common names are mapped onto those 14; anything still unknown is
returned to the caller so the app can say plainly that it was NOT checked,
instead of crashing or dropping it silently.
"""
from __future__ import annotations

from typing import get_args

from schemas import Allergen, DietBrief

ALLERGENS: tuple[str, ...] = get_args(Allergen)
DIETS: tuple[str, ...] = get_args(DietBrief.model_fields["diets"].annotation.__args__[0])

ALLERGEN_ALIASES: dict[str, tuple[str, ...]] = {
    "shellfish": ("crustaceans", "molluscs"),
    "seafood": ("fish", "crustaceans", "molluscs"),
    "shrimp": ("crustaceans",), "prawn": ("crustaceans",), "prawns": ("crustaceans",),
    "crab": ("crustaceans",), "lobster": ("crustaceans",), "crustacean": ("crustaceans",),
    "squid": ("molluscs",), "octopus": ("molluscs",), "oyster": ("molluscs",), "oysters": ("molluscs",),
    "clam": ("molluscs",), "clams": ("molluscs",), "mussel": ("molluscs",), "mussels": ("molluscs",),
    "scallop": ("molluscs",), "scallops": ("molluscs",), "mollusc": ("molluscs",), "mollusk": ("molluscs",),
    "mollusks": ("molluscs",),
    "dairy": ("milk",), "lactose": ("milk",), "cheese": ("milk",),
    "egg": ("eggs",),
    "peanut": ("peanuts",), "groundnut": ("peanuts",), "groundnuts": ("peanuts",),
    "tree nut": ("nuts",), "tree nuts": ("nuts",), "nut": ("nuts",),
    "wheat": ("gluten",), "barley": ("gluten",), "rye": ("gluten",), "oats": ("gluten",),
    "soya": ("soy",), "soybean": ("soy",), "soybeans": ("soy",),
    "sulfites": ("sulphites",), "sulfite": ("sulphites",), "sulphite": ("sulphites",),
    "lupine": ("lupin",),
}

DIET_ALIASES: dict[str, str] = {
    "veggie": "vegetarian", "plant-based": "vegan", "plant based": "vegan",
    "pescetarian": "pescatarian",
}


def _items(text: str | None) -> list[str]:
    return [t.strip().lower() for t in (text or "").split(",") if t.strip()]


def parse_brief(diets_text: str | None, allergens_text: str | None,
                spice_max: int | None = None) -> tuple[DietBrief, list[str], list[str]]:
    """Return (brief, unknown_allergens, unknown_diets)."""
    allergens: list[str] = []
    unknown_allergens: list[str] = []
    for item in _items(allergens_text):
        mapped = (item,) if item in ALLERGENS else ALLERGEN_ALIASES.get(item)
        if mapped:
            allergens += [a for a in mapped if a not in allergens]
        else:
            unknown_allergens.append(item)

    diets: list[str] = []
    unknown_diets: list[str] = []
    for item in _items(diets_text):
        d = item if item in DIETS else DIET_ALIASES.get(item)
        if d:
            if d not in diets:
                diets.append(d)
        else:
            unknown_diets.append(item)

    brief = DietBrief(diets=diets, allergens=allergens, spice_max=spice_max)
    return brief, unknown_allergens, unknown_diets


def warning_text(unknown_allergens: list[str], unknown_diets: list[str]) -> str:
    """Markdown warning for anything the app could not check; empty if none."""
    parts = []
    if unknown_allergens:
        parts.append(
            f"⚠ **Not checked: {', '.join(unknown_allergens)}.** I can only tag these 14 allergens: "
            f"{', '.join(ALLERGENS)}. Ask the staff about anything else."
        )
    if unknown_diets:
        parts.append(f"⚠ Diet not recognized: {', '.join(unknown_diets)}. I know: {', '.join(DIETS)}.")
    return "\n\n".join(parts)
