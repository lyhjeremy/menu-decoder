"""Pydantic contracts for Menu Decoder. See MENU_DECODER_SPEC.md §1."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Allergen = Literal[
    "gluten", "crustaceans", "eggs", "fish", "peanuts", "soy", "milk",
    "nuts", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs",
]

AllergenLevel = Literal["contains", "may", "not_indicated"]


class Dish(BaseModel):
    id: str
    original: str
    translation: str | None = None
    description: str | None = None
    price: str | None = None
    unreadable: bool = False
    confidence: float = 1.0


class MenuSection(BaseModel):
    name: str
    dishes: list[Dish]


class Menu(BaseModel):
    language: str
    sections: list[MenuSection]

    def all_dishes(self) -> list[Dish]:
        return [d for s in self.sections for d in s.dishes]


class DietBrief(BaseModel):
    diets: list[Literal["vegetarian", "vegan", "pescatarian", "halal", "kosher"]] = []
    allergens: list[Allergen] = []
    dislikes: list[str] = []
    spice_max: int | None = None
    budget: str | None = None


class AllergenCall(BaseModel):
    allergen: Allergen
    level: AllergenLevel


class DishTags(BaseModel):
    dish_id: str
    cuisine: str
    allergen_calls: list[AllergenCall]
    spice: int  # 0-3
    vegetarian: Literal["yes", "no", "unclear"]
    vegan: Literal["yes", "no", "unclear"]


class DishTagsBatch(BaseModel):
    tags: list[DishTags]


class Recommendation(BaseModel):
    dish_id: str
    reason: str


class Recommendations(BaseModel):
    picks: list[Recommendation]
