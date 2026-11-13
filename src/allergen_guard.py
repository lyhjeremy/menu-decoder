"""Layered allergen guard around the LoRA tagger.

    layer 1  LoRA tagger (training/adapters): dish text -> 14 allergen calls
    layer 2  allergen_lexicon: ingredient, sauce and stock words -> raise calls
    layer 3  second opinion: a cheap Claude model is asked about every allergen
             that is still "not_indicated" after layers 1 and 2, and can only
             raise it to "may" or "contains"

No layer can lower a call, so adding a layer can only cut false-safe calls, at
the price of more warnings. Layer 3 runs only on dishes that still have an
allergen at "not_indicated"; in the app that means only the user's own
allergens, in the eval all 14 are scored so every dish is sent.

The second-opinion reply must satisfy SecondOpinionReply (pydantic). A reply that
fails to parse or validate is retried once with the error appended; if the retry
fails too, every open allergen of that batch is raised to "may", so a broken
check fails toward warning.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from typing import Literal

from pydantic import BaseModel, ValidationError

from allergen_lexicon import LEVEL_RANK, lexicon_calls, raise_levels

ALLERGENS = ["gluten", "crustaceans", "eggs", "fish", "peanuts", "soy", "milk",
             "nuts", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs"]

SECOND_OPINION_SYSTEM = ("You check restaurant dishes for allergens that a first tagger may have missed. "
                         "Reply with JSON only.")

SECOND_OPINION_PROMPT = """A first tagger marked the allergens listed under each dish below as "not indicated".
For each listed allergen, decide from the dish's standard recipe, as named and described:
- "contains": the standard recipe includes it.
- "may": it is often present, for example in a sauce, stock, marinade, wine, batter, dressing or garnish.
- leave it out: it would be unusual in this dish.
Use the EU allergen meanings: nuts are tree nuts (not peanuts, not pine nuts); sulphites come mainly from
wine, vinegar and dried fruit; molluscs include squid, octopus, clams, mussels, oysters, scallops and oyster
sauce; celery includes celeriac and celery salt. A missed allergen can hurt someone, so when unsure, say "may".
Only use allergens from that dish's own list.

Reply as JSON: {{"dishes": [{{"id": "d1", "flags": [{{"allergen": "fish", "level": "contains", "why": "under 12 words"}}]}}]}}
Include every dish id, with an empty flags list if nothing applies.

{items}
"""


class Flag(BaseModel):
    allergen: Literal["gluten", "crustaceans", "eggs", "fish", "peanuts", "soy", "milk",
                      "nuts", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs"]
    level: Literal["contains", "may"]
    why: str = ""


class DishFlags(BaseModel):
    id: str
    flags: list[Flag]


class SecondOpinionReply(BaseModel):
    dishes: list[DishFlags]


def _render(batch: list[dict]) -> str:
    """Dishes are shown as d1..dN, whatever their real ids: on the first run Haiku
    copied the example id "d1" for every dish when the real ids were slugs."""
    out = []
    for k, d in enumerate(batch, 1):
        out.append(f"### d{k}\n{d['text']}\nNot indicated so far: {', '.join(d['open'])}")
    return "\n\n".join(out)


def _claude(prompt: str, system: str, model: str, timeout: int = 240) -> dict:
    """One isolated claude -p call: no tools, no settings, no session, temp cwd.
    Returns the CLI's JSON envelope (result text, total_cost_usd, duration_ms, usage)."""
    cmd = ["claude", "-p", "--tools", "", "--setting-sources", "", "--no-session-persistence",
           "--output-format", "json", "--system-prompt", system, "--model", model]
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
                              cwd=cwd, env={**os.environ})
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p exit {proc.returncode}: {proc.stderr[:300]}")
    return json.loads(proc.stdout)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def validate_reply(text: str, batch: list[dict]) -> tuple[SecondOpinionReply, int]:
    """Parse and validate a reply against the request (ids d1..dN). Flags for an
    allergen that was not open are dropped and counted: they cannot change anything,
    since the earlier layers already warn about it. Returns (reply with real ids, dropped)."""
    reply = SecondOpinionReply.model_validate(json.loads(_strip_fences(text)))
    by_pos = {f"d{k}": d for k, d in enumerate(batch, 1)}
    got = {d.id for d in reply.dishes}
    if got != set(by_pos):
        raise ValueError(f"dish ids differ: missing {sorted(set(by_pos) - got)}, extra {sorted(got - set(by_pos))}")
    dropped = 0
    for d in reply.dishes:
        src = by_pos[d.id]
        keep = [f for f in d.flags if f.allergen in src["open"]]
        dropped += len(d.flags) - len(keep)
        d.flags = keep
        d.id = src["id"]
    return reply, dropped


def second_opinion(batch: list[dict], model: str = "haiku", call=_claude) -> dict:
    """batch: [{"id", "text", "open": [allergens]}] -> {"calls": {id: {allergen: level}},
    "why": {id: {allergen: why}}, "attempts": [...], "fallback": bool}"""
    prompt = SECOND_OPINION_PROMPT.format(items=_render(batch))
    attempts = []
    for attempt in range(2):
        t0 = time.perf_counter()
        env = call(prompt, SECOND_OPINION_SYSTEM, model)
        rec = {"attempt": attempt + 1, "model": model, "cost_usd": env.get("total_cost_usd"),
               "duration_ms": env.get("duration_ms"), "wall_s": round(time.perf_counter() - t0, 2),
               "usage": env.get("usage"), "raw": env.get("result", "")}
        try:
            reply, rec["dropped_flags"] = validate_reply(env.get("result", ""), batch)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            rec["error"] = str(e)[:400]
            attempts.append(rec)
            prompt = (SECOND_OPINION_PROMPT.format(items=_render(batch)) +
                      f"\n\nYour previous reply was rejected: {str(e)[:300]}\nReply again with valid JSON only.")
            continue
        attempts.append(rec)
        calls = {d.id: {f.allergen: f.level for f in d.flags} for d in reply.dishes}
        why = {d.id: {f.allergen: f.why for f in d.flags} for d in reply.dishes}
        return {"calls": calls, "why": why, "attempts": attempts, "fallback": False}
    calls = {d["id"]: {a: "may" for a in d["open"]} for d in batch}
    return {"calls": calls, "why": {}, "attempts": attempts, "fallback": True}


def open_allergens(levels: dict[str, str], only: list[str] | None = None) -> list[str]:
    pool = only or ALLERGENS
    return [a for a in pool if levels.get(a, "not_indicated") == "not_indicated"]


def guard_levels(text: str, lora_levels: dict[str, str]) -> dict[str, str]:
    """Layers 1 and 2: the LoRA's calls raised by the lexicon. Deterministic, no network."""
    base = {a: lora_levels.get(a, "not_indicated") for a in ALLERGENS}
    return raise_levels(base, lexicon_calls(text))


def level_max(a: str, b: str) -> str:
    return a if LEVEL_RANK[a] >= LEVEL_RANK[b] else b


def guard_menu(dishes, tags_list, user_allergens: list[str], second=True, call=_claude, model: str = "haiku"):
    """Apply layers 2 and 3 to the app's tags for one menu, in place of the raw tagger output.

    dishes: schemas.Dish list; tags_list: schemas.DishTags list (same dish ids).
    Layer 3 asks only about the user's allergens that are still not indicated, ten
    dishes per call, and is skipped when `second` is False or the CLI call fails
    (the lexicon result stands). Returns (new tags list, notes)."""
    from schemas import AllergenCall, DishTags

    by_id = {d.id: d for d in dishes}
    texts, levels = {}, {}
    for t in tags_list:
        d = by_id.get(t.dish_id)
        if d is None:
            continue
        texts[t.dish_id] = " ".join(x for x in (d.original, d.translation, d.description) if x)
        levels[t.dish_id] = guard_levels(texts[t.dish_id], {c.allergen: c.level for c in t.allergen_calls})
    notes = {"second_opinion_calls": 0, "second_opinion_error": None}
    if second and user_allergens:
        todo = [{"id": i, "text": texts[i], "open": open_allergens(lv, only=user_allergens)} for i, lv in levels.items()]
        todo = [x for x in todo if x["open"]]
        for b in range(0, len(todo), 10):
            try:
                res = second_opinion(todo[b:b + 10], model=model, call=call)
            except Exception as e:  # no CLI, timeout: keep layers 1 and 2
                notes["second_opinion_error"] = str(e)[:200]
                break
            notes["second_opinion_calls"] += len(res["attempts"])
            for i, extra in res["calls"].items():
                levels[i] = raise_levels(levels[i], extra)
    out = []
    for t in tags_list:
        lv = levels.get(t.dish_id)
        if lv is None:
            out.append(t)
            continue
        out.append(DishTags(**{**t.model_dump(), "allergen_calls": [AllergenCall(allergen=a, level=lv[a]) for a in ALLERGENS]}))
    return out, notes
