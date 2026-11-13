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
    out = []
    for d in batch:
        out.append(f"### {d['id']}\n{d['text']}\nNot indicated so far: {', '.join(d['open'])}")
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


def validate_reply(text: str, batch: list[dict]) -> SecondOpinionReply:
    """Parse and validate a reply; also reject ids or allergens outside the request."""
    reply = SecondOpinionReply.model_validate(json.loads(_strip_fences(text)))
    wanted = {d["id"]: set(d["open"]) for d in batch}
    got = {d.id for d in reply.dishes}
    if got != set(wanted):
        raise ValueError(f"dish ids differ: missing {sorted(set(wanted) - got)}, extra {sorted(got - set(wanted))}")
    for d in reply.dishes:
        bad = [f.allergen for f in d.flags if f.allergen not in wanted[d.id]]
        if bad:
            raise ValueError(f"{d.id}: flags allergens that were not open: {bad}")
    return reply


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
            reply = validate_reply(env.get("result", ""), batch)
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
