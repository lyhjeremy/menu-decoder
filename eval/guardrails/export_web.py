"""Write overview/guard.json for the write-up's dish inspector and per-allergen table.

    python eval/guardrails/export_web.py

Reads eval/guardrails/results.json and outputs/scored_*.json (from run_guard.py
score) and the run-1 Haiku replies for their reasons. Levels are packed one
character per allergen (c contains, m may, n not indicated) in ALLERGENS order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from allergen_lexicon import lexicon_hits  # noqa: E402
ALLERGENS = ["gluten", "crustaceans", "eggs", "fish", "peanuts", "soy", "milk",
             "nuts", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs"]
CODE = {"contains": "c", "may": "m", "not_indicated": "n"}


def pack(levels: dict) -> str:
    return "".join(CODE[levels[a]] for a in ALLERGENS)


def lex_by_allergen(text: str) -> dict:
    out: dict[str, list[str]] = {}
    for h in lexicon_hits(text):
        for allergen, _ in h["calls"]:
            if h["term"] not in out.setdefault(allergen, []):
                out[allergen].append(h["term"])
    return out


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    why = {}
    for b in json.loads((HERE / "outputs/second_adversarial_haiku_r1.json").read_text())["batches"]:
        why.update(b["why"])
    adv = json.loads((HERE / "outputs/scored_adversarial.json").read_text())
    dishes = [{"id": d["id"], "dish": d["text"][len("Dish: "):], "trap": d["trap"], "hidden": d["hidden"],
               "basis": d["basis"], "truth": pack(d["truth"]), "l1": pack(d["L1"]), "l2": pack(d["L2"]),
               "l3": pack(d["L3"]), "lex": lex_by_allergen(d["text"]), "why": why.get(d["id"], {})} for d in adv]
    per = {}
    for name, block in res["sets"].items():
        per[name] = {k: {a: [v["all"]["per_allergen"][a]["n_contains"], v["all"]["per_allergen"][a]["false_safe_count"],
                             v["all"]["per_allergen"][a]["recall"]] for a in ALLERGENS}
                     for k, v in block["layers"].items()}
    out = {"source": "eval/guardrails/results.json and outputs/scored_adversarial.json",
           "allergens": ALLERGENS, "dishes": dishes, "per_allergen": per}
    (REPO / "overview" / "guard.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("wrote overview/guard.json", len(dishes), "dishes")


if __name__ == "__main__":
    main()
