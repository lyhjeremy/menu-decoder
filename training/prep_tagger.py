"""Dish -> cuisine/allergen tagger: dataset prep. See MENU_DECODER_SPEC.md §5.

This is a GENERATIVE STRUCTURED-OUTPUT fine-tune, not a single-label
classifier: given a dish name (+ optional description), the model must
emit the full DishTags JSON -- cuisine, all 14 allergen calls, spice,
vegetarian, vegan -- matching the asymmetric fail-safe rubric in
scripts/gen_dishes.py.

Split by DISH FAMILY (all 3 variants of the same dish -- name_only,
name_with_description, original_script_with_translation -- must land on
the same side of the split), not by row. data/dishes.jsonl's rows only
carry a hashed `item_id` (sha256(cuisine|seq|variant)[:16], see
scripts/gen_dishes.py's `_stable_id`) with no plaintext seq/variant field,
so the family identity is recovered by rebuilding the exact same
`build_items()` list gen_dishes.py used to generate the corpus and
matching on item_id -- deterministic, no guessing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "training" / "lora_harness"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prep import prep_dataset  # noqa: E402
from gen_dishes import ALLERGENS, CUISINES, build_items  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "dishes.jsonl"
OUT_DIR = REPO_ROOT / "data" / "lora"
EVAL_DIR = REPO_ROOT / "eval"

TAGGING_FIELDS = ["cuisine", "allergen_calls", "spice", "vegetarian", "vegan"]


def load_rows() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found -- run scripts/gen_dishes.py first.")
    rows = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def attach_dish_family(rows: list[dict]) -> list[dict]:
    """Recomputes item_id -> (cuisine, seq, variant) via the exact same
    build_items() gen_dishes.py used, then attaches (cuisine, seq) to each
    row as its dish-family identity -- all 3 variants of one dish share the
    same (cuisine, seq) and must never straddle train/test."""
    id_to_family = {item["id"]: (item["cuisine"], item["seq"], item["variant"]) for item in build_items()}
    assert len(id_to_family) == 15 * 150 * 3, f"expected 6750 synthetic items, got {len(id_to_family)}"

    out = []
    missing = 0
    for row in rows:
        family = id_to_family.get(row["item_id"])
        if family is None:
            missing += 1
            continue
        cuisine, seq, variant = family
        row = {**row, "_cuisine": cuisine, "_seq": seq, "_variant": variant}
        out.append(row)
    if missing:
        print(f"WARNING: {missing}/{len(rows)} rows had an item_id not found in the "
              f"recomputed build_items() -- these rows are dropped from training "
              f"(cannot verify their dish-family identity, so cannot safely split them).")
    return out


def to_tagging_json(row: dict) -> str:
    """Assistant target: compact JSON matching gen_dishes.py's teacher
    schema minus dish_name/description (those are the prompt, not the
    target -- the model isn't asked to echo its own input back)."""
    tagging = {k: row[k] for k in TAGGING_FIELDS}
    # Defensive normalization: guarantee all 14 allergens present, in the
    # canonical order, so the model always learns a fixed-shape target
    # (matters for training stability and for a clean parse at inference).
    calls_by_allergen = {c["allergen"]: c["level"] for c in tagging["allergen_calls"]}
    tagging["allergen_calls"] = [{"allergen": a, "level": calls_by_allergen.get(a, "may")} for a in ALLERGENS]
    return json.dumps(tagging, ensure_ascii=False, separators=(",", ":"))


def to_messages(row: dict) -> list[dict]:
    # Matches app.py's tag_dishes() runtime prompt format exactly:
    #   f"Dish: {d.translation or d.original}. {d.description or ''}"
    # so the trained adapter sees the same distribution at inference it saw
    # in training.
    desc = row.get("description") or ""
    prompt = f"Dish: {row['dish_name']}. {desc}".rstrip()
    if not prompt.endswith("."):
        prompt += "."
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": to_tagging_json(row)},
    ]


def build_dataset() -> dict:
    rows = load_rows()
    print(f"{len(rows)} rows loaded from {DATA_PATH}")

    rows = attach_dish_family(rows)
    n_families = len({(r["_cuisine"], r["_seq"]) for r in rows})
    print(f"{n_families} distinct dish families recovered (expect {15 * 150} = 2250)")

    card = prep_dataset(
        rows,
        entity_key_fn=lambda r: f"{r['_cuisine']}::{r['_seq']}",
        to_messages_fn=to_messages,
        out_dir=OUT_DIR,
        label_key="cuisine",
    )

    # Extra honesty fields beyond prep_dataset's generic card: per-cuisine
    # dish-family counts (not just row counts) and the allergen/spice/veg
    # vocabulary the model is expected to learn, so eval/dataset_card.json
    # is self-describing without cross-referencing gen_dishes.py.
    card["n_dish_families"] = n_families
    card["cuisines"] = CUISINES
    card["allergens"] = ALLERGENS
    card["variants"] = ["name_only", "name_with_description", "original_script_with_translation"]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return card


if __name__ == "__main__":
    print(json.dumps(build_dataset(), indent=2, ensure_ascii=False))
