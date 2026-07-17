"""3-way benchmark for the dish -> cuisine/allergen tagger: base model vs
base+LoRA vs Claude teacher. See MENU_DECODER_SPEC.md §5.

Unlike the sibling projects' single-label classifiers, this is a
GENERATIVE STRUCTURED-OUTPUT task -- the model must emit a full DishTags
JSON (cuisine + 14 allergen calls + spice + vegetarian/vegan), not one
token. The metric that matters per spec §5 is **allergen recall on
"contains"** (a missed allergen is dangerous) and the **false-safe rate**
(model says "not_indicated" when the true label is "contains" -- the
specific dangerous failure mode), reported PER ALLERGEN, not just
averaged. Cuisine accuracy / spice tolerance / veg agreement are the
secondary columns, same 3-way pattern as Cellar Scanner and Receipt
Auditor's benchmarks.

Full 303-row held-out (by dish-family) test set is run in full for
base/LoRA (mlx_lm.generate reloads the model from disk every call but the
set is small enough to be tractable, same call as Receipt Auditor's full
220-row run). Claude teacher runs on a 100-row subsample of that same set
(disclosed, not hidden) to keep the benchmark runnable in one sitting --
same convention as Cellar Scanner/Receipt Auditor.

TWO REAL BUGS FOUND LIVE, in this exact order, during this benchmark's
own run -- both about the same root cause, the second one caught only
because the first "fix" made the numbers *obviously* worse rather than
better (a bigger implausibility than the one it was chasing):

1. The first Claude pass used `llm.generate(..., tier="smart")`, which
   shells out to the `claude` CLI with the default working directory
   (this repo). `claude -p` is agentic by default -- given the
   deliberately terse "Dish: X." prompt (matching the LoRA's actual
   runtime prompt, no schema spelled out), several calls used their
   file-access tools to go read THIS REPO'S OWN `training/prep_tagger.py`
   / `scripts/gen_dishes.py` / `src/decoder.py` mid-call to infer the
   target schema before answering -- e.g. one raw response opened with
   "I don't need the tool for this -- I already have the exact target
   schema (`training/prep_tagger.py`'s `to_tagging_json`...)". That's not
   a blind zero-shot comparison anymore -- it's a system peeking at its
   own training pipeline, breaking the "all three systems see the
   identical blind prompt" premise this project family's benchmarks
   (Cellar Scanner, Receipt Auditor) rely on for a fair fight. 8 of the
   first 100 responses showed this.
2. The obvious-looking fix -- `claude -p --tools ""` to disable all tool
   access -- made things categorically WORSE, not better: nearly every
   response degenerated into fabricated/narrated tool-call transcripts
   ("Tool: bash", "1 tool used", even a hallucinated claim of "prompt
   injection" tool-output tampering) instead of an answer, because the
   CLI's system prompt still primes it to behave like an agentic coding
   assistant even with no tools actually available to execute. Parse-OK
   collapsed to near zero and every accuracy metric fell to ~0% -- an
   unmissable red flag (the exact "a benchmark number that contradicts
   the obvious prior is a bug signal, not a finding" lesson from
   AI_GAP_PROJECTS_ROADMAP.md §8.5), caught before it was ever written to
   eval/benchmark.md. The real fix: run `claude -p` with `cwd` set to a
   directory OUTSIDE this repo (this project's scratchpad), keeping tools
   enabled (so no roleplay-without-a-backend failure mode) but removing
   any file worth reading -- verified live against the exact two prompts
   that triggered problem #1 above, both came back clean. `run_claude`
   below uses a direct subprocess call with that `cwd` (bypassing
   `llm.py`'s `_generate_cli`, which has no passthrough for a custom
   `cwd`) rather than the shared wrapper, specifically for this
   benchmark-critical call.

A THIRD thing found live, deliberately NOT filed as a bug -- it's a real,
on-thesis finding about what fine-tuning actually buys you. Even with both
CLI bugs above fixed, `claude_teacher`'s bare-prompt numbers still read as
near-zero across cuisine/spice/veg/allergen fields, with parse rate high
(~90%+). Inspecting the raw generations shows why: given the literal bare
prompt `"Dish: X."` with NO schema described anywhere, Claude answers a
different, entirely reasonable question -- "tell me about this dish" --
and returns valid JSON with its OWN field names
(`dish_name_arabic`/`region`/`main_ingredients`/`flavor_profile`, etc.),
not `cuisine`/`allergen_calls`/`spice`/`vegetarian`/`vegan`. It parses
(hence high Parse OK) but scores ~0% on every derived metric BY
CONSTRUCTION, because none of its keys match what's being looked for. The
untrained base model's ~0% Parse OK is the same root cause from the other
direction -- it has no idea what format to produce either, so it free-
associates prose instead of JSON. This is genuinely the whole point of
the fine-tune: **the LoRA model needs zero schema in its prompt because
the schema is baked into its weights; a zero-shot system needs the full
14-allergen output contract spelled out in every single call just to
attempt the task.** That's a context-engineering/token-optimization
result worth reporting on its own terms, not hiding behind one number.
So this benchmark reports Claude TWICE: `claude_teacher` (the bare
prompt, identical to what LoRA/base see -- shows the fine-tune's real
prompt-compression value) and `claude_teacher_schema_primed` (the same
prompt with the exact output contract appended, i.e. `src/decoder.py`'s
own `TAGGING_SYSTEM` rubric plus a single-dish JSON format description --
no answer key, no training data, just the schema) so Claude also gets a
fair shot at showing what it actually knows about cuisines and allergens
once told the format it should use.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "training" / "lora_harness"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gen_dishes import ALLERGENS, CUISINES  # noqa: E402
from decoder import TAGGING_SYSTEM  # noqa: E402

# The schema-primed prompt reuses the app's REAL production tagging system
# prompt (src/decoder.py's TAGGING_SYSTEM -- the same asymmetric rubric
# used both for gen_dishes.py's teacher and app.py's Gemini-fallback path)
# plus a single-dish (not the app's multi-dish batch) output-format
# description. This is deliberately just the output CONTRACT, no answer
# key or training examples -- Claude still has to know the actual cuisine/
# allergen facts on its own.
SCHEMA_PRIMED_FORMAT = (
    "Respond with ONLY JSON for this one dish: {\"cuisine\": \"...\", "
    "\"allergen_calls\": [{\"allergen\": \"...\", \"level\": \"contains|may|not_indicated\"}, "
    "...one entry per: gluten, crustaceans, eggs, fish, peanuts, soy, milk, nuts, celery, "
    "mustard, sesame, sulphites, lupin, molluscs], \"spice\": 0-3, "
    "\"vegetarian\": \"yes|no|unclear\", \"vegan\": \"yes|no|unclear\"}. No prose, no code fences."
)
SCHEMA_PRIMED_PREAMBLE = TAGGING_SYSTEM + "\n\n" + SCHEMA_PRIMED_FORMAT

TEST_PATH = REPO_ROOT / "data" / "lora" / "test.jsonl"
EVAL_DIR = REPO_ROOT / "eval"
ADAPTER_PATH = REPO_ROOT / "training" / "adapters"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CLAUDE_SUBSAMPLE = 100  # disclosed subsample, not hidden -- keeps the run tractable
MAX_TOKENS = 300  # true target is ~204 tokens (measured); headroom for all 3 systems, same budget each


def load_test_set() -> list[dict]:
    """test.jsonl is written in dish-family iteration order (cuisine, then
    seq, then variant) by prep_dataset -- NOT randomly ordered. Shuffling
    with a fixed seed before any subsampling avoids the exact bug Cellar
    Scanner's confusion-matrix figure caught (AI_GAP_PROJECTS_ROADMAP.md
    §8.5): taking the first N rows unshuffled would bias Claude's subsample
    toward whichever cuisines happen to sort first (Italian, Japanese...)
    rather than a representative spread across all 15.
    """
    rows = []
    for line in TEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = rec["messages"][0]["content"]
        true_tagging = json.loads(rec["messages"][1]["content"])
        rows.append({"prompt": prompt, "true": true_tagging})
    random.Random(42).shuffle(rows)
    return rows


def extract_json_body(text: str) -> str:
    """mlx_lm.generate wraps its answer as '==========\\n<answer>\\n==========\\n
    Prompt: ...' -- the generated text is the middle segment. Claude's
    output is requested json_only (fences already stripped by llm.py)."""
    if "==========" in text:
        parts = text.split("==========")
        if len(parts) >= 2:
            return parts[1].strip()
    return text.strip()


def extract_json_object(text: str) -> str | None:
    """Find the first balanced {...} object in text (robust to stray
    leading/trailing prose or truncated generations that ran out of
    max-tokens mid-object)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None  # never closed -- truncated generation


def parse_tagging(raw_text: str) -> dict | None:
    body = extract_json_body(raw_text)
    obj_str = extract_json_object(body)
    if obj_str is None:
        return None
    try:
        parsed = json.loads(obj_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    # Defensive unwrap: an ungrounded zero-shot system given a bare "Dish: X."
    # prompt (no schema shown) can guess the app's OTHER JSON shape --
    # src/decoder.py's build_tagging_prompt asks for {"tags": [{dish_id,
    # cuisine, ...}]} for a whole-menu batch call -- instead of the flat
    # single-dish object this benchmark's training target actually uses.
    # Unwrap it rather than silently scoring every field as missing.
    if "cuisine" not in parsed and isinstance(parsed.get("tags"), list) and parsed["tags"]:
        first = parsed["tags"][0]
        if isinstance(first, dict):
            parsed = first
    return parsed


# Unambiguous aliases within THIS 15-cuisine taxonomy only -- Claude, given
# no cuisine list, sometimes answers with a more specific regional name that
# is a genuine synonym for one (and only one) of the 15 buckets, e.g.
# "Sichuan" for "Chinese-Sichuan". These three are safe to fold in because
# there's no other Chinese/American bucket in CUISINES they could collide
# with. Broader demonym cases (e.g. "Lebanese"/"Levantine" for the single
# "Middle-Eastern" bucket, which covers many countries) are deliberately
# NOT auto-matched here -- that would require this script to decide on
# Claude's behalf which countries count as "Middle-Eastern", a judgment
# call outside a benchmark script's job. Those show up as real, disclosed
# mismatches instead of being silently folded in.
_CUISINE_ALIASES = {
    "Chinese-Sichuan": ["sichuan", "szechuan"],
    "Chinese-Cantonese": ["cantonese", "guangdong"],
    "American-diner": ["american diner", "american"],
}


def normalize_cuisine(value: str | None) -> str | None:
    """Match against the known 15-cuisine vocabulary (case-insensitive
    substring, longest-first) -- same fix pattern as the sibling projects'
    parsers (AI_GAP_PROJECTS_ROADMAP.md §8.5), robust to a model wrapping
    the answer in extra words. Also checks the small unambiguous alias
    table above before giving up."""
    if not value:
        return None
    text = str(value).lower()
    sorted_cuisines = sorted(CUISINES, key=len, reverse=True)
    match = next((c for c in sorted_cuisines if c.lower() in text), None)
    if match:
        return match
    for cuisine, aliases in _CUISINE_ALIASES.items():
        if any(a in text for a in aliases):
            return cuisine
    return None


def allergen_levels(parsed: dict | None) -> dict[str, str]:
    """Returns {allergen: level}, defaulting missing/unparseable allergens
    to 'not_indicated' -- the FAIL-SAFE-BREAKING default (a system that
    can't parse its own output should be scored as if it warned about
    nothing, since that's what a real caller would see: no signal is
    exactly as dangerous as a wrong 'not_indicated'). This makes parse
    failures show up as recall misses, not get silently excluded."""
    if not parsed:
        return {a: "not_indicated" for a in ALLERGENS}
    calls = parsed.get("allergen_calls", [])
    out: dict[str, str] = {}
    if isinstance(calls, list):
        for c in calls:
            if isinstance(c, dict) and c.get("allergen") in ALLERGENS and c.get("level") in ("contains", "may", "not_indicated"):
                out[c["allergen"]] = c["level"]
    return {a: out.get(a, "not_indicated") for a in ALLERGENS}


def run_mlx(prompts: list[str], adapter_path: str | None) -> tuple[list[str], list[float]]:
    outputs, latencies = [], []
    for i, prompt in enumerate(prompts):
        args = ["mlx_lm.generate", "--model", BASE_MODEL, "--prompt", prompt, "--max-tokens", str(MAX_TOKENS)]
        if adapter_path:
            args += ["--adapter-path", adapter_path]
        start = time.time()
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
            out = proc.stdout.strip()
        except subprocess.TimeoutExpired:
            out = ""
        latencies.append(time.time() - start)
        outputs.append(out)
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(prompts)}")
    return outputs, latencies


BLIND_CWD = Path(tempfile.gettempdir()) / "menu-decoder-bench-scratch"


def run_claude_blind(prompt: str, *, preamble: str | None = None, timeout_s: int = 90) -> str:
    """Same claude -p CLI llm.py._generate_cli uses, but run with `cwd` set
    to a directory OUTSIDE this repo (see the module docstring's two-bug
    writeup): a bare "Dish: X." prompt with no schema is ambiguous enough
    that claude -p's default agentic tool access will sometimes go read
    THIS repo's own training code to infer the answer format when run from
    it. Disabling tools outright (`--tools ""`) was tried first and made
    things worse (the CLI still narrates fake tool calls with no backend
    to run them against) -- moving the cwd elsewhere keeps tools available
    (no roleplay-without-a-backend failures) while removing anything worth
    reading. llm.generate() has no passthrough for a custom cwd, so this
    bypasses it for this one benchmark-critical call.

    `preamble`, when given, is prepended (e.g. SCHEMA_PRIMED_PREAMBLE) --
    used for the claude_teacher_schema_primed column, see module docstring."""
    full_prompt = (preamble + "\n\n" if preamble else "") + prompt + \
        "\n\nRespond with ONLY valid JSON, no prose, no code fences."
    args = ["claude", "-p", "--model", "sonnet", "--output-format", "json"]
    BLIND_CWD.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(args, input=full_prompt, capture_output=True, text=True,
                           encoding="utf-8", timeout=timeout_s, cwd=str(BLIND_CWD))
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed (exit {proc.returncode}): {proc.stderr[:300]}")
    parsed = json.loads(proc.stdout.strip())
    text = parsed.get("result", proc.stdout.strip())
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def run_claude(prompts: list[str], *, preamble: str | None = None) -> tuple[list[str], list[float]]:
    outputs, latencies = [], []
    for i, prompt in enumerate(prompts):
        start = time.time()
        try:
            outputs.append(run_claude_blind(prompt, preamble=preamble))
        except Exception as e:
            outputs.append("")
            print(f"  claude call {i} failed: {e}")
        latencies.append(time.time() - start)
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(prompts)}")
    return outputs, latencies


def compute_metrics(raw_outputs: list[str], truths: list[dict]) -> dict:
    parsed_list = [parse_tagging(o) for o in raw_outputs]
    n = len(truths)
    parse_success = sum(1 for p in parsed_list if p is not None) / n if n else 0.0

    # cuisine accuracy
    cuisine_hits = 0
    for p, t in zip(parsed_list, truths):
        pred_cuisine = normalize_cuisine(p.get("cuisine") if p else None)
        if pred_cuisine == t["cuisine"]:
            cuisine_hits += 1
    cuisine_acc = cuisine_hits / n if n else 0.0

    # spice +/-1 tolerance
    spice_hits = 0
    for p, t in zip(parsed_list, truths):
        pred_spice = p.get("spice") if p else None
        try:
            pred_spice = int(pred_spice)
        except (TypeError, ValueError):
            pred_spice = None
        if pred_spice is not None and abs(pred_spice - t["spice"]) <= 1:
            spice_hits += 1
    spice_tolerance_agreement = spice_hits / n if n else 0.0

    # vegetarian / vegan exact agreement
    veg_hits = sum(1 for p, t in zip(parsed_list, truths) if p and p.get("vegetarian") == t["vegetarian"])
    vegan_hits = sum(1 for p, t in zip(parsed_list, truths) if p and p.get("vegan") == t["vegan"])
    vegetarian_agreement = veg_hits / n if n else 0.0
    vegan_agreement = vegan_hits / n if n else 0.0

    # per-allergen recall-on-contains + false-safe rate (the metric that matters)
    per_allergen = {}
    for allergen in ALLERGENS:
        n_contains = 0
        recall_hits = 0
        false_safe = 0
        for p, t in zip(parsed_list, truths):
            true_levels = {c["allergen"]: c["level"] for c in t["allergen_calls"]}
            if true_levels.get(allergen) != "contains":
                continue
            n_contains += 1
            pred_level = allergen_levels(p)[allergen]
            if pred_level == "contains":
                recall_hits += 1
            if pred_level == "not_indicated":
                false_safe += 1
        per_allergen[allergen] = {
            "n_contains": n_contains,
            "recall_on_contains": round(recall_hits / n_contains, 4) if n_contains else None,
            "false_safe_rate": round(false_safe / n_contains, 4) if n_contains else None,
        }

    total_contains = sum(v["n_contains"] for v in per_allergen.values())
    total_recall_hits = sum(round(v["recall_on_contains"] * v["n_contains"]) for v in per_allergen.values() if v["n_contains"])
    total_false_safe = sum(round(v["false_safe_rate"] * v["n_contains"]) for v in per_allergen.values() if v["n_contains"])
    macro_recall = sum(v["recall_on_contains"] for v in per_allergen.values() if v["recall_on_contains"] is not None) / \
        sum(1 for v in per_allergen.values() if v["recall_on_contains"] is not None)
    macro_false_safe = sum(v["false_safe_rate"] for v in per_allergen.values() if v["false_safe_rate"] is not None) / \
        sum(1 for v in per_allergen.values() if v["false_safe_rate"] is not None)

    return {
        "n": n,
        "parse_success_rate": round(parse_success, 4),
        "cuisine_acc": round(cuisine_acc, 4),
        "spice_tolerance_agreement": round(spice_tolerance_agreement, 4),
        "vegetarian_agreement": round(vegetarian_agreement, 4),
        "vegan_agreement": round(vegan_agreement, 4),
        "allergen_recall_on_contains_macro": round(macro_recall, 4),
        "allergen_false_safe_rate_macro": round(macro_false_safe, 4),
        "allergen_recall_on_contains_micro": round(total_recall_hits / total_contains, 4) if total_contains else None,
        "allergen_false_safe_rate_micro": round(total_false_safe / total_contains, 4) if total_contains else None,
        "per_allergen": per_allergen,
    }


def main():
    raw_outputs_path = EVAL_DIR / "raw_outputs.json"
    rows = load_test_set()
    print(f"Held-out test set: {len(rows)} rows (dish-family split)")

    prompts = [r["prompt"] for r in rows]
    truths = [r["true"] for r in rows]
    claude_rows = rows[:CLAUDE_SUBSAMPLE]
    claude_truths = [r["true"] for r in claude_rows]

    # Cache raw generations to disk PER SYSTEM, not just at the end -- a
    # mid-run claude -p rate-limit failure must not discard already-
    # finished base/LoRA outputs (real incident, AI_GAP_PROJECTS_ROADMAP.md
    # §8.5: ~40 minutes of local compute was lost this exact way before the
    # fix). Each block below loads its cached result if present, else
    # generates AND immediately saves before moving to the next system.
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cached = json.loads(raw_outputs_path.read_text()) if raw_outputs_path.exists() else {}

    def _save(key: str, value) -> None:
        cached[key] = value
        raw_outputs_path.write_text(json.dumps(cached, ensure_ascii=False))

    if "base_out" in cached:
        print("Reusing cached base model outputs")
        base_out, base_lat = cached["base_out"], cached["base_lat"]
    else:
        print(f"Running base model on {len(prompts)} prompts...")
        base_out, base_lat = run_mlx(prompts, adapter_path=None)
        _save("base_out", base_out)
        _save("base_lat", base_lat)

    if "lora_out" in cached:
        print("Reusing cached base+LoRA outputs")
        lora_out, lora_lat = cached["lora_out"], cached["lora_lat"]
    else:
        if not ADAPTER_PATH.exists():
            raise SystemExit(f"{ADAPTER_PATH} not found -- run training/lora_harness/train.sh first.")
        print(f"Running base+LoRA on {len(prompts)} prompts...")
        lora_out, lora_lat = run_mlx(prompts, adapter_path=str(ADAPTER_PATH))
        _save("lora_out", lora_out)
        _save("lora_lat", lora_lat)

    if "claude_out" in cached:
        print("Reusing cached Claude outputs (bare prompt)")
        claude_out, claude_lat = cached["claude_out"], cached["claude_lat"]
    else:
        print(f"Running Claude teacher, BARE prompt (smart tier) on a {len(claude_rows)}-row subsample...")
        claude_out, claude_lat = run_claude([r["prompt"] for r in claude_rows])
        _save("claude_out", claude_out)
        _save("claude_lat", claude_lat)

    if "claude_schema_out" in cached:
        print("Reusing cached Claude outputs (schema-primed)")
        claude_schema_out, claude_schema_lat = cached["claude_schema_out"], cached["claude_schema_lat"]
    else:
        print(f"Running Claude teacher, SCHEMA-PRIMED prompt (smart tier) on a {len(claude_rows)}-row subsample...")
        claude_schema_out, claude_schema_lat = run_claude([r["prompt"] for r in claude_rows], preamble=SCHEMA_PRIMED_PREAMBLE)
        _save("claude_schema_out", claude_schema_out)
        _save("claude_schema_lat", claude_schema_lat)

    results = []
    for name, outs, truth_set, lat in [
        ("base", base_out, truths, base_lat),
        ("lora", lora_out, truths, lora_lat),
        ("claude_teacher", claude_out, claude_truths, claude_lat),
        ("claude_teacher_schema_primed", claude_schema_out, claude_truths, claude_schema_lat),
    ]:
        metrics = compute_metrics(outs, truth_set)
        results.append({
            "system": name,
            **{k: v for k, v in metrics.items() if k != "per_allergen"},
            "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else 0,
            "cost_per_1k_calls_usd": 0 if name in ("base", "lora") else "Max subscription (no per-call API cost)",
            "per_allergen": metrics["per_allergen"],
        })

    (EVAL_DIR / "benchmark.json").write_text(json.dumps(results, indent=2))

    # --- sanity check before writing anything human-facing ---
    # NOTE: claude_teacher (bare prompt) is EXPECTED to score near-zero on
    # every derived metric by construction -- it answers a different
    # question when given no schema (see module docstring's 3rd finding).
    # Sanity checks below compare against claude_teacher_schema_primed
    # instead, since that's the column meant to reflect what Claude
    # actually knows about cuisines/allergens.
    base_r = next(r for r in results if r["system"] == "base")
    lora_r = next(r for r in results if r["system"] == "lora")
    claude_r = next(r for r in results if r["system"] == "claude_teacher")
    claude_schema_r = next(r for r in results if r["system"] == "claude_teacher_schema_primed")
    warnings = []
    if claude_schema_r["cuisine_acc"] < base_r["cuisine_acc"]:
        warnings.append(f"claude_teacher_schema_primed cuisine_acc ({claude_schema_r['cuisine_acc']:.1%}) < base ({base_r['cuisine_acc']:.1%}) -- implausible, check parser.")
    if lora_r["allergen_recall_on_contains_macro"] < base_r["allergen_recall_on_contains_macro"]:
        warnings.append(f"lora allergen recall ({lora_r['allergen_recall_on_contains_macro']:.1%}) < base ({base_r['allergen_recall_on_contains_macro']:.1%}) -- fine-tune should beat the untrained floor on its own training distribution; check for a bug before trusting this.")
    if claude_r["parse_success_rate"] < 0.5 and claude_r["cuisine_acc"] < 0.05:
        print(f"NOTE (not a warning, see module docstring): claude_teacher bare-prompt cuisine_acc is "
              f"{claude_r['cuisine_acc']:.1%} despite {claude_r['parse_success_rate']:.1%} parse-OK -- expected, "
              f"since it's answering a different, schema-less question. See claude_teacher_schema_primed instead.")
    for w in warnings:
        print(f"SANITY WARNING: {w}")
    (EVAL_DIR / "sanity_warnings.json").write_text(json.dumps(warnings, indent=2))

    md_lines = ["# Menu Decoder -- Dish Tagger Benchmark", "",
                f"Full held-out test set: {len(rows)} rows, held out by **dish family** (all 3 "
                f"variants of a dish -- name-only, name+description, original-script+translation "
                f"-- share one side of the split; see `eval/dataset_card.json`). base/LoRA "
                f"evaluated on the full {len(rows)}-row set; Claude teacher on a separate "
                f"{CLAUDE_SUBSAMPLE}-row subsample of that same set (disclosed, not hidden -- "
                "mlx_lm.generate reloads the model from disk on every call, and `claude -p` runs "
                "at Max-subscription latency, so a full 3-way run at full size would take hours).",
                "",
                "**Human slice not yet available** -- Jeremy hasn't spot-checked the stratified "
                "60-dish sample yet (MENU_DECODER_SPEC.md §5). The numbers below compare the "
                "synthetic teacher labels against base/LoRA/Claude; they are not validated "
                "against a human ground truth, and the spec names this circularity risk "
                "explicitly (Claude both labeled the training data and is a benchmark column).",
                "",
                "## Why Claude appears twice", "",
                "`claude_teacher` gets the **identical bare prompt** LoRA and base see "
                "(`\"Dish: X.\"`, no schema described anywhere) -- same \"all systems see the "
                "identical blind prompt\" rule this project family's benchmarks use throughout. "
                "But with no schema given, Claude reasonably answers a different question "
                "(\"tell me about this dish\") and returns valid JSON with its own field names "
                "(`dish_name_arabic`, `region`, `main_ingredients`, `flavor_profile`, etc.) "
                "instead of `cuisine`/`allergen_calls`/`spice`/`vegetarian`/`vegan` -- it parses "
                "(high Parse OK) but scores ~0% on every derived metric BY CONSTRUCTION, because "
                "none of its keys match what's being scored. This is not a parsing bug (verified "
                "by reading the raw generations directly); it's the real, measured cost of a "
                "prompt with no schema in it. `claude_teacher_schema_primed` gets the same dish "
                "prompt plus the exact output contract appended (`src/decoder.py`'s own "
                "`TAGGING_SYSTEM` rubric + a single-dish JSON format description -- no answer "
                "key, no training examples) so Claude gets a fair shot at showing what it "
                "actually knows about cuisines and allergens. Reading both rows together is the "
                "point: **the LoRA model needs zero schema in its prompt because the schema is "
                "baked into its weights; a zero-shot system needs the full 14-allergen contract "
                "spelled out in every call just to attempt the task** -- a measured context-"
                "engineering/token-optimization result, not just an accuracy comparison.",
                "",
                "## Headline metrics", "",
                "| System | N | Parse OK | Cuisine acc | Spice ±1 | Veg agree | Vegan agree | "
                "Allergen recall (contains, macro) | False-safe rate (macro) | Latency (s/item) | Cost/1k |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        md_lines.append(
            f"| {r['system']} | {r['n']} | {r['parse_success_rate']:.1%} | {r['cuisine_acc']:.1%} | "
            f"{r['spice_tolerance_agreement']:.1%} | {r['vegetarian_agreement']:.1%} | "
            f"{r['vegan_agreement']:.1%} | {r['allergen_recall_on_contains_macro']:.1%} | "
            f"{r['allergen_false_safe_rate_macro']:.1%} | {r['mean_latency_s']} | {r['cost_per_1k_calls_usd']} |"
        )

    md_lines += ["", "## Per-allergen recall-on-`contains` and false-safe rate", "",
                 "The safety-critical table -- **false-safe rate** is the fraction of true "
                 "\"contains\" cases where the system said \"not_indicated\" (the dangerous "
                 "miss; a \"may\" call is imprecise but still warns, so it does not count as "
                 "false-safe). Blank cells mean the test set had zero true-`contains` examples "
                 "for that allergen in that sample. `claude_teacher` (bare prompt) is omitted "
                 "from this table -- its allergen field is essentially never populated with the "
                 "right keys (see \"Why Claude appears twice\" above), so a per-allergen recall "
                 "number for it would be noise, not signal; `claude_teacher_schema_primed` "
                 "(labeled `claude` below) is the fair comparison. Full numbers for all 4 systems "
                 "are in `eval/benchmark.json` regardless.", "",
                 "| Allergen | True contains (N) | base recall | base false-safe | lora recall | "
                 "lora false-safe | claude (schema-primed) recall | claude (schema-primed) false-safe |",
                 "|---|---|---|---|---|---|---|---|"]

    def _fmt(v):
        return f"{v:.1%}" if v is not None else "-"

    for allergen in ALLERGENS:
        b, l, c = base_r["per_allergen"][allergen], lora_r["per_allergen"][allergen], claude_schema_r["per_allergen"][allergen]
        md_lines.append(
            f"| {allergen} | {b['n_contains']} | {_fmt(b['recall_on_contains'])} | {_fmt(b['false_safe_rate'])} | "
            f"{_fmt(l['recall_on_contains'])} | {_fmt(l['false_safe_rate'])} | "
            f"{_fmt(c['recall_on_contains'])} | {_fmt(c['false_safe_rate'])} |"
        )

    if warnings:
        md_lines += ["", "## Sanity-check warnings", ""]
        md_lines += [f"- {w}" for w in warnings]

    md_lines += ["", "## Known limitation: no human-validated ground truth yet", "",
                 "All four rows are compared against Claude-teacher-generated synthetic labels "
                 "(`scripts/gen_dishes.py`, `tier=\"smart\"`), not human judgment. The spec "
                 "(MENU_DECODER_SPEC.md §5) calls for a mandatory ~60-dish human-spot-check slice "
                 "specifically because Claude is both the label source and a benchmark column -- "
                 "a same-source comparison can look better than it should. That slice requires "
                 "Jeremy personally and has not been run yet; until it lands, treat "
                 "`claude_teacher_schema_primed`'s numbers here as an internal-consistency check "
                 "(does Claude agree with itself under a blind prompt), not independent "
                 "validation."]

    (EVAL_DIR / "benchmark.md").write_text("\n".join(md_lines))
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
