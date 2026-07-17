"""Menu Decoder -- Gradio app. Runs locally and as an HF Space.

Menu photo + spoken dietary preferences -> every dish translated, allergen-
badged (fail-safe: no green checkmark ever), and a safe-list-filtered "what
I'd order" pick, with dish-name TTS. See MENU_DECODER_SPEC.md §4.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

import audio
import vision
from cache import FileCache, SemanticCache
from decoder import build_recommend_prompt, build_tagging_prompt
from guardrails import GuardrailError, Refusal, generate_validated
from safelist import ALLERGEN_BADGE, filter_safe_dishes, is_safe_for_brief
from schemas import DietBrief, Dish, DishTags, DishTagsBatch, Menu

DATA_DIR = Path(__file__).resolve().parent / "data"
translation_cache = SemanticCache(DATA_DIR / "translation_cache.db", similarity_threshold=0.94)
audio_cache = FileCache(DATA_DIR / "audio_cache")
ADAPTER_PATH = Path(__file__).resolve().parent / "training" / "adapters"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def extract_menu(image) -> tuple[Menu | None, str]:
    if image is None:
        return None, "Upload a menu photo."

    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    result = vision.extract(
        image, Menu,
        task_prompt="language, and sections (name, dishes: original text, translation, "
                    "description if present, price if present; set unreadable=true for any "
                    "line you genuinely cannot read rather than skipping it)",
        domain_description="a restaurant menu",
        min_confidence=0.5,
        # menu language is unknown ahead of time -- cast a wide net across
        # every language pack installed (setup_ocr.sh). Combining many packs
        # in one tesseract pass costs some accuracy vs. a single known
        # language, but a menu in the wrong script would otherwise OCR to
        # near-nothing.
        ocr_langs="eng+jpn+tha+ita+chi_sim+chi_tra+kor+vie+spa+fra+deu+por+ell",
    )
    if isinstance(result, Refusal):
        return None, f"⚠ {result.user_message}"

    # completeness check: every dish must be translated or explicitly flagged
    dishes = result.all_dishes()
    incomplete = [d for d in dishes if not d.unreadable and not d.translation]
    completeness = 1 - (len(incomplete) / len(dishes)) if dishes else 1.0
    note = f" ({len(incomplete)} line(s) need review)" if incomplete else ""
    return result, f"✓ {len(dishes)} dishes found, {completeness:.0%} translation completeness{note}"


def _extract_lora_json(raw_stdout: str) -> dict:
    """mlx_lm.generate's CLI wraps its answer as
    '==========\\n<json>\\n==========\\nPrompt: ...' -- BUG FOUND & FIXED:
    the previous version called DishTags.model_validate_json(proc.stdout.strip())
    directly on that whole banner-wrapped blob, which can never parse as JSON,
    AND DishTags requires a `dish_id` field that training/prep_tagger.py's
    target schema never includes (dish_id is assigned per-menu at extraction
    time, not something the tagger predicts) -- so every real call silently
    fell into the `except Exception` fail-safe-default branch, meaning the
    trained LoRA adapter was never actually used even once it existed. Fixed
    by extracting the generated JSON body and adding dish_id back in from the
    Dish being tagged (see tag_dishes below)."""
    body = raw_stdout.split("==========")[1].strip() if "==========" in raw_stdout else raw_stdout.strip()
    start, end = body.find("{"), body.rfind("}")
    return json.loads(body[start:end + 1])


def tag_dishes(menu: Menu) -> list[DishTags]:
    dishes = menu.all_dishes()
    if ADAPTER_PATH.exists():
        tags = []
        for d in dishes:
            prompt = f"Dish: {d.translation or d.original}. {d.description or ''}"
            try:
                proc = subprocess.run(
                    ["mlx_lm.generate", "--model", BASE_MODEL, "--adapter-path", str(ADAPTER_PATH),
                     "--prompt", prompt, "--max-tokens", "300"],
                    capture_output=True, text=True, timeout=30,
                )
                raw = _extract_lora_json(proc.stdout)
                tags.append(DishTags(dish_id=d.id, **raw))
            except Exception:
                tags.append(_default_unsure_tags(d.id))
        return tags

    # Gemini path (Space / no adapter yet): one batched call for the whole menu.
    prompt, _ = build_tagging_prompt(dishes)
    expected_ids = {d.id for d in dishes}

    def _verify_coverage(batch: DishTagsBatch) -> list[str]:
        got_ids = {t.dish_id for t in batch.tags}
        missing = expected_ids - got_ids
        return [f"missing tags for dish ids: {sorted(missing)}"] if missing else []

    try:
        batch = generate_validated(prompt, DishTagsBatch, verifier=_verify_coverage, max_retries=1)
        tags_by_id = {t.dish_id: t for t in batch.tags}
        return [tags_by_id.get(d.id) or _default_unsure_tags(d.id) for d in dishes]
    except GuardrailError:
        return [_default_unsure_tags(d.id) for d in dishes]


def _default_unsure_tags(dish_id: str) -> DishTags:
    """Fail-safe default when tagging fails outright: mark every allergen
    'may' (never a false-safe default) and dietary status 'unclear'."""
    from schemas import AllergenCall
    allergens = ["gluten", "crustaceans", "eggs", "fish", "peanuts", "soy", "milk",
                 "nuts", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs"]
    return DishTags(
        dish_id=dish_id, cuisine="unknown",
        allergen_calls=[AllergenCall(allergen=a, level="may") for a in allergens],
        spice=1, vegetarian="unclear", vegan="unclear",
    )


def process_menu_and_brief(menu: Menu | None, diets_text: str, allergens_text: str, spice_max: int | None):
    if menu is None:
        return "⚠ Extract a menu first.", None

    diets = [d.strip() for d in diets_text.split(",") if d.strip()] if diets_text else []
    allergens = [a.strip() for a in allergens_text.split(",") if a.strip()] if allergens_text else []
    brief = DietBrief(diets=diets, allergens=allergens, spice_max=spice_max)

    dishes = menu.all_dishes()
    tags_list = tag_dishes(menu)
    tags_by_id = {t.dish_id: t for t in tags_list}

    rows = []
    for d in dishes:
        tags = tags_by_id.get(d.id)
        badges = []
        if tags:
            for call in tags.allergen_calls:
                if call.allergen in allergens:
                    badges.append(f"{call.allergen}:{ALLERGEN_BADGE[call.level]}")
        rows.append([d.translation or d.original, d.original, ", ".join(badges) or "-",
                     tags.spice if tags else "?", tags.vegetarian if tags else "?"])

    safe_dishes = filter_safe_dishes([(d, tags_by_id[d.id]) for d in dishes if d.id in tags_by_id], brief)
    picks_summary = f"{len(safe_dishes)} of {len(dishes)} dishes match your preferences."

    return picks_summary, rows


def speak_dish(original_text: str, language: str) -> str | None:
    voice = audio.voice_for(language)
    if not voice:
        return None
    path = audio.speak_cached(original_text, voice, audio_cache)
    return str(path)


with gr.Blocks(title="Menu Decoder") as demo:
    gr.Markdown("# 🍽️ Menu Decoder")
    gr.Markdown("No green checkmark for allergens, ever -- uncertain always reads as ⚠, never ✓.")

    menu_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            menu_image = gr.Image(type="pil", label="Menu photo")
            extract_btn = gr.Button("Read this menu")
            extract_status = gr.Markdown()
            diets_input = gr.Textbox(label="Diets (comma-separated, e.g. vegetarian, vegan)")
            allergens_input = gr.Textbox(label="Allergens to avoid (comma-separated)")
            spice_input = gr.Slider(0, 3, step=1, label="Max spice level", value=3)
            decode_btn = gr.Button("Decode for me", variant="primary")
        with gr.Column():
            picks_output = gr.Markdown()
            dishes_table = gr.Dataframe(headers=["dish", "original", "your allergens", "spice", "vegetarian"])

    dish_name_input = gr.Textbox(label="Type a dish name to hear it pronounced")
    dish_lang_input = gr.Textbox(label="Language code (e.g. ja, th, it)", value="en")
    speak_btn = gr.Button("🔊 Pronounce")
    dish_audio = gr.Audio(label="Pronunciation")

    extract_btn.click(extract_menu, inputs=[menu_image], outputs=[menu_state, extract_status])
    decode_btn.click(process_menu_and_brief, inputs=[menu_state, diets_input, allergens_input, spice_input],
                      outputs=[picks_output, dishes_table])
    speak_btn.click(speak_dish, inputs=[dish_name_input, dish_lang_input], outputs=[dish_audio])

if __name__ == "__main__":
    demo.launch()
