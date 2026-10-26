# Menu Decoder: an allergen badge that can warn but never clear

Menu Decoder reads a photographed menu in any language, translates each dish, and tags it for cuisine, 14 EU allergens, spice and diet. The tagger is Qwen2.5-1.5B with a LoRA adapter that learned the whole output schema, so it needs a one-line prompt. On 303 held-out dishes it returns valid tags every time and catches 69.3% of the allergens a dish contains. Claude given the full schema catches 79.2%. The app is built on the assumption that the tagger will miss some, so it never tells you a dish is safe.

Design and copy follow [these standards](https://github.com/lyhjeremy/lyhjeremy/blob/main/DESIGN_STANDARDS.md).

## The idea

Ask a chatbot to read a menu you can't and tell you what's safe, and you are trusting it to translate correctly and never wave through an allergen it wasn't sure about. The spec (`MENU_DECODER_SPEC.md` §2) starts from an asymmetry: a missed allergen is dangerous, and a false warning costs a question to the waiter. So the app leans toward warning in three layers, each of which holds on its own.

It is one of four apps that each fine-tune a small local model and benchmark it against Claude. The others are [Cellar Scanner](https://github.com/lyhjeremy/cellar-scanner), [Race Day Copilot](https://github.com/lyhjeremy/race-day-copilot) and [Receipt Auditor](https://github.com/lyhjeremy/receipt-auditor).

## Three layers of fail-safe

1. **The rubric.** The synthetic-data generator (`scripts/gen_dishes.py`) and the runtime tagging prompt (`src/decoder.py`) carry the same instruction: answer `contains` when an allergen is a defining or named ingredient, `may` when it is plausible in a typical preparation even if unnamed (sesame oil in many Asian dishes, gluten in most sauces and breading, peanuts in Southeast Asian cooking), and `not_indicated` only where the allergen would be unusual. The sentence doing the work is "when uncertain, prefer `may` over `not_indicated`".
2. **The renderer.** `ALLERGEN_BADGE` in `src/safelist.py` maps `contains` to a cross, `may` to "ask the staff" and `not_indicated` to "not indicated" with an open circle. There is no entry that shows a check mark, so a model that decided to be reassuring has nothing to render it with.
3. **The filter.** `filter_safe_dishes` removes every dish marked `contains` or `may` for any of the user's allergens, and every dish that fails a stated vegetarian or vegan diet, before the recommendation call. The model chooses from the filtered list and can't argue a dish back onto it. Receipt Auditor applies the same split to arithmetic: code decides, the model phrases.

## The fine-tune

Cellar Scanner and Receipt Auditor fine-tune a single-label classifier. This one has to write a whole JSON object per dish: cuisine, a `contains` / `may` / `not_indicated` level for each of 14 allergens, a spice score, and vegetarian and vegan calls.

**Data.** `scripts/gen_dishes.py` generated 6,750 rows overnight with `claude -p`: 15 cuisines, 150 dishes each, three prompt variants per dish (name only, name with description, and original script with translation, such as `"海老柚子味噌漬け丼 (Ebi Yuzu Miso-zuke Don)"`).

**Split.** The spec requires splitting by dish family so a dish's three variants never straddle train and test; otherwise the model could learn a dish's allergens from one variant and get credit on another. The rows have no plain family ID. The generator's `_stable_id(cuisine, seq, variant)` hashes all three into a 16-character `item_id` and stores neither `seq` nor `variant`. So `training/prep_tagger.py` rebuilds the generator's `build_items()` list and matches each `item_id` back to its `(cuisine, seq, variant)`. Grouping by `dish_name` would have failed, since `"Pad Thai"` and `"ผัดไทย (Pad Thai)"` are different strings. All 6,750 rows matched, giving 2,250 families split 90/5/5 into 6,096 train, 351 validation and 303 test rows. The leakage check passed (`eval/dataset_card.json`).

**Training.** LoRA on Qwen2.5-1.5B-Instruct (4-bit) with MLX on Apple Silicon: batch 4, 12 layers, 800 iterations, learning rate 1.5e-4, sequence length 768. I measured the median target at about 204 tokens with the tokenizer before choosing the sequence length. Loss curve: `eval/loss_curve.png`.

## The benchmark

The spec (§5) asks for recall on `contains`, per allergen, because a missed allergen is the dangerous error, and for the false-safe rate: the share of `contains` cases tagged `not_indicated`. A `may` still produces a warning, so it doesn't count as false-safe. `training/bench_tagger.py` reports both per allergen, plus cuisine accuracy, spice within 1, and vegetarian and vegan agreement.

Base and LoRA ran on all 303 test rows. Claude ran on a 100-row subsample, because `mlx_lm.generate` reloads the model on every call and `claude -p` is slow, so a full run of every system would take hours.

| System | N | Valid JSON | Cuisine | Spice ±1 | Veg | Vegan | Allergen recall | False-safe | Latency |
|---|---|---|---|---|---|---|---|---|---|
| Base Qwen2.5-1.5B, bare prompt | 303 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 5.4 s |
| Qwen2.5-1.5B + LoRA, bare prompt | 303 | **100.0%** | **99.3%** | **99.0%** | 94.1% | **96.7%** | 69.3% | 7.3% | 6.7 s |
| Claude, bare prompt | 100 | 94.0% | 43.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 16.5 s |
| Claude, schema in prompt | 100 | 99.0% | 81.0% | 99.0% | **97.0%** | 96.0% | **79.2%** | **1.7%** | 13.4 s |

Claude has two rows on purpose. The bare-prompt row gets exactly what the LoRA model gets, `"Dish: X."`. The schema row adds `src/decoder.py`'s `TAGGING_SYSTEM` rubric and a single-dish JSON format description, with no answers and no examples.

With the bare prompt, Claude answers a sensible but different question, "tell me about this dish", and returns valid JSON with its own keys (`dish_name_arabic`, `region`, `main_ingredients`, `flavor_profile`). It parses, but none of the keys are the ones being scored, so the row is zero. The untrained base model fails for the same reason from the other side: it doesn't know what format to produce and writes prose. Given the schema, Claude reaches 81.0% on cuisine and 79.2% allergen recall with a 1.7% false-safe rate, ahead of the fine-tune on allergen safety.

So the fine-tune doesn't win on allergens. What it buys is that the 14-allergen contract is in the weights: the prompt is one line, and the model runs locally with no per-call cost, while a zero-shot model needs the full rubric in every call to attempt the task at all.

<p align="center">
  <img src="eval/allergen_recall_chart.png" alt="Per-allergen recall (top) and false-safe rate (bottom) for base, LoRA and schema-primed Claude across 14 allergens; the base model's false-safe rate is 1.0 everywhere, and LoRA's is highest on molluscs" width="720">
</p>

A confusion matrix doesn't fit a multi-label task, so the chart shows recall and false-safe rate per allergen for base, LoRA and schema-primed Claude. The bare-prompt Claude row is left out because its allergen fields are never populated. On common allergens the fine-tune is close to Claude: gluten 91.5% against 84.4%, fish 94.4% against 96.0%, nuts 92.7% against 94.1%. It falls behind on rarer ones. Molluscs is the worst case, with 39.1% of mollusc dishes tagged `not_indicated`, then peanuts at 18.2%. On sulphites and celery it has 0% recall and no false-safes, meaning it answered `may` every time. All 14 allergens are in `eval/benchmark.md`.

## Problems found while building the benchmark

**The CLI read the training code.** The first Claude pass ran `claude -p` from the repo directory. The CLI has file tools by default, and some calls read `training/prep_tagger.py`, `scripts/gen_dishes.py` or `src/decoder.py` to work out the target schema before answering. One response opened with "I don't need the tool for this. I already have the exact target schema (`training/prep_tagger.py`'s `to_tagging_json`...)". 8 of the first 100 responses did this.

**Turning tools off made it worse.** The obvious fix, `claude -p --tools ""`, turned most responses into narrated tool calls ("Tool: bash", "1 tool used") and, once, a claim that a `grep` result had been manipulated by a prompt injection. The CLI's system prompt still sets it up as a coding agent even with no tools attached. Every metric dropped to about 0%, which was implausible enough that I checked it before writing anything to `eval/benchmark.md`. The working fix runs `claude -p` with its working directory set to a scratch folder outside the repo. Tools stay on, but there is nothing to read. The two prompts that triggered the first problem came back clean, and the full 100-row schema rerun showed no contaminated responses.

**The app never used the adapter.** The LoRA path in `app.py`'s `tag_dishes()` called `DishTags.model_validate_json` on `mlx_lm.generate`'s raw stdout, which wraps the answer in a `==========` banner, so parsing could never succeed. `DishTags` also requires a `dish_id` that the training target never contains, because IDs are assigned per menu at extraction time. Every call fell into the `except` branch and returned the fail-safe default. The fix extracts the JSON body and re-attaches the real `dish_id`; the 10 tests pass.

## Limitations

Every label was generated by Claude, and Claude is also a benchmark column. The spec (§5) names this circularity and requires a human spot-check of about 60 dishes stratified across cuisines, because agreeing with yourself is not the same as being right. That check needs me to do it and has not been done. Until it is, the schema-primed Claude row measures how well Claude agrees with its own labels, and every figure here is about synthetic data.

The tagger misses allergens: 69.3% recall overall and a 39.1% false-safe rate on molluscs. The three fail-safe layers exist because of that. This is not a medical device; confirm allergens with restaurant staff.

---

Built by [Jeremy Lee](https://github.com/lyhjeremy). Code, benchmark and dataset card: [github.com/lyhjeremy/menu-decoder](https://github.com/lyhjeremy/menu-decoder).
