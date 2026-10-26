# Menu Decoder

Photograph a foreign-language menu and get each dish translated and tagged for cuisine, 14 EU allergens, spice and diet by a Qwen2.5-1.5B LoRA fine-tune, with an allergen display that can warn but never clear.

[Project page](https://lyhjeremy.github.io/menu-decoder/) · [Write-up](https://lyhjeremy.github.io/menu-decoder/overview/) · [Engineering notes](writeup.md) · [Full benchmark](eval/benchmark.md)

Design and copy follow [these standards](https://github.com/lyhjeremy/lyhjeremy/blob/main/DESIGN_STANDARDS.md).

One of four apps that fine-tune a small local model and benchmark it against Claude, with [Cellar Scanner](https://github.com/lyhjeremy/cellar-scanner), [Race Day Copilot](https://github.com/lyhjeremy/race-day-copilot) and [Receipt Auditor](https://github.com/lyhjeremy/receipt-auditor).

## Results

The tagger takes a one-line prompt (`Dish: Pad Thai.`) and emits JSON with cuisine, a `contains` / `may` / `not_indicated` call for each of 14 allergens, spice, vegetarian and vegan. Held-out test set of 303 synthetic dishes, split by dish family:

| System | N | Valid JSON | Cuisine | Allergen recall | False-safe | Latency |
|---|---|---|---|---|---|---|
| Base Qwen2.5-1.5B, bare prompt | 303 | 0.0% | 0.0% | 0.0% | 100.0% | 5.4 s |
| Qwen2.5-1.5B + LoRA, bare prompt | 303 | **100.0%** | **99.3%** | 69.3% | 7.3% | 6.7 s |
| Claude, bare prompt | 100 | 94.0% | 43.0% | 0.0% | 100.0% | 16.5 s |
| Claude, schema in prompt | 100 | 99.0% | 81.0% | **79.2%** | **1.7%** | 13.4 s |

Allergen recall is macro recall on dishes labeled `contains`. False-safe is the share of `contains` dishes tagged `not_indicated`, the dangerous miss.

Given the bare prompt, Claude describes the dish in its own JSON keys and scores zero by construction. Given the tagging rubric and output format, it beats the fine-tune on allergens. The fine-tune's advantage is that the schema lives in its weights: the prompt is one line, it runs locally, and it costs nothing per call. Its weakest allergens are molluscs (39.1% false-safe) and peanuts (18.2%).

![Per-allergen recall and false-safe rate for base, LoRA and schema-primed Claude across 14 allergens; LoRA's false-safe rate is highest on molluscs](eval/allergen_recall_chart.png)

## Method

- **Data.** `scripts/gen_dishes.py` generated 6,750 rows with `claude -p`: 15 cuisines, 150 dishes each, three prompt variants per dish (name only, name with description, original script with translation).
- **Split.** By dish family, so a dish's three variants share a side. Row IDs are hashes, so `training/prep_tagger.py` rebuilds the generator's item list to recover families: 2,250 families, 6,096 train / 351 validation / 303 test rows, leakage check passed.
- **Fine-tune.** LoRA on `mlx-community/Qwen2.5-1.5B-Instruct-4bit` with MLX on Apple Silicon: batch 4, 12 layers, 800 iterations, learning rate 1.5e-4, sequence length 768 (median target about 204 tokens). This is a generative structured-output fine-tune, not a single-label classifier.
- **Benchmark.** `training/bench_tagger.py` scores parse rate, cuisine, spice within 1, vegetarian and vegan agreement, and per-allergen recall and false-safe rate. Claude runs from a scratch directory outside the repo so it can't read the training code.
- **Fail-safe.** Three layers: the tagging rubric asks for `may` whenever an allergen is plausible; `src/safelist.py` renders `contains` as a cross, `may` as "ask the staff" and `not_indicated` as "not indicated", with no green check; and `filter_safe_dishes` removes every `contains` or `may` dish for the user's allergens, and any dish failing a stated diet, before the LLM recommends anything.
- **Other guardrails.** A domain gate refuses non-menu photos, unreadable lines are flagged rather than dropped, recommendations are checked against real dish IDs, and translations and tags are cached on `(original_text, language)`.

## Stack

Python, Gradio, MLX (`mlx_lm`), Qwen2.5-1.5B-Instruct, Pydantic, tesseract, Claude via `claude -p`, Gemini as the batched tagging fallback when no adapter is present, text-to-speech for dish names.

## How to run

```bash
pip install -r requirements.txt
./setup_ocr.sh                          # tesseract and language packs, one time
python scripts/gen_dishes.py            # synthetic corpus (long-running, resumable)
python training/prep_tagger.py          # dish-family split train/valid/test
bash training/lora_harness/train.sh mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    data/lora training/adapters 4 12 800 1.5e-4 768
python training/bench_tagger.py         # benchmark, writes to eval/
python app.py
```

Menu extraction uses `claude -p` on a Claude subscription. Tests: `pytest tests/` (10 tests).

## Repo layout

| Path | Contents |
|---|---|
| `app.py` | Gradio app: photo, badged menu, safe picks, dish-name speech |
| `scripts/gen_dishes.py` | Synthetic dish-tagging corpus generator |
| `training/prep_tagger.py` | Dish-family split with leakage check |
| `training/bench_tagger.py` | Four-row benchmark, per allergen |
| `training/make_tagger_chart.py` | Per-allergen chart |
| `src/decoder.py` | Tagging and recommendation prompts |
| `src/safelist.py` | Badge rendering and the deterministic filter |
| `src/` | Shared toolkit: llm, vision, guardrails, context, cache, audio |
| `eval/` | Benchmark, raw outputs, chart, loss curve, dataset card |

## Limitations

- Every label was generated by Claude, which is also a benchmark column. The spec's human spot-check of about 60 dishes across cuisines has not been done, so the figures measure agreement with synthetic labels.
- The fine-tune's allergen recall (69.3%) is below schema-primed Claude's (79.2%), and it misses rare allergens: 0% recall on sulphites and celery, where it answers `may` instead of `contains`.
- Claude's figures come from a 100-row subsample.
- Not a medical device. Confirm allergens with restaurant staff.

## License

[MIT](LICENSE) © 2026 Jeremy Lee
