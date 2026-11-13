# Layered allergen guard: results

Generated 2026-11-13 by `eval/guardrails/run_guard.py score`. API-equivalent cost reported by the Claude CLI (total_cost_usd); the CLI runs on a subscription.

## test set (303 dishes)

| Layer | Recall (macro) | False-safe (macro) | False-safe (micro) | Over-warning |
|---|---|---|---|---|
| LoRA | 69.3% | 7.3% | 4.6% (30 of 658) | 6.2% |
| LoRA + lexicon | 79.7% | 4.5% | 2.6% (17 of 658) | 7.2% |
| LoRA + lexicon + Haiku second opinion | 82.0% | 0.8% | 0.3% (2 of 658) | 8.9% |

## adversarial set (60 dishes)

| Layer | Recall (macro) | False-safe (macro) | False-safe (micro) | Over-warning |
|---|---|---|---|---|
| LoRA | 43.8% | 28.7% | 16.7% (22 of 132) | 23.3% |
| LoRA + lexicon | 61.8% | 12.4% | 8.3% (11 of 132) | 23.5% |
| LoRA + lexicon + Haiku second opinion | 74.2% | 0.0% | 0.0% (0 of 132) | 27.0% |

Hidden-allergen targets only:

| Layer | Recall | False-safe |
|---|---|---|
| LoRA | 53.7% | 17.9% (22 of 123) |
| LoRA + lexicon | 65.9% | 8.9% (11 of 123) |
| LoRA + lexicon + Haiku second opinion | 74.8% | 0.0% (0 of 123) |

