"""Per-allergen recall-on-'contains' / false-safe-rate bar chart, built
from eval/benchmark.json -- no new LLM/local-generation calls needed.

A single confusion matrix doesn't fit this multi-label task the way it fit
Cellar Scanner's single-variety classifier, so this is the honest
alternative the task calls for: two stacked bar charts (recall, false-safe)
across all 14 allergens, one bar group per system.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gen_dishes import ALLERGENS  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval"
COLORS = {"base": "#B0B7C6", "lora": "#C9622D", "claude_teacher_schema_primed": "#2D5C88"}
LABELS = {"base": "Base Qwen2.5-1.5B", "lora": "+ LoRA (this project)",
          "claude_teacher_schema_primed": "Claude teacher (schema-primed)"}


def main():
    results = json.loads((EVAL_DIR / "benchmark.json").read_text())
    by_system = {r["system"]: r for r in results}
    # claude_teacher (bare prompt, no schema given) is deliberately excluded
    # here -- its allergen_calls field is essentially never populated with
    # the right keys when given no schema (see bench_tagger.py's module
    # docstring and eval/benchmark.md's "Why Claude appears twice"), so a
    # per-allergen bar for it would be a flat zero bar, not a real signal.
    # claude_teacher_schema_primed is the fair, on-topic comparison.
    systems = ["base", "lora", "claude_teacher_schema_primed"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = np.arange(len(ALLERGENS))
    width = 0.26

    for ax, metric, title, ylabel in [
        (axes[0], "recall_on_contains", "Allergen recall on \"contains\" (higher is better)", "Recall"),
        (axes[1], "false_safe_rate", "False-safe rate: predicted \"not_indicated\" when true is \"contains\" (lower is better -- this is the dangerous miss)", "False-safe rate"),
    ]:
        for i, system in enumerate(systems):
            vals = []
            for allergen in ALLERGENS:
                v = by_system[system]["per_allergen"][allergen][metric]
                vals.append(v if v is not None else 0.0)
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=LABELS[system], color=COLORS[system])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ALLERGENS, rotation=45, ha="right")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Menu Decoder -- per-allergen safety metrics, base vs LoRA vs Claude teacher (schema-primed)", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "allergen_recall_chart.png", dpi=150)
    print(f"Saved {EVAL_DIR / 'allergen_recall_chart.png'}")


if __name__ == "__main__":
    main()
