"""Banner + architecture diagram for the showcase page. matplotlib (not SVG)
-- matches _ai-gap-toolkit's documented reasoning (qlmanage crops SVG, no
libcairo for cairosvg). Terracotta/paprika palette, distinct from sibling
projects (Cellar Scanner's wine reds, Receipt Auditor's money greens)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "assets"
TERRACOTTA = "#B5451B"
OLIVE = "#6B7A3A"
SAFFRON = "#D9A441"
CREAM = "#FBF3E7"


def make_banner():
    fig, ax = plt.subplots(figsize=(12, 3), dpi=150)
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    fig.patch.set_facecolor("#2b1608")
    ax.set_facecolor("#2b1608")
    ax.text(6, 1.9, "Menu Decoder", ha="center", va="center", fontsize=42,
             color=CREAM, family="serif", weight="bold")
    ax.text(6, 1.1, "Photograph any menu. Get every dish translated, allergen-flagged fail-safe, and pronounced.",
             ha="center", va="center", fontsize=14, color="#EAD9C4", family="serif", style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "banner.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def _box(ax, x, y, w, h, text, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                          facecolor=color, edgecolor="none")
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5,
             color="white", family="sans-serif", weight="bold", wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                  color="#8a7a68", linewidth=1.5))


def make_architecture():
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=150)
    ax.set_xlim(0, 10.8); ax.set_ylim(0, 5.8); ax.axis("off")
    fig.patch.set_facecolor(CREAM)

    _box(ax, 0.3, 4.6, 2.0, 0.8, "Menu photo", TERRACOTTA)
    _box(ax, 2.9, 4.6, 2.3, 0.8, "vision.extract\n(OCR + claude -p)", OLIVE)
    _box(ax, 5.8, 4.6, 1.9, 0.8, "Menu\n(dishes, translations)", TERRACOTTA)
    _box(ax, 8.3, 4.6, 2.1, 0.8, "LoRA (Qwen2.5-1.5B)\ntag cuisine+allergens", SAFFRON)

    _arrow(ax, 2.3, 5.0, 2.9, 5.0)
    _arrow(ax, 5.2, 5.0, 5.8, 5.0)
    _arrow(ax, 7.7, 5.0, 8.3, 5.0)

    _box(ax, 8.3, 3.0, 2.1, 0.8, "Badges\n✗ / ⚠ / ○ (never ✓)", TERRACOTTA)
    _arrow(ax, 9.35, 4.6, 9.35, 3.8)

    _box(ax, 5.8, 3.0, 1.9, 0.8, "Safe-list filter\n(deterministic code)", OLIVE)
    _arrow(ax, 8.3, 3.4, 7.7, 3.4)

    _box(ax, 3.0, 3.0, 2.3, 0.8, "Spoken diet brief\n(mic or text)", TERRACOTTA)
    _arrow(ax, 5.3, 3.4, 5.8, 3.4)

    _box(ax, 0.3, 3.0, 2.2, 0.8, "\"vegetarian,\npeanut allergy\"", SAFFRON)
    _arrow(ax, 2.5, 3.4, 3.0, 3.4)

    _box(ax, 5.8, 1.3, 2.6, 0.8, "LLM picks from\nsafe list only", OLIVE)
    _arrow(ax, 6.85, 3.0, 6.85, 2.1)

    _box(ax, 8.9, 1.3, 1.6, 0.8, "Dish TTS\n(tap to hear)", TERRACOTTA)
    _box(ax, 0.3, 1.3, 4.9, 0.8, "Dev panel: translation completeness % · cache hit rate · backend", SAFFRON)

    ax.text(0.3, 0.4, "Guardrails: domain gate · asymmetric allergen fail-safe (3 layers) · translation completeness · grounded recs",
            fontsize=9, color="#6b5a45", family="sans-serif", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_banner()
    make_architecture()
    print("Wrote banner.png + architecture.png to", OUT)
