"""Layered allergen guard eval: LoRA, then LoRA + lexicon, then LoRA + lexicon +
second opinion, on the held-out test set and the frozen adversarial set.

    python eval/guardrails/run_guard.py second test        --model haiku --run 1
    python eval/guardrails/run_guard.py second adversarial --model haiku --run 1   # and --run 2, 3
    python eval/guardrails/run_guard.py second adversarial --model sonnet --run 1
    python eval/guardrails/run_guard.py score      # -> eval/guardrails/results.json, results.md
    python eval/guardrails/run_guard.py sheet      # -> eval/guardrails/labeling_sheet.csv

Inputs: data/lora/test.jsonl with eval/raw_outputs.json (the LoRA's saved test
outputs from training/bench_tagger.py, same seed-42 order), and
eval/guardrails/adversarial_set.json with outputs/lora_adversarial.json
(from run_lora.py). Second-opinion replies are saved per batch to
outputs/second_<set>_<model>_r<run>.json, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "scripts"))

from allergen_guard import ALLERGENS, guard_levels, open_allergens, second_opinion  # noqa: E402
from allergen_lexicon import lexicon_hits, raise_levels  # noqa: E402
from bench_tagger import allergen_levels, parse_tagging  # noqa: E402

OUT = HERE / "outputs"
BATCH = 10
CLAUDE_SUBSAMPLE = 100  # bench_tagger.py's Claude rows: the first 100 of the shuffled test set


def load_set(name: str) -> list[dict]:
    """[{id, text, truth: {allergen: level}, lora: {allergen: level}, lora_latency_s, hidden?}]"""
    if name == "test":
        rows = []
        for line in (REPO / "data/lora/test.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rows.append({"prompt": rec["messages"][0]["content"],
                             "true": json.loads(rec["messages"][1]["content"])})
        random.Random(42).shuffle(rows)  # same order as training/bench_tagger.py
        raw = json.loads((REPO / "eval/raw_outputs.json").read_text())
        out = []
        for i, (r, lo, lat, cl) in enumerate(zip(rows, raw["lora_out"], raw["lora_lat"],
                                                 raw["claude_schema_out"] + [None] * len(rows))):
            truth = {c["allergen"]: c["level"] for c in r["true"]["allergen_calls"]}
            out.append({"id": f"t{i:03d}", "text": r["prompt"], "truth": {a: truth.get(a, "not_indicated") for a in ALLERGENS},
                        "lora": allergen_levels(parse_tagging(lo)), "lora_raw": lo, "lora_latency_s": lat,
                        "claude": allergen_levels(parse_tagging(cl)) if cl is not None else None})
        return out
    if name == "adversarial":
        spec = json.loads((HERE / "adversarial_set.json").read_text())
        gens = {g["id"]: g for g in json.loads((OUT / "lora_adversarial.json").read_text())["rows"]}
        out = []
        for d in spec["dishes"]:
            g = gens[d["id"]]
            out.append({"id": d["id"], "text": d["prompt"], "truth": d["labels"], "hidden": d["hidden"],
                        "trap": d["trap"], "basis": d["basis"], "lora": allergen_levels(parse_tagging(g["raw"])),
                        "lora_raw": g["raw"], "lora_latency_s": g["latency_s"]})
        return out
    raise SystemExit(f"unknown set {name}")


def dish_text(text: str) -> str:
    return text[len("Dish: "):] if text.startswith("Dish: ") else text


def run_second(name: str, model: str, run: int) -> None:
    rows = load_set(name)
    dest = OUT / f"second_{name}_{model}_r{run}.json"
    done = json.loads(dest.read_text()) if dest.exists() else {"set": name, "model": model, "run": run,
                                                                "batch_size": BATCH, "batches": []}
    have = {d for b in done["batches"] for d in b["ids"]}
    todo = []
    for r in rows:
        lv = guard_levels(r["text"], r["lora"])
        opn = open_allergens(lv)
        if opn and r["id"] not in have:
            todo.append({"id": r["id"], "text": dish_text(r["text"]), "open": opn})
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        t0 = time.time()
        res = second_opinion(chunk, model=model)
        done["batches"].append({"ids": [c["id"] for c in chunk], "open": {c["id"]: c["open"] for c in chunk},
                                "calls": res["calls"], "why": res["why"], "fallback": res["fallback"],
                                "attempts": res["attempts"], "wall_s": round(time.time() - t0, 2)})
        dest.write_text(json.dumps(done, ensure_ascii=False, indent=1))
        cost = sum(a.get("cost_usd") or 0 for a in res["attempts"])
        print(f"{name} {model} r{run}: batch {b // BATCH + 1}/{-(-len(todo) // BATCH)} "
              f"${cost:.4f} {res['attempts'][-1].get('duration_ms')} ms fallback={res['fallback']}", flush=True)


def load_second(name: str, model: str, run: int) -> dict | None:
    p = OUT / f"second_{name}_{model}_r{run}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    calls, why = {}, {}
    for b in data["batches"]:
        calls.update(b["calls"])
        why.update(b["why"])
    attempts = [a for b in data["batches"] for a in b["attempts"]]
    return {"calls": calls, "why": why, "batches": data["batches"], "attempts": attempts}


def pct(x):
    return None if x is None else round(100 * x, 1)


def metrics(rows: list[dict], key: str, targets_only: bool = False) -> dict:
    per = {a: {"n_contains": 0, "hit": 0, "false_safe": 0, "n_neg": 0, "warn_neg": 0} for a in ALLERGENS}
    for r in rows:
        pred = r[key]
        pool = r["hidden"] if targets_only else ALLERGENS
        for a in pool:
            t, p = r["truth"][a], pred[a]
            if t == "contains":
                per[a]["n_contains"] += 1
                per[a]["hit"] += p == "contains"
                per[a]["false_safe"] += p == "not_indicated"
            elif t == "not_indicated":
                per[a]["n_neg"] += 1
                per[a]["warn_neg"] += p != "not_indicated"
    have = [a for a in ALLERGENS if per[a]["n_contains"]]
    tot = sum(per[a]["n_contains"] for a in ALLERGENS)
    neg = sum(per[a]["n_neg"] for a in ALLERGENS)
    out = {
        "n_dishes": len(rows), "n_contains_pairs": tot,
        "recall_macro": pct(statistics.mean(per[a]["hit"] / per[a]["n_contains"] for a in have)) if have else None,
        "false_safe_macro": pct(statistics.mean(per[a]["false_safe"] / per[a]["n_contains"] for a in have)) if have else None,
        "recall_micro": pct(sum(per[a]["hit"] for a in ALLERGENS) / tot) if tot else None,
        "false_safe_micro": pct(sum(per[a]["false_safe"] for a in ALLERGENS) / tot) if tot else None,
        "false_safe_count": sum(per[a]["false_safe"] for a in ALLERGENS),
        "over_warning_micro": pct(sum(per[a]["warn_neg"] for a in ALLERGENS) / neg) if neg else None,
        "n_not_indicated_pairs": neg,
        "per_allergen": {a: {"n_contains": per[a]["n_contains"],
                             "recall": pct(per[a]["hit"] / per[a]["n_contains"]) if per[a]["n_contains"] else None,
                             "false_safe": pct(per[a]["false_safe"] / per[a]["n_contains"]) if per[a]["n_contains"] else None,
                             "false_safe_count": per[a]["false_safe"]}
                         for a in ALLERGENS},
    }
    return out


def apply_layers(rows: list[dict], second: dict | None) -> None:
    for r in rows:
        r["L1"] = r["lora"]
        r["lex_terms"] = [h["term"] for h in lexicon_hits(r["text"])]
        r["L2"] = guard_levels(r["text"], r["lora"])
        extra = (second or {}).get("calls", {}).get(r["id"], {}) if second else {}
        r["second"] = extra
        r["L3"] = raise_levels(r["L2"], extra)


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else None


def cost_latency(second: dict | None, n_dishes: int) -> dict | None:
    if not second:
        return None
    atts = second["attempts"]
    durs = [a["duration_ms"] / 1000 for a in atts if a.get("duration_ms")]
    batch_walls = [b["wall_s"] for b in second["batches"]]
    cost = sum(a.get("cost_usd") or 0 for a in atts)
    sent = sum(len(b["ids"]) for b in second["batches"])
    return {"calls": len(atts), "batches": len(second["batches"]), "dishes_sent": sent,
            "retries": sum(1 for a in atts if a["attempt"] > 1),
            "fallback_batches": sum(1 for b in second["batches"] if b["fallback"]),
            "total_cost_usd": round(cost, 4), "cost_per_dish_usd": round(cost / sent, 5) if sent else None,
            "cost_per_1000_dishes_usd": round(1000 * cost / sent, 2) if sent else None,
            "batch_latency_median_s": round(statistics.median(batch_walls), 2) if batch_walls else None,
            "batch_latency_p95_s": round(p95(batch_walls), 2) if batch_walls else None,
            "cli_duration_median_s": round(statistics.median(durs), 2) if durs else None,
            "latency_per_dish_s": round(sum(batch_walls) / sent, 2) if sent else None}


def lexicon_timing(rows: list[dict]) -> dict:
    t0 = time.perf_counter()
    for _ in range(20):
        for r in rows:
            guard_levels(r["text"], r["lora"])
    per = (time.perf_counter() - t0) / (20 * len(rows))
    return {"per_dish_ms": round(per * 1000, 3)}


def misses(rows: list[dict], key: str, set_name: str) -> list[dict]:
    out = []
    for r in rows:
        for a in ALLERGENS:
            if r["truth"][a] == "contains" and r[key][a] == "not_indicated":
                out.append({"set": set_name, "id": r["id"], "dish": dish_text(r["text"]), "allergen": a,
                            "hidden_target": a in r.get("hidden", []), "lora": r["L1"][a],
                            "lexicon_terms": r["lex_terms"], "basis": r.get("basis")})
    return out


def flips(runs: list[dict]) -> dict:
    """Run-to-run agreement of the second opinion on the same open (dish, allergen) pairs."""
    keys = set()
    for s in runs:
        for b in s["batches"]:
            for d, opn in b["open"].items():
                keys.update((d, a) for a in opn)
    agree = 0
    for d, a in keys:
        vals = {s["calls"].get(d, {}).get(a, "not_indicated") for s in runs}
        agree += len(vals) == 1
    return {"open_pairs": len(keys), "runs": len(runs), "all_runs_agree": agree,
            "agreement_pct": pct(agree / len(keys)) if keys else None}


def score() -> None:
    res = {"generated": time.strftime("%Y-%m-%d"), "machine": "Apple M4",
           "cost_note": "API-equivalent cost reported by the Claude CLI (total_cost_usd); the CLI runs on a subscription.",
           "sets": {}}
    all_misses = []
    for name in ("test", "adversarial"):
        rows = load_set(name)
        main = load_second(name, "haiku", 1)
        apply_layers(rows, main)
        block = {"n_dishes": len(rows), "layers": {}}
        for key, label in (("L1", "LoRA"), ("L2", "LoRA + lexicon"), ("L3", "LoRA + lexicon + Haiku second opinion")):
            block["layers"][key] = {"label": label, "all": metrics(rows, key)}
            if name == "adversarial":
                block["layers"][key]["hidden"] = metrics(rows, key, targets_only=True)
        lat = [r["lora_latency_s"] for r in rows]
        block["lora_latency"] = {"median_s": round(statistics.median(lat), 2), "p95_s": round(p95(lat), 2),
                                 "how": ("saved by training/bench_tagger.py, one mlx_lm.generate subprocess per dish "
                                         "(reloads the model each call)") if name == "test" else
                                        "run_lora.py, model loaded once, generation only"}
        block["lexicon_latency"] = lexicon_timing(rows)
        block["second_opinion"] = cost_latency(main, len(rows))
        block["dishes_sent_to_second_opinion"] = sum(1 for r in rows if open_allergens(r["L2"]))
        if name == "test":
            sub = rows[:CLAUDE_SUBSAMPLE]
            for r in sub:
                r["CL"] = r["claude"]
            block["claude_subsample"] = {"n": len(sub), "note": "first 100 shuffled test rows, the rows bench_tagger.py sent to schema-primed Claude",
                                         "claude_schema_primed": metrics(sub, "CL"),
                                         **{k: metrics(sub, k) for k in ("L1", "L2", "L3")}}
        if name == "adversarial":
            extra = {}
            for model, run in (("haiku", 2), ("haiku", 3), ("sonnet", 1)):
                s = load_second(name, model, run)
                if not s:
                    continue
                rr = load_set(name)
                apply_layers(rr, s)
                extra[f"{model}_r{run}"] = {"all": metrics(rr, "L3"), "hidden": metrics(rr, "L3", targets_only=True),
                                            "cost": cost_latency(s, len(rr))}
            block["other_runs"] = extra
            runs = [s for s in (load_second(name, "haiku", k) for k in (1, 2, 3)) if s]
            if len(runs) > 1:
                block["haiku_run_agreement"] = flips(runs)
                fs = []
                for s in runs:
                    rr = load_set(name)
                    apply_layers(rr, s)
                    fs.append(metrics(rr, "L3", targets_only=True)["false_safe_micro"])
                block["haiku_hidden_false_safe_by_run"] = fs
        res["sets"][name] = block
        all_misses += [m | {"dish": m["dish"].replace("\u2014", "--")} for m in misses(rows, "L3", name)]
        # The generator's dish text has em dashes; the saved copy writes them as "--" (house style).
        (OUT / f"scored_{name}.json").write_text(json.dumps(
            [{k: (r[k].replace("\u2014", "--") if k == "text" else r[k])
              for k in ("id", "text", "truth", "L1", "L2", "L3", "lex_terms", "second") if k in r} |
             ({"hidden": r["hidden"], "trap": r["trap"], "basis": r["basis"]} if name == "adversarial" else {})
             for r in rows], ensure_ascii=False, indent=1))
    res["remaining_false_safe"] = all_misses
    (HERE / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    write_md(res)
    print(json.dumps({n: {k: {"fs": v["all"]["false_safe_micro"], "rec": v["all"]["recall_macro"],
                              "ow": v["all"]["over_warning_micro"]} for k, v in b["layers"].items()}
                      for n, b in res["sets"].items()}, indent=1))


def write_md(res: dict) -> None:
    lines = ["# Layered allergen guard: results", "",
             f"Generated {res['generated']} by `eval/guardrails/run_guard.py score`. {res['cost_note']}", ""]
    for name, b in res["sets"].items():
        lines += [f"## {name} set ({b['n_dishes']} dishes)", "",
                  "| Layer | Recall (macro) | False-safe (macro) | False-safe (micro) | Over-warning |",
                  "|---|---|---|---|---|"]
        for k, v in b["layers"].items():
            m = v["all"]
            lines.append(f"| {v['label']} | {m['recall_macro']}% | {m['false_safe_macro']}% | "
                         f"{m['false_safe_micro']}% ({m['false_safe_count']} of {m['n_contains_pairs']}) | {m['over_warning_micro']}% |")
        if name == "adversarial":
            lines += ["", "Hidden-allergen targets only:", "", "| Layer | Recall | False-safe |", "|---|---|---|"]
            for k, v in b["layers"].items():
                m = v["hidden"]
                lines.append(f"| {v['label']} | {m['recall_micro']}% | {m['false_safe_micro']}% ({m['false_safe_count']} of {m['n_contains_pairs']}) |")
        lines.append("")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")


def sheet() -> None:
    spec = json.loads((HERE / "adversarial_set.json").read_text())
    with (HERE / "labeling_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dish_id", "prompt", "allergen", "draft_label", "hidden_target", "draft_basis",
                    "reviewer_label (contains/may/not_indicated)", "reviewer_source", "reviewer_notes"])
        for d in spec["dishes"]:
            for a in ALLERGENS:
                w.writerow([d["id"], d["prompt"], a, d["labels"][a], "yes" if a in d["hidden"] else "",
                            d["basis"], "", "", ""])
    print("wrote", HERE / "labeling_sheet.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["second", "score", "sheet"])
    ap.add_argument("set", nargs="?")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--run", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "second":
        run_second(a.set, a.model, a.run)
    elif a.cmd == "score":
        score()
    else:
        sheet()
