"""Run the LoRA tagger in-process over a set of dish prompts and save the raw
generations with per-dish latency.

    python eval/guardrails/run_lora.py valid        # data/lora/valid.jsonl (optional, not used in results)
    python eval/guardrails/run_lora.py adversarial  # eval/guardrails/adversarial_set.json

The held-out test set is not rerun: its LoRA outputs are already saved in
eval/raw_outputs.json by training/bench_tagger.py. A check on the first three
test rows (greedy decoding, same adapter) reproduced those saved outputs exactly,
so in-process and subprocess generation give the same text; only the latency
differs, because bench_tagger.py reloaded the model for every call.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MAX_TOKENS = 300  # same budget as training/bench_tagger.py


def prompts_for(which: str) -> list[dict]:
    if which == "valid":
        rows = []
        for i, line in enumerate((REPO / "data/lora/valid.jsonl").read_text().splitlines()):
            if line.strip():
                rec = json.loads(line)
                rows.append({"id": f"v{i}", "prompt": rec["messages"][0]["content"]})
        return rows
    if which == "adversarial":
        data = json.loads((HERE / "adversarial_set.json").read_text())
        return [{"id": d["id"], "prompt": d["prompt"]} for d in data["dishes"]]
    raise SystemExit(f"unknown set {which}")


def main() -> None:
    which = sys.argv[1]
    from mlx_lm import generate, load

    model, tok = load(BASE_MODEL, adapter_path=str(REPO / "training/adapters"))
    rows = prompts_for(which)
    out = []
    for i, r in enumerate(rows):
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                    add_generation_prompt=True, tokenize=False)
        t0 = time.perf_counter()
        text = generate(model, tok, prompt=p, max_tokens=MAX_TOKENS)
        out.append({"id": r["id"], "prompt": r["prompt"], "raw": text,
                    "latency_s": round(time.perf_counter() - t0, 3)})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    dest = HERE / "outputs" / f"lora_{which}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps({"model": BASE_MODEL, "adapter": "training/adapters",
                                "machine": "Apple M4", "rows": out}, ensure_ascii=False, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
