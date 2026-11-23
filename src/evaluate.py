"""Evaluate base-prompted vs LoRA-finetuned on a stratified Banking77 test sample.

python src/evaluate.py --mode zeroshot
python src/evaluate.py --mode fewshot
python src/evaluate.py --mode lora --adapter adapter
Strict scoring: the output must be exactly one valid label (no fuzzy credit).
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (MODEL, chat_prompt, load_banking77, macro_f1, stratified_sample, to_label, user_msg_finetuned,
                    user_msg_prompted)

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zeroshot", "fewshot", "lora"], required=True)
    ap.add_argument("--adapter", default=str(ROOT / "adapter"))
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Run this in Colab (T4 runtime) or on a machine with a CUDA GPU.")
    if a.mode == "lora" and not (Path(a.adapter) / "adapter_config.json").exists():
        raise SystemExit(f"No LoRA adapter at {a.adapter}. Run src/train.py first (the adapter is not in this repo).")

    ds, names = load_banking77()
    test = stratified_sample(ds["test"], a.n_per_class, seed=1)
    shots_src = stratified_sample(ds["train"], 1, seed=5).select(range(8))
    shots = [(t, names[y]) for t, y in zip(shots_src["text"], shots_src["label"])]

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    if a.mode == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    if a.mode == "lora":
        prompts = [chat_prompt(tok, user_msg_finetuned(t)) for t in test["text"]]
    else:
        prompts = [chat_prompt(tok, user_msg_prompted(t, names, shots if a.mode == "fewshot" else None))
                   for t in test["text"]]
    gold = [names[y] for y in test["label"]]

    preds, raw, t0 = [], [], time.time()
    for i in range(0, len(prompts), a.bs):
        enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=14, do_sample=False, pad_token_id=tok.pad_token_id)
        for o in out[:, enc["input_ids"].shape[1]:]:
            txt = tok.decode(o, skip_special_tokens=True)
            raw.append(txt)
            preds.append(to_label(txt, names))
    secs = time.time() - t0

    valid = sum(p is not None for p in preds)
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    res = {"mode": a.mode, "n": len(gold), "accuracy": round(acc, 4),
           "macro_f1": round(macro_f1(gold, [p or "<invalid>" for p in preds], names), 4),
           "valid_label_rate": round(valid / len(gold), 4), "sec_per_example": round(secs / len(gold), 3),
           "common_errors": Counter((g, r.strip()[:30]) for g, p, r in zip(gold, preds, raw) if p != g).most_common(5)}
    print(json.dumps(res, indent=2, default=str))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"eval_{a.mode}.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
