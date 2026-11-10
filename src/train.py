"""LoRA fine-tune Qwen2.5-1.5B-Instruct on Banking77 with a hand-written training loop.

python src/train.py --n-per-class 40 --epochs 1
Loss is computed on the label tokens only (prompt tokens masked with -100).
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL, chat_prompt, load_banking77, stratified_sample, user_msg_finetuned

ROOT = Path(__file__).resolve().parents[1]


def encode(tok, text, label_name, max_len=160):
    prompt = chat_prompt(tok, user_msg_finetuned(text))
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(label_name + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return ids, labels


def collate(batch, pad_id):
    m = max(len(i) for i, _ in batch)
    ids = torch.tensor([i + [pad_id] * (m - len(i)) for i, _ in batch])
    lab = torch.tensor([l + [-100] * (m - len(l)) for _, l in batch])
    att = torch.tensor([[1] * len(i) + [0] * (m - len(i)) for i, _ in batch])
    return ids, lab, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "adapter"))
    a = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    ds, names = load_banking77()
    train = stratified_sample(ds["train"], a.n_per_class, seed=0)
    print("train examples:", len(train), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 adapters, bf16 base
    model.print_trainable_parameters()

    data = [encode(tok, t, names[y]) for t, y in zip(train["text"], train["label"])]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.epochs * ((len(data) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / steps))

    model.train()
    log, step, t0 = [], 0, time.time()
    for ep in range(a.epochs):
        random.shuffle(data)
        for i in range(0, len(data), a.bs):
            ids, lab, att = [x.to("cuda") for x in collate(data[i:i + a.bs], tok.pad_token_id)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0 or step == steps:
                log.append({"step": step, "loss": round(loss.item(), 4), "sec": round(time.time() - t0)})
                print(log[-1], flush=True)
    model.save_pretrained(a.out)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "train_log.json").write_text(json.dumps(
        {"args": vars(a), "n_train": len(train), "train_seconds": round(time.time() - t0),
         "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "log": log}, indent=2))
    print("saved adapter to", a.out)


if __name__ == "__main__":
    main()
