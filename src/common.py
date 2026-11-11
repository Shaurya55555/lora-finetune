"""Shared data + prompt + scoring for the Banking77 LoRA experiment."""
import difflib
import random

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a banking support classifier. Reply with exactly one intent label and nothing else."


def load_banking77():
    from datasets import load_dataset  # lazy: keeps the pure helpers importable without the heavy deps

    ds = load_dataset("PolyAI/banking77")
    names = ds["train"].features["label"].names
    return ds, names


def stratified_sample(split, n_per_class: int, seed: int):
    rnd = random.Random(seed)
    by = {}
    for i, y in enumerate(split["label"]):
        by.setdefault(y, []).append(i)
    idx = []
    for y, items in by.items():
        rnd.shuffle(items)
        idx += items[:n_per_class]
    rnd.shuffle(idx)
    return split.select(idx)


def user_msg_finetuned(text: str) -> str:
    return f"Message: {text}\nIntent:"


def user_msg_prompted(text: str, labels: list[str], shots: list[tuple[str, str]] | None = None) -> str:
    s = "Allowed intent labels:\n" + ", ".join(labels) + "\n\n"
    for t, l in shots or []:
        s += f"Message: {t}\nIntent: {l}\n\n"
    return s + f"Message: {text}\nIntent:"


def chat_prompt(tok, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def to_label(pred: str, labels: list[str]) -> str | None:
    """Exact label if the model produced one; otherwise None (strict scoring, no fuzzy credit)."""
    p = pred.strip().split("\n")[0].strip().strip("`'\" .").lower()
    # Case-insensitive match that returns the CANONICAL label: one Banking77 label is
    # 'Refund_not_showing_up' (capital R), so comparing a lowercased prediction to the raw list never matched it.
    canon = {l.lower(): l for l in labels}
    return canon.get(p)


def nearest_label(pred: str, labels: list[str]) -> str:
    m = difflib.get_close_matches(pred.strip().lower(), labels, n=1, cutoff=0.0)
    return m[0] if m else labels[0]
