# lora-finetune: LoRA on Qwen2.5-1.5B vs prompted baselines

Fine-tuning Qwen2.5-1.5B-Instruct with LoRA on Banking77 intent classification (77 labels) and comparing it with
two prompted baselines on the same held-out sample.

1. base model, zero-shot, prompted with the full label list
2. base model, 8-shot, prompted with the label list plus 8 examples
3. **LoRA fine-tuned** (rank 16, alpha 32, all attention and MLP projections), hand-written PyTorch training loop,
   loss on the label tokens only; the fine-tuned model sees just the message, no label list

Strict scoring: the output must be exactly one valid label (matched case-insensitively to the canonical label name);
anything else counts as wrong, with no fuzzy credit.

## Results

Evaluated on 770 test examples (10 per class, stratified, fixed seed) out of the 3,080 in the Banking77 test set.

| model | accuracy | macro-F1 | valid-label rate |
|---|---|---|---|
| prompted zero-shot | 0.4247 | 0.3941 | 0.9571 |
| prompted 8-shot | 0.4299 | 0.4013 | 0.9351 |
| **LoRA fine-tuned** | **0.8532** | **0.8507** | **0.9896** |

LoRA doubles accuracy over both prompted baselines (+42.3 points over 8-shot), and almost always emits a valid label.
Adding 8 examples to the prompt barely helped (+0.5 points), so the gain comes from training, not from prompt engineering.

Training: 3,075 examples (40 per class), 1 epoch, 18.5M trainable parameters (1.18% of 1.56B), AdamW at 2e-4 with
warmup and linear decay, bf16 base weights with fp32 adapters, about 12 minutes on a Colab T4.

## Run
Locally (needs a CUDA GPU) or in Colab (`colab.ipynb`, T4 runtime, Run all):

    pip install torch transformers peft accelerate datasets
    python src/train.py --n-per-class 40 --epochs 1
    python src/evaluate.py --mode zeroshot
    python src/evaluate.py --mode fewshot
    python src/evaluate.py --mode lora
    pytest                  # 9 tests, CPU only, no model download

## Failure handling
| situation | behavior |
|---|---|
| model output is not exactly one valid label (extra words, a made-up label, empty text) | `to_label` returns `None`; scored as wrong and counted against the valid-label rate, with no fuzzy credit |
| prediction differs from the canonical label only by case or surrounding quotes, backticks, spaces | matched to the canonical label (one Banking77 label starts with a capital letter) |
| label with no support in the sample | contributes an F1 of 0 to macro-F1 rather than being skipped |
| no CUDA GPU | `evaluate.py` exits with a message pointing to Colab instead of crashing inside PyTorch |
| `--mode lora` without a trained adapter | `evaluate.py` exits with a message to run `src/train.py` first |
| tokenized example longer than 160 tokens | truncated in training (`encode`); Banking77 messages are short, so this is a guard, not a measured case |

Not handled: no checkpointing or resume in `train.py`, and no out-of-memory fallback (batch size is a flag).

## Limits and honesty notes
- One training run, one seed, one training size. Treat differences of about one point as noise; the LoRA-versus-prompting
  gap (over 40 points) is far larger than that.
- 770 test examples (10 per class) is a sample, not the full 3,080-example test set.
- Only the 1.5B model on one task; no comparison with full fine-tuning, other ranks, or larger prompted models.
- Trained and evaluated on a free Colab T4 because the local machine could not install PyTorch under its memory limits.
- Two bugs were found and fixed along the way, both in tooling rather than the model: Colab's preinstalled `torchao` was
  too old for current `peft` (the notebook now uninstalls it), and my scorer lowercased predictions, so one Banking77 label
  spelled with a capital letter (`Refund_not_showing_up`) could never match. An earlier run before that fix scored
  slightly lower on all three models; the table above is from a full re-run after the fix.
- The raw `results/eval_*.json` files from the Colab run were not saved into this repo, so the table above is
  reproducible by re-running the notebook but cannot be checked against stored output here.
- The model is not published to the Hugging Face Hub; the adapter is not in this repo.
