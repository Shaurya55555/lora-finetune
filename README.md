# lora-finetune: LoRA on Qwen2.5-1.5B vs a prompted baseline

Status: **training and evaluation in progress, results below are filled in as runs finish.**

Task: Banking77 intent classification (77 labels). Compares, on the same stratified test sample:
1. base model, zero-shot prompted with the full label list
2. base model, 8-shot prompted
3. LoRA fine-tuned (r=16, all attention + MLP projections), hand-written PyTorch training loop,
   loss on label tokens only

Strict scoring: the output must be exactly one valid label (no fuzzy credit).

## Results
| model | accuracy | macro-F1 | valid-label rate |
|---|---|---|---|
| prompted zero-shot | TBD | TBD | TBD |
| prompted 8-shot | TBD | TBD | TBD |
| LoRA fine-tuned | TBD | TBD | TBD |

## Run
    pip install torch transformers peft accelerate datasets
    python src/train.py --n-per-class 40 --epochs 1
    python src/evaluate.py --mode zeroshot
    python src/evaluate.py --mode fewshot
    python src/evaluate.py --mode lora

Trained locally on one RTX 4060 (8 GB), bf16 base weights with fp32 adapters.
