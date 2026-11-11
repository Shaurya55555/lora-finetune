# Design decisions and likely questions: lora-finetune

**What is LoRA?**
Instead of updating a weight matrix W, learn a low-rank update: W + (alpha/r) * B * A, with A (r x d) and B (d x r)
small. Only A and B train, so a 1.5B model needs a few million trainable parameters instead of 1.5 billion.
r=16, alpha=32, applied to all attention and MLP projections.

**Why a hand-written training loop?**
To show I understand it: tokenize prompt plus label, mask prompt tokens with -100 so loss is only on the label
tokens, bf16 autocast, gradient clipping, AdamW with warmup then linear decay. The adapter weights are kept in
fp32 while the frozen base is bf16.

**Why Banking77 and why a classification task?**
It has 77 fine-grained intents, which is hard to do well with prompting alone, and it has an unambiguous metric.
Scoring is strict: the output must be exactly one valid label, with no fuzzy credit.

**What are the baselines and are they fair?**
The base model prompted zero-shot with the full label list, and 8-shot with examples. The fine-tuned model sees
only the message, a shorter prompt. Comparing against both prompted baselines shows what fine-tuning buys beyond
prompt engineering.

**Why is the eval a stratified sample?**
10 test examples per class (770) from the 3,080 test set keeps evaluation fast on one GPU while covering all
classes equally. Macro-F1 is reported next to accuracy.

**What are the risks?**
One seed and one training size. Results should be read as a single run, not a distribution; the README says so.

**Local vs Colab:** the same scripts run in `colab.ipynb` on a free T4 if the local machine lacks memory.
