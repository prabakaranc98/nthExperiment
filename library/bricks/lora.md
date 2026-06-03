# LoRA — Low-Rank Adaptation

**One-liner:** Fine-tune only a low-rank decomposition ΔW = AB (r << min(d,k)); freeze the pretrained weights; dramatically reduces trainable parameters.

## The math

For a pretrained weight matrix W₀ ∈ ℝ^{d×k}, LoRA constrains the update:

W = W₀ + ΔW = W₀ + BA

where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r << min(d,k).

During training: only A and B are updated. W₀ is frozen.
During inference: W can be merged: W_merged = W₀ + BA (no added latency).

**Initialization:** A ~ N(0, σ²), B = 0 (so ΔW = 0 at init — the model starts from the pretrained checkpoint).

## Why it works

Empirical finding: the weight update matrices ΔW during fine-tuning have low *intrinsic rank* — SVD of ΔW shows that the top-r singular values capture most of the variance, even for small r. The useful fine-tuning signal lives in a low-dimensional subspace.

## The scaling factor

In practice: ΔW = (α/r) · BA where α is a scaling hyperparameter. This separates the learning rate from the rank, making sweeps easier.

## Variants

- **QLoRA** ([arXiv 2305.14314](https://arxiv.org/abs/2305.14314)) — 4-bit quantized base model + LoRA adapters; fine-tune 65B on one GPU
- **DoRA** — decomposes weight into magnitude + direction; LoRA for the direction only
- **LoRA+** — different LRs for A and B; better convergence

## Where it appears

Every practical LLM fine-tuning workflow. PEFT library (HuggingFace). Post-training in most open-source labs.

## Common mistake

Applying LoRA to all weight matrices equally. In practice: applying LoRA to Q and V projections (and sometimes the FFN) gives most of the benefit; applying to K often helps less.

## See also
- [[svd]] — the theoretical motivation (low intrinsic rank of ΔW)
- [[dpo]] — LoRA is the standard way to run DPO fine-tuning
