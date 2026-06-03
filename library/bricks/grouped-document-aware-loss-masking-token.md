# Grouped / Document-Aware Loss Masking & Token Weighting

**One-liner:** A per-token mask/weight vector that decides which tokens contribute to the next-token loss and how much — completion-only SFT masks the prompt, packing masks cross-document spans, and weighting corrects for length/turn imbalance — a quietly load-bearing post-training detail.

## The formula / definition

Standard LM loss with a per-token weight `w_i ∈ {0,1}` (mask) or `w_i ∈ ℝ≥0` (weight):

```
L = (Σ_i w_i · (-log p_θ(x_i | x_<i))) / (Σ_i w_i)
```

- **Prompt masking (completion-only SFT):** `w_i = 1` only for assistant/response tokens, `w_i = 0` for system+user prompt. Gradient flows only through what the model must *generate*.
- **Packed/grouped batches:** concatenate docs to fill context; mask so token i never attends across a document boundary (block-diagonal attention) AND `w_i = 0` on padding / EOS-bridging tokens.
- **The normalizer matters:** dividing by `Σ w_i` (token-mean) vs by batch size (sum, then /B) vs per-sequence-then-mean changes the effective weight of long vs short examples.

## Where it appears

- **Alpaca / completion-only SFT (TRL `DataCollatorForCompletionOnlyLM`)** — masks the instruction; training on prompts hurts instruction-following.
- **Sequence packing in pretraining/SFT (FlashAttention `cu_seqlens`, document attention masks)** — group multiple docs per sequence without cross-contamination.
- **Multi-turn chat SFT** — mask all turns except the targets; naive per-token mean over-weights long-transcript samples (the "last-turn vs all-turns" choice).
- **DPO/GRPO and RLHF** — length normalization of the per-token logprob sum is the same masking/weighting question; mis-normalization drives length bias and reward hacking.

## Common mistake

Using token-mean normalization (`Σ loss / Σ w_i`) inside gradient accumulation: each micro-batch is averaged independently, so a micro-batch with few unmasked tokens gets the same weight as one with many — silently distorting the loss versus a true global token-mean. Same trap underlies the "DPO/GRPO length normalization" debates: the *choice of denominator* is a modeling decision, not a formatting detail.

## See also
- [[sequence-packing-document-attention-masking]] — the attention-mask half of document-aware training
- [[sft-instruction-tuning]] — where completion-only masking lives
- [[length-normalization-bias-control]] — the same denominator choice in preference/RL objectives
