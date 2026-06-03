# Inference-Optimal Over-Training

**One-liner:** Deliberately train a *smaller* model on far *more* tokens than Chinchilla-optimal (often 100-1000:1 token-to-param ratios) because lifetime inference cost — not one-time training cost — dominates the total compute bill of a deployed model.

## The key insight

Chinchilla minimizes *training* compute for a target loss: N* ∝ C^0.5, D* ∝ C^0.5, giving D/N ≈ 20 tokens/param. But the relevant objective for a shipped model is **total cost = training + inference**:

C_total ≈ 6·N·D_train + 2·N·D_inference

Inference is ~2N FLOPs/token, and D_inference (tokens served over the model's lifetime) can dwarf D_train. Minimizing C_total over (N, D_train) at fixed target loss pushes you to a smaller N and a much larger D_train than Chinchilla — you "over-train" the small model to recover the loss you lost by shrinking N. The optimum depends on the expected inference-to-training token ratio: the more you'll serve, the smaller and more over-trained you go.

Sardana et al. (2023/24, "Beyond Chinchilla-Optimal") formalize this — modified scaling-law fits show optimal token ratios climbing into the hundreds:1 once you assume realistic serving volumes.

## Where it appears

- **Llama 3 (8B/70B)** — trained on ~15T tokens (≈1875 tokens/param for 8B), wildly past Chinchilla, explicitly for deployment efficiency.
- **Llama 2 7B, Mistral 7B, Gemma, Phi** — small models over-trained so they're cheap to serve and run on edge/single-GPU.
- **Distillation pipelines** — over-trained small models pair with [[knowledge-distillation]] (Gemma 2, Llama 3.2) to push the quality/cost frontier further.
- **Inference cost modeling** — choosing 8B-over-trained vs 70B-Chinchilla for an API with high QPS.

## Common mistake

Reporting an over-trained small model as "Chinchilla-suboptimal" as if it were a mistake. It is *training-compute*-suboptimal **on purpose** — it is *total-cost*-optimal given inference volume. The other trap: over-training has diminishing returns (loss is a power law in D), so past some D the extra tokens buy almost nothing — and data-constrained repetition penalties kick in once you exhaust unique high-quality tokens.

## See also
- [[scaling-laws]] — Chinchilla is the training-optimal baseline this deliberately departs from
- [[compute-budget-identity-c-6nd]] — the 6ND / 2N accounting that drives the train-vs-inference tradeoff
- [[data-constrained-scaling-repetition-laws]] — the ceiling on over-training once unique tokens run out
