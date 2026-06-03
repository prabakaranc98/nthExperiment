# Multi-Token Prediction (MTP)

**One-liner:** Predict the next *k* tokens at each position instead of just one — used as an auxiliary training objective for denser supervision (DeepSeek-V3) and/or as built-in draft heads for self-speculative decoding (Medusa, EAGLE).

## The objective

Standard LM loss predicts one token: L = −Σₜ log p(xₜ₊₁ | x≤ₜ).

MTP adds *k* heads (or sequential modules) predicting xₜ₊₁, …, xₜ₊ₖ from position t:

L_MTP = Σₜ Σⱼ₌₁ᵏ λⱼ · ( −log p_j(xₜ₊ⱼ | x≤ₜ) )

- **Gloeckle et al. (2024)** — *k* independent output heads on a shared trunk, predicted in parallel; extra heads discarded at inference (or kept for self-speculation).
- **DeepSeek-V3** — *sequential* MTP modules: head j conditions on the trunk hidden state **and** the embedding of the previously predicted token, preserving the causal chain. Depth D=1 (predict 2 tokens total) in practice; main-model output is what's deployed, MTP is auxiliary.

## Where it appears

- **DeepSeek-V3** — auxiliary MTP loss (λ≈0.3, depth 1) densifies training signal and improves data efficiency; the MTP module is then repurposed as a speculative draft head, ~1.8x decode speedup at ~85-90% acceptance.
- **Gloeckle et al. (2024, "Better & Faster LLMs via MTP")** — n=4 heads help most on code/reasoning and at scale; near-free 3x self-speculative inference.
- **Medusa** — adds k cheap MLP heads on a *frozen* backbone to draft k tokens, verified in one pass via a tree.
- **EAGLE** — autoregressive feature-level draft head (the more accurate successor); MTP-style multi-step drafting.

## Common mistake

Conflating the two uses. The *training* benefit (richer gradient signal, better representations) and the *inference* benefit (self-speculation) are separable: a frozen model can get Medusa heads bolted on for speed only, while DeepSeek-V3 trains MTP mainly for quality and gets drafting as a bonus. Also: extra heads are usually **dropped** at inference for the deployed next-token model — they are not changing the main model's predictions.

## See also
- [[eagle-medusa-self-speculation]] — the canonical self-speculative decoders MTP heads enable
- [[speculative-decoding]] — the draft-then-verify paradigm MTP self-speculation instantiates
- [[teacher-forcing-exposure-bias]] — MTP's multi-step targets partially mitigate single-step teacher forcing
