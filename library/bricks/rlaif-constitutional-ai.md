# RLAIF / Constitutional AI

**One-liner:** Replace human preference labels with AI feedback — a model self-critiques and revises its outputs against a written "constitution" of principles, producing the SFT and preference data to align a policy with little-to-no human labeling.

## The pipeline (Bai et al. 2022, Anthropic)

**Phase 1 — Supervised (critique-revise).** Sample a (often harmful) response, then loop: pick a constitutional principle, ask the model to *critique* its own output against it, then *revise*. Fine-tune on the final revisions → SL-CAI model.

  y₀ → critique(y₀, principleᵢ) → y₁ → … → y_n  ;  SFT on (x, y_n)

**Phase 2 — RLAIF.** For each prompt, sample two responses from SL-CAI. An AI judge picks the better one *per a sampled principle*, yielding preference pairs (y_w ≻ y_l) with **no human labels**. Train a reward model under Bradley–Terry, then RL (PPO) the policy against it — identical machinery to RLHF, only the labeler changed.

The constitution = a short list of natural-language principles (e.g. "choose the response that is most harmless and least likely to be discriminatory"). It is the *only* human input.

## Where it appears

- **Constitutional AI / Claude** (Anthropic) — the original; RLAIF for the harmlessness objective, human feedback retained for helpfulness.
- **RLAIF vs RLHF** (Lee et al. 2023, Google) — off-the-shelf LLM labeler matches human-preference RLHF on summarization/dialogue; "d-RLAIF" skips the RM and scores directly.
- **Self-Rewarding LMs** (Yuan et al. 2024) — the policy *is* its own LLM-as-judge, iterating DPO on self-generated preferences.
- Synthetic preference pipelines broadly (UltraFeedback, Llama-3 reward data) — AI annotators now supply most modern preference datasets.

## Common mistake

Thinking RLAIF removes the need for a reward model or RL — it does not; it only replaces the *human annotator*. The RM-training and PPO/DPO stages are unchanged. The real risk is **reward hacking inherited from the judge**: the policy optimizes the AI labeler's biases (length, sycophancy, self-preference), so judge quality and constitution coverage cap final alignment.

## See also
- [[rlhf]] — same RM+PPO machinery; RLAIF swaps the human labeler for an AI one
- [[llm-as-a-judge]] — the preference-labeling primitive RLAIF runs on, with its biases
- [[reward-hacking-over-optimization]] — failure mode amplified when the labeler is itself a model
