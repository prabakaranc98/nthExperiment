# Reasoning Distillation

**One-liner:** Teach a small model to reason by SFT on long chain-of-thought traces sampled from a strong reasoner (e.g., DeepSeek-R1), trading expensive RLVR for cheap supervised cloning — and the distilled student often beats the same small model trained with RL directly.

## The recipe

1. Take a strong reasoning teacher T (R1, o-series, QwQ) that already emits long CoT.
2. Sample traces for a prompt set: full `<think>...</think>` reasoning + final answer. Optionally **reject-sample** (keep only traces whose answer passes a verifier — rejection-FT / STaR-style).
3. Standard SFT (next-token CE) on the student S over `prompt → (CoT, answer)`:

   L(S) = − Σ_t log p_S(y_t | y_<t, x),  y = teacher trace

   No reward model, no RL loop, no value function.

This is **sequence-level distillation** (Kim & Rush 2016): cloning teacher *samples*, not the teacher's full token distribution (no KL on logits). The "knowledge" being transferred is the reasoning *trajectory*, not soft labels.

## Where it appears

- **DeepSeek-R1 (2025)** — distilled R1 traces into Qwen/Llama 1.5B–70B by pure SFT; the paper's headline finding: **distillation > running GRPO/RLVR directly on the small base** at equal scale. RL on a small model can't easily discover the long-CoT behavior the teacher already found.
- **s1 / s1K (Muennighoff 2025)** — 1K curated long traces + budget forcing; near-o1 math with tiny SFT.
- **Sky-T1, OpenThoughts, Bespoke-Stratos, LIMO** — open trace datasets distilled from R1/QwQ; "less is more" curation effects.
- **Distill-then-RL** — common 2025–26 pipeline: SFT-distill to bootstrap CoT, then a short RLVR phase to push past the teacher.

## Common mistake

Assuming distillation only copies the teacher and caps the student at teacher performance. The point is the opposite: SFT-distillation is a far cheaper, more sample-efficient way to *instill* long-CoT than rediscovering it via RL from scratch — and it's a launchpad, not a ceiling (RLVR on top can exceed the teacher). A second trap: distilling raw traces without verifying answers, importing the teacher's wrong reasoning and hallucinated steps.

## See also
- [[knowledge-distillation]] — the general teacher→student framework this specializes
- [[rlvr]] — the expensive alternative distillation competes with / precedes
- [[self-improvement-star-bootstrapping]] — rejection-sampling on verified traces, the self-taught variant
