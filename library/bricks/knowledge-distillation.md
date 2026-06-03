# Knowledge Distillation

**One-liner:** Train a small student to match a large teacher's soft output distribution (and often intermediate features/attention), transferring "dark knowledge" — the relative probabilities of wrong classes — that hard labels throw away.

## The formula (Hinton et al., 2015)

Soften both teacher and student logits with temperature T, then match:

L = (1−λ)·CE(y, σ(z_s)) + λ·T²·KL( σ(z_t/T) ‖ σ(z_s/T) )

- z_t, z_s = teacher / student logits; σ = softmax
- High T (e.g. 2–10) flattens the distribution, exposing the dark knowledge (e.g. "this 7 looks a bit like a 1")
- The T² factor rescales the soft-loss gradient back to the magnitude of the hard-loss gradient (soft gradients scale as 1/T²)
- λ trades off matching the teacher vs. the ground-truth labels

In the T→∞ limit, the soft loss reduces to matching logits up to a constant (logit-matching / MSE on logits).

## Variants beyond logit matching

- **Feature/hint distillation** (FitNets): regress student intermediate activations onto teacher's (with a projection)
- **Attention transfer**: match attention maps / attention matrices
- **Sequence-level KD** (Kim & Rush): for autoregressive models, distill on teacher-generated sequences, not token-level distributions
- **On-policy / generative KD** (GKD, MiniLLM): student samples its own sequences, teacher scores them — fixes train/inference distribution mismatch

## Where it appears

- DistilBERT, TinyBERT, MobileBERT — early small-LM families; ~40% smaller, ~95% of teacher quality
- Gemma 2 / Gemma 3, Llama 3.x small models — pretraining-scale distillation from a larger sibling teacher's next-token distribution
- DeepSeek-R1-Distill, reasoning distillation — distill long chain-of-thought traces from a strong reasoner into small dense models (SFT on teacher CoT, not soft logits)
- Diffusion distillation — collapse many sampling steps into few (a different "teacher")

## Common mistake

Distilling off-policy only. If the student trains purely on teacher outputs/data it never produced, at inference it sees its own (lower-quality) generations and drifts (exposure bias). On-policy KD — student generates, teacher provides the target distribution — closes this gap; it is the dominant modern recipe for sequence models.

## See also
- [[reasoning-distillation]] — distilling chain-of-thought traces into smaller models
- [[kl-divergence]] — the soft-target objective is forward/reverse KL to the teacher
- [[diffusion-distillation]] — step-count reduction, the same teacher-student idea for samplers
