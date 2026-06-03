# Catastrophic Forgetting & Continual Learning

**One-liner:** When you train on task B, gradient descent overwrites the weights encoding task A — the central tension in post-training, model editing, and continual pretraining, where "improve on new data" silently means "regress on old".

## The mechanism

A network has no constraint tying current weights to past tasks. Training on B minimizes L_B(θ); nothing penalizes drift from θ_A*. The Bayesian view: the posterior after A is the prior for B, but SGD ignores the curvature of L_A and walks freely off the A-optimal ridge.

**EWC (Kirkpatrick et al., 2017)** — quadratic penalty toward old weights, weighted by Fisher information (importance):

L(θ) = L_B(θ) + Σ_i (λ/2) F_i (θ_i − θ_A,i*)²

F_i ≈ diagonal of the empirical Fisher (curvature of L_A). High-Fisher params are anchored; low-Fisher ones are free to adapt.

## Other levers

- **Replay / rehearsal:** mix a fraction of old data (or generated pseudo-data) into the new batch — the single most reliable fix; "experience replay" beats most regularizers.
- **Parameter isolation:** freeze the base, learn a low-rank delta (LoRA) or separate adapter — forgetting is structurally impossible because the base never moves.
- **Functional / output-space:** distill old-model logits (LwF, "learning without forgetting") instead of constraining weights.

## Where it appears

- **RLHF / post-training** — the KL-to-reference-policy term is exactly a forgetting regularizer; it keeps the aligned model from drifting off the pretrained/SFT manifold and losing capabilities.
- **Continual pretraining (Llama, domain-adaptation)** — re-warming LR then replaying a slice of the original pretraining mix to absorb new data without degrading general benchmarks.
- **Model editing (ROME/MEMIT)** — surgical fact edits aim to change one association without "ripple" damage to unrelated knowledge.
- **Machine unlearning** — the inverse problem: forget *on purpose* without nuking everything else.

## Common mistake

Believing forgetting is a slow, graceful decay. It is *catastrophic* — abrupt and near-total — and it hits capabilities you never measured. Low new-task loss tells you nothing about retention; you must explicitly evaluate the old task, or replay old data, to detect and prevent it.

## See also
- [[kl-regularization-to-reference-policy]] — RLHF's anti-forgetting anchor, the functional analog of EWC
- [[fisher-information-natural-gradient]] — the curvature that weights EWC's per-parameter penalty
- [[machine-unlearning]] — the deliberate-forgetting flip side of the same dynamics
