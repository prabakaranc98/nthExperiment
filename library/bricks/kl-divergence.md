# KL Divergence

**One-liner:** Asymmetric measure of how different distribution Q is from distribution P; appears in RLHF, VAEs, conformal prediction, and generalization bounds.

## The formula

KL(P ‖ Q) = Eₓ~P [log P(x)/Q(x)] = Eₓ~P [log P(x)] − Eₓ~P [log Q(x)]

- KL(P‖Q) ≥ 0 always (Gibbs' inequality)
- KL(P‖Q) = 0 iff P = Q almost everywhere
- **Not symmetric**: KL(P‖Q) ≠ KL(Q‖P)

## Forward vs. reverse KL

- **Forward KL(P‖Q)**: Q must cover P's support → zero-avoiding (Q spreads out)
- **Reverse KL(Q‖P)**: Q focuses on P's modes → zero-forcing (Q collapses to modes)
- VAEs minimize reverse KL(q(z|x) ‖ p(z)); mean-field VI also uses reverse KL

## Where it appears

- **RLHF PPO** — KL(π‖π_ref) penalty prevents reward hacking; β controls the tradeoff
- **DPO** — the optimal RLHF policy π*(y|x) ∝ π_ref(y|x) exp(R/β) is derived from the KL-constrained objective
- **VAE** — ELBO = −KL(q(z|x)‖p(z)) + E[log p(x|z)]
- **PAC-Bayes** — generalization bounds use KL(Q‖P) between posterior and prior
- **Information theory** — KL(P‖Q) = H(P,Q) − H(P); cross-entropy = entropy + KL

## Common mistake

Confusing the direction. KL(model‖data) penalizes the model for missing modes (mode-covering). KL(data‖model) penalizes the model for hallucinating modes (mode-seeking). RLHF uses KL(π‖π_ref) — it penalizes the policy for deviating from the reference, not the reverse.

## See also
- [[cross-entropy]] — H(P,Q) = H(P) + KL(P‖Q)
- [[dpo]] — derived from the KL-constrained RLHF objective
- [[implicit-bias]] — PAC-Bayes bounds use KL
