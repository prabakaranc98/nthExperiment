# Cross-Entropy

**One-liner:** Expected negative log-likelihood of data under a model's predicted distribution; the loss minimized by classification and next-token prediction; equals entropy plus KL, so minimizing it minimizes KL to the truth.

## The formula

H(P, Q) = −Eₓ~P [log Q(x)] = −Σₓ P(x) log Q(x)

Decomposition (the identity worth memorizing):

H(P, Q) = H(P) + KL(P ‖ Q)

- P = true/target distribution, Q = model's predicted distribution
- H(P) is fixed (doesn't depend on the model), so minimizing H(P,Q) over Q is exactly minimizing KL(P‖Q)
- H(P,Q) ≥ H(P) always, with equality iff Q = P

## In practice (the empirical loss)

For a dataset, P is the empirical distribution (one-hot targets). The per-example loss collapses to plain negative log-likelihood of the correct class/token:

L = −(1/N) Σᵢ log Q(yᵢ | xᵢ)

where Q(· | x) = softmax(logits). With one-hot labels, only the true-class log-prob survives the sum. This is identical to MLE: minimizing CE ⇔ maximizing likelihood.

## Where it appears

- **LM pretraining** — next-token CE over the vocab is *the* objective; reported as loss, and exp(loss) = perplexity
- **Classification** — softmax + CE is the default head; logits → log-softmax → NLL
- **Label smoothing** — replace one-hot P with (1−ε) one-hot + ε uniform, so target H(P)>0; regularizes overconfidence (Transformer, ViT)
- **Knowledge distillation** — soft-target CE between student Q and teacher's temperature-scaled distribution P
- **RLHF / DPO** — the SFT stage is CE on demonstrations; DPO's loss is a CE-style logistic loss on preference pairs

## Common mistake

Confusing CE with KL. They differ by H(P): when targets are fixed (one-hot or a fixed teacher), H(P) is constant so the *gradients* of CE and KL are identical — but the loss *values* differ, and the H(P) offset is why CE cannot reach zero with label smoothing or soft targets. Also: PyTorch `CrossEntropyLoss` takes raw logits (applies log-softmax internally) — feeding it softmax probabilities double-applies softmax and silently degrades training.

## See also
- [[kl-divergence]] — H(P,Q) = H(P) + KL(P‖Q); minimizing CE minimizes KL to truth
- [[softmax]] — produces Q from logits; CE's gradient w.r.t. logits is the clean (softmax − onehot)
- [[scaling-laws]] — the loss whose power-law decay scaling laws describe is this CE
