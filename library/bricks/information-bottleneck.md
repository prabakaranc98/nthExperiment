# Information Bottleneck

**One-liner:** Learn a representation T that maximally compresses X (min I(X;T)) while retaining everything predictive of Y (max I(T;Y)); the trade-off curve is the bottleneck, formalizing what a "sufficient, minimal" representation is.

## The formula / definition

Minimize over the stochastic encoder p(t|x):

  L_IB = I(X;T) − β·I(T;Y)

- I(X;T) = compression term (bits T keeps about input) — push down
- I(T;Y) = relevance term (bits T keeps about label) — push up
- β ≥ 0 = trade-off knob. β→0: T collapses to constant. β→∞: T keeps all info (no compression).

The optimal encoder satisfies a self-consistent fixed point (Tishby–Pereira–Bialek, 1999):

  p(t|x) ∝ p(t) · exp(−β · D_KL[ p(y|x) ‖ p(y|t) ])

Sweeping β traces the **information plane** curve I(T;Y) vs I(X;T) — the optimal frontier of achievable representations.

## The variational bound (what actually gets trained)

Exact MI is intractable, so VIB (Alemi et al., 2017) optimizes a tractable bound with a variational decoder q(y|t) and marginal r(t):

  L = E[ −log q(y|t) ] + β · E[ D_KL[ p(t|x) ‖ r(t) ] ]

This is exactly the supervised analogue of a VAE objective: cross-entropy reconstruction of Y plus a KL "rate" penalty on the encoder, trained with the reparameterization trick.

## Where it appears

- **VIB / Deep VIB (Alemi 2017)** — the practical, gradient-trainable IB; the β·KL term doubles as a regularizer giving robustness to adversarial perturbations.
- **"IB theory of deep learning" (Shwartz-Ziv & Tishby 2017)** — claimed SGD has a fast fitting phase then a long compression phase in the information plane; sparked the debate below.
- **Disentanglement / β-VAE** — same rate-distortion structure (β on the KL); unsupervised cousin where Y is X itself.
- **Conditional entropy bottleneck, contrastive learning (InfoNCE), CEB** — modern self-supervised objectives are MI bounds with an implicit bottleneck framing.

## Common mistake

Treating "deep nets compress, therefore IB explains generalization" as established fact. The compression phase claim is contested: Saxe et al. (2018) showed it is an artifact of saturating (tanh) nonlinearities and the binning estimator used to measure MI — ReLU nets need not compress, and I(X;T) is technically infinite/ill-defined for deterministic continuous encoders. IB is a clean normative principle, not a confirmed description of what SGD does.

## See also
- [[kl-divergence]] — the relevance and compression terms are both KL-based; the IB fixed point is a KL projection
- [[reparameterization]] — how the variational IB encoder is made differentiable
- [[cross-entropy]] — the variational relevance term is exactly a cross-entropy decoder loss
