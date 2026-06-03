# Label Smoothing

**One-liner:** Replace one-hot targets with a mixture of the hard label and a uniform distribution — softening the target curbs overconfidence, improves calibration, and acts as a regularizer (Szegedy et al., 2016).

## The formula

For K classes, smoothing parameter ε, and true class y, the soft target is:

q'(k) = (1 − ε)·δ(k=y) + ε/K

So the correct class gets 1 − ε + ε/K and every other class gets ε/K. Typical ε ≈ 0.1.

Equivalently, the loss is a blend of cross-entropy against the hard label and against the uniform distribution u:

L = (1 − ε)·H(δ_y, p) + ε·H(u, p)
  = CE_hard + ε·(KL(u ‖ p) + const)

The second term penalizes the model for placing too little mass everywhere except y — i.e. it pulls logits away from ±∞.

## The geometric effect

Müller, Kornblith, Hinton (2019, "When Does Label Smoothing Help?") showed it makes the penultimate-layer representations of same-class examples cluster into tight equidistant groups, shrinking logit gaps between the correct and incorrect classes. This is why it helps calibration: the model stops driving the max softmax probability to ~1.

## Where it appears

- **Inception-v3 / vision classifiers** (Szegedy 2016) — original use; ε=0.1 on ImageNet for a small top-1 gain and regularization.
- **Transformer / "Attention Is All You Need"** (Vaswani 2017) — ε_ls=0.1 on NMT; hurts perplexity but improves BLEU and accuracy. Standard in LM/seq2seq training recipes since.
- **Calibration literature** — used as a baseline against temperature scaling; smoothing reduces ECE but can over-smooth.

## Common mistake

Stacking label smoothing on top of a process where you later need the model's logits/features as a teacher. Müller et al. found smoothing *erases* the inter-class similarity structure in logits, so a label-smoothed teacher distills *worse* than an un-smoothed one — even though it is itself more accurate. Also: people forget smoothed CE no longer has loss floor 0, so absolute loss/perplexity values aren't comparable to un-smoothed runs.

## See also
- [[cross-entropy]] — label smoothing is just CE against a softened target distribution
- [[calibration]] — primary motivation; reduces overconfidence and ECE
- [[knowledge-distillation]] — interacts badly: smoothed teachers transfer less dark knowledge
