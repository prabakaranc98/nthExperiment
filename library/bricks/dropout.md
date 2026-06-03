# Dropout

**One-liner:** Stochastic regularization that randomly zeros activations during training (and rescales the survivors) so the network can't co-adapt units — still everywhere in 2024-2026 as attention/residual dropout and as the basis for MC-dropout uncertainty.

## The formula / definition

For a layer activation h with keep probability p (drop rate 1−p), during **training**:

m ~ Bernoulli(p), elementwise
h̃ = (m ⊙ h) / p     ← "inverted dropout": divide by p so E[h̃] = h

At **inference**: no mask, no scaling — just use h. The /p at train time means the expected activation matches, so no rescale is needed at test time (this is why modern frameworks default to inverted dropout).

Interpretation: training ≈ implicit ensemble over 2^n thinned subnetworks sharing weights; a single deterministic forward pass at test time approximates the geometric-mean ensemble.

## Where it appears

- **Transformers** — dropout on attention weights (post-softmax) and on residual/MLP outputs; original Transformer used p_drop=0.1. Large LLM pretraining often sets dropout to **0** (data is plentiful, one epoch, no overfit) but re-enables it for SFT/fine-tuning on small data.
- **MC-dropout (Gal & Ghahramani 2016)** — keep dropout ON at test time, run K stochastic forward passes; sample mean = prediction, sample variance = epistemic uncertainty. Cheap Bayesian approximation.
- **DropPath / stochastic depth** — drops whole residual blocks (ViT, ConvNeXt, deep nets); the structured cousin used heavily in vision.
- **LoRA / adapter fine-tuning** — lora_dropout on the low-rank update is a standard regularizer.

## Common mistake

Forgetting to switch modes — calling MC-dropout aside, you must set the model to eval (`model.eval()` / `training=False`) at inference so the mask is disabled and no scaling double-counts. A model left in train mode gives noisy, degraded predictions. Conversely, MC-dropout requires *deliberately* keeping it on; plain eval-mode dropout gives you point predictions, not uncertainty.

## See also
- [[deep-ensembles-mc-dropout]] — MC-dropout as a cheap, single-model uncertainty estimator vs. true ensembles
- [[batch-norm]] — the other classic regularizer/normalizer; interacts badly with dropout (variance shift)
- [[epistemic-vs-aleatoric-uncertainty]] — what MC-dropout's predictive variance actually captures
