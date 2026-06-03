# Weight Averaging / EMA & Model Soups

**One-liner:** Average weights — either across training steps (EMA) or across independently fine-tuned checkpoints (soups) — to land in a flatter, better-generalizing region at near-zero extra inference cost, exploiting the near-linear-mode-connectivity of models sharing a pretrained init.

## The formulas

**EMA (across steps, online):** maintain a shadow copy that tracks the trajectory.

θ_ema ← β · θ_ema + (1−β) · θ_t,  with β ≈ 0.999–0.9999

Equivalent to a low-pass filter over the SGD trajectory; reduces variance from the noisy late-training oscillation. Used as the *evaluation* weights, not the training weights.

**Soup (across checkpoints, offline):** average N models fine-tuned from the same init with different seeds/hyperparams/data order.

θ_soup = (1/N) Σ_i θ_i   (uniform soup)

**Greedy soup:** sort candidates by val accuracy; add a model to the average only if it improves held-out accuracy. Almost always beats the uniform soup and the single best model.

## Where it appears

- **Model Soups (Wortsman et al., 2022)** — averaging CLIP/ViT fine-tunes beats the best single run and ensembling, with one model's inference cost.
- **Diffusion training** — EMA weights are *the* sampling weights; raw weights give visibly worse samples. EDM/Karras use power-function EMA with post-hoc EMA-length tuning.
- **WSD / annealing schedules** — averaging checkpoints across the decay phase (LAWA, "checkpoint merging") substitutes for or augments LR decay.
- **SWA (Izmailov et al., 2018)** — the SGD-trajectory precursor: average periodic snapshots along a cyclic-LR tail.
- **Task arithmetic / merging** — soups are the same operation as task-vector merging when models share an init.

## Common mistake

Averaging weights of models that do *not* share a pretrained initialization (or differ wildly). Weight averaging only works inside a single low-loss basin — it relies on (linear) mode connectivity. Average two from-scratch runs and you get garbage; permutation alignment (Git Re-Basin) is needed first. Also: EMA decay must be coupled to step count — a fixed β=0.9999 is far too slow for short runs.

## See also
- [[task-arithmetic-task-vectors]] — soups are uniform averaging; task vectors are the directional, signed generalization
- [[sharpness-aware-minimization-flat-minima]] — both seek flat minima; averaging finds them implicitly
- [[lora-merging-multi-adapter-serving]] — averaging adapters is the parameter-efficient analog of souping
