# Automated Interpretability / Auto-Labeling

**One-liner:** Use an LLM to generate a natural-language explanation for what a feature/neuron does, then *score* that explanation by how well it predicts the feature's activations — letting you label millions of SAE features without humans in the loop.

## The two-stage loop

**1. Explain.** Show an explainer LLM the top-activating text examples for feature *f* (token contexts + per-token activation values). It emits a short hypothesis, e.g. "fires on legal citations."

**2. Score.** Quantify how well the explanation predicts activations. Two dominant protocols:

- **Simulation / correlation scoring** (OpenAI "neurons" 2023): a simulator LLM, given only the explanation, predicts the activation at each token; score = Pearson/EV correlation between predicted and true activations.
- **Detection scoring** (cheaper, EleutherAI 2024): given the explanation + a context, the LLM does **binary** classify "does *f* fire here?" Score = balanced accuracy / AUC over activating vs. random-negative contexts.
- **Fuzzing**: a harder detection variant — distinguish texts where *f* actually fired from *confusable* texts (random tokens highlighted), penalizing explanations that are vacuously broad.

```
score(f) ≈ accuracy of [explanation → predict activation] on held-out contexts
```

## Where it appears

- **OpenAI "Language models can explain neurons"** (2023) — GPT-4 explains + simulates every neuron in GPT-2; introduced the correlation-based auto-interp score.
- **EleutherAI `delphi`/`sae_auto_interp`** (2024) — scalable detection + fuzzing scoring; the practical pipeline for labeling SAE dictionaries.
- **SAEBench** (2024-25) — standardized auto-interp scores alongside sparsity, downstream-loss, and feature-disentanglement metrics to compare SAE training recipes.
- **Anthropic / Gemma Scope** — auto-interp labels attached to released SAE feature catalogs for browsing.

## Common mistake

Treating a high score as proof the feature *is* the explained concept. Detection scoring rewards explanations that merely **co-fire** with activations; a too-broad label ("text about people") and a confounded one can both score well. Scores measure predictive sufficiency on a narrow distribution — not causal necessity, not faithfulness, and not monosemanticity. Always pair with fuzzing/causal checks.

## See also
- [[sparse-autoencoders]] — produces the millions of features this pipeline labels
- [[llm-as-a-judge]] — same explain-then-score machinery; shares its biases (verbosity, self-preference)
- [[faithfulness-completeness-of-explanations]] — what a high auto-interp score does and does not certify
