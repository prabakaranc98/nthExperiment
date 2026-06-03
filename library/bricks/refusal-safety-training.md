# Refusal & Safety Training

**One-liner:** Post-training (SFT + preference optimization) that teaches a model to decline harmful requests — but the refusal behavior is often mediated by a *single linear direction* in the residual stream, so ablating that direction (abliteration) removes refusals while leaving capabilities intact, exposing how shallow the safety layer is.

## The key insight

Refusal is approximately a one-dimensional feature. Arditi et al. (2024), "Refusal in LLMs is mediated by a single direction":

1. **Find the direction.** Compute mean residual-stream activations on harmful vs. harmless prompts at some layer ℓ, take the difference-of-means:
   r = mean(h_harmful) − mean(h_harmless)   (then normalize: r̂ = r/‖r‖)
2. **Ablate it (abliteration).** Project the refusal direction *out* of every residual-stream write (attention/MLP outputs and the stream itself), at all token positions and layers:
   h' = h − (h · r̂) r̂
   → model stops refusing, near-zero capability loss, no fine-tuning needed (weight-orthogonalization makes it permanent: W' = W − r̂ r̂ᵀ W).
3. **Add it (steering).** Conversely, adding +α·r̂ induces refusal even on benign prompts.

This is a concrete instance of the **linear representation hypothesis**: a behavioral concept = a direction. Safety training installs the direction; it does not deeply entangle refusal with the rest of the network.

## Where it appears

- **Arditi et al. (2024)** — difference-of-means refusal direction; abliteration jailbreak. Spawned the "abliterated" model genre (uncensored open-weight checkpoints on HF) by orthogonalizing weights.
- **RLHF / DPO / Constitutional AI (RLAIF)** — the *mechanisms* that install refusals: harmlessness preference pairs, constitutional self-critique, refusal SFT data. Anthropic HH, Llama-Guard, safety-tuned chat models.
- **Adversarial robustness work** — refusal brittleness explains why GCG suffix attacks, fine-tuning attacks (10 examples can strip safety — Qi et al. 2023), and many-shot jailbreaks succeed: they shift activations off the refusal direction.
- **Deep vs. shallow safety alignment (Qi et al. 2024)** — refusal is concentrated in the first few generated tokens ("shallow safety alignment"); robust training must extend safety deeper into the response.

## Common mistake

Believing refusal training makes the model *unable* to produce harmful content. It only adds a thin, often linear gate on top of latent capabilities that remain fully present in the weights. Safety post-training suppresses; it does not unlearn. (For actual removal, see machine unlearning — and even that is contested.)

## See also
- [[steering-vectors-activation-steering]] — refusal direction is a steering vector; add to induce, ablate to remove
- [[linear-representation-hypothesis]] — why a single direction can encode the refusal behavior
- [[rlaif-constitutional-ai]] — the post-training recipe that installs refusals in the first place
