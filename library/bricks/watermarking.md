# Watermarking (LLM Output)

**One-liner:** Embed a statistically detectable signal into generated text by biasing token sampling at decode time (green-list logit boost, or sampling-based SynthID-Text) so a key-holder can verify provenance via a hypothesis test; paraphrase robustness is the open weakness.

## The formula / definition

**Kirchenbauer et al. (2023) — green/red list.** At each step, hash the previous token(s) with a secret key to seed a PRNG; partition the vocabulary into a green list (fraction γ) and red list. Add a constant δ to green-list logits before softmax:

  logit'_v = logit_v + δ·1[v ∈ green]

Detection (no model needed, just the key): count green tokens among T generated tokens. Under the null (no watermark), green count ~ Binomial(T, γ). Use the z-statistic

  z = (|s|_G − γT) / sqrt(T·γ(1−γ))

Reject H₀ (declare "watermarked") when z exceeds a threshold; the per-token false-positive rate is controlled analytically.

**SynthID-Text (Dadalto et al., DeepMind, Nature 2024).** Tournament sampling: draw multiple candidate tokens, score each with key-seeded random g-functions, keep the tournament winner. Distortion-free / low-distortion variant preserves the output distribution (in expectation) while still implanting a detectable bias; detection sums the g-scores.

## Where it appears

- **Kirchenbauer et al. 2023 (green-list)** — the canonical logit-bias scheme; δ and γ trade detectability vs. text-quality degradation.
- **SynthID-Text (Gemini, 2024)** — deployed in production on Gemini outputs; tournament sampling with a distortion-free mode, released open-source.
- **Aaronson / OpenAI Gumbel scheme** — exponential-minimum-sampling cryptographic watermark; provably distortion-free, key seeds the Gumbel noise.
- **Image/video analogue** — SynthID watermarks Imagen/Veo pixels; same provenance goal, different signal carrier.

## Common mistake

Conflating watermarking with detection of arbitrary AI text. Watermarking only flags text from *your own* keyed generator — it says nothing about un-watermarked outputs from other models. It is also brittle: paraphrasing, translation, or token-level editing dilutes the green-token statistic and can drop z below threshold; high entropy is required (low-entropy/templated text can't carry a strong signal).

## See also
- [[decoding-sampling-strategies]] — watermarks are implemented as a modified sampling rule at decode time
- [[membership-inference-training-data-extraction]] — adjacent provenance/attribution problem on the training-data side
- [[differential-privacy]] — shares the formal-guarantee-under-an-adversary framing for ML outputs
