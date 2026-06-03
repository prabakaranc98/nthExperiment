# Semantic Entropy for Hallucination Detection

**One-liner:** Detect LLM confabulation by sampling N generations for a prompt, clustering them by bidirectional entailment (same meaning), and computing entropy over *semantic* clusters rather than tokens — high entropy flags answers the model is uncertain about, not just paraphrase variation.

## The formula / definition

Sample M answers s₁..s_M at temperature > 0. Cluster into semantic equivalence classes C using a bidirectional-entailment NLI check: sᵢ and sⱼ share a cluster iff entail(sᵢ→sⱼ) AND entail(sⱼ→sᵢ) (both directions, conditioned on the question). Aggregate per-sequence likelihoods into cluster probabilities:

  p(C) = Σ_{s ∈ C} p(s | x),  with p(s|x) = ∏ₜ p(sₜ | s_<t, x)  (length-normalized in practice)

Semantic entropy:

  SE(x) = − Σ_C p(C) log p(C)

High SE → many distinct meanings → likely confabulation. Discrete/black-box variant (no logits): estimate p(C) by cluster frequency over samples. Threshold SE (e.g. via AUROC sweep) to make the hallucination decision.

## Where it appears

- **Kuhn et al. (ICLR 2023)** — original semantic entropy; showed token-level (naive) entropy fails because it conflates lexical and meaning variation.
- **Farquhar et al. (Nature 2024)** — scaled it to detect *confabulations* (arbitrary, wrong, prompt-sensitive answers) across QA/biography/medical; beats sequence-likelihood and p(True) baselines; introduced the discrete sample-frequency estimator for closed models.
- **SAR / Semantic Entropy Probes (SEPs, 2024)** — linear probes on a single forward pass approximate SE cheaply, avoiding N samples + NLI.

## Common mistake

Treating high token-level / sequence entropy as the signal. A model can be perfectly *certain in meaning* while spreading probability over many surface forms ("Paris" / "It's Paris" / "The capital is Paris") — naive entropy is high, semantic entropy is ~0. You must collapse paraphrases via entailment first; otherwise you measure linguistic invariance, not epistemic uncertainty. Also: SE targets *epistemic* confabulation, not factuality of confidently-wrong memorized errors.

## See also
- [[epistemic-vs-aleatoric-uncertainty]] — SE estimates epistemic uncertainty; aleatoric noise (paraphrase) is what clustering removes
- [[self-consistency]] — also samples N generations and aggregates by answer agreement, but for accuracy not uncertainty
- [[calibration]] — SE is a calibration/confidence signal; thresholding trades coverage vs. error rate
