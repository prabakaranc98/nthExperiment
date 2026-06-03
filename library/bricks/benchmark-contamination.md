# Benchmark Contamination

**One-liner:** Leakage of eval/benchmark examples (or near-duplicates) into pretraining or fine-tuning corpora, inflating scores so they measure memorization rather than generalization — detected via n-gram overlap, canary strings, and membership inference.

## The definition

A model is contaminated on benchmark B if examples from B (test inputs, labels, or rephrasings) appear in its training data D. Effect: measured accuracy A_obs = A_gen + Δ_contam, where Δ_contam is uplift from memorization. Quantify by comparing performance on "clean" vs "dirty" splits:

  Δ = acc(seen examples) − acc(unseen / freshly collected examples)

Detection methods:
- **N-gram overlap:** flag a test example if a high-order n-gram (e.g. 8-/13-gram, GPT-3/Llama style) or a normalized substring appears in D. Classic decontamination filter.
- **Canaries:** insert a unique random GUID string into a doc; if the model can complete it, that doc was memorized (BIG-bench canary).
- **Membership inference:** test point likely trained-on if loss/perplexity is anomalously low vs a reference (Min-K% Prob, zlib ratio, reference-model LR).
- **Behavioral / order tests:** "Did the model see the gold answer?" probes; reorder MCQ options or mask the answer and check if accuracy collapses (a contaminated model recalls position/answer).

## Where it appears

- **GPT-4 / Llama / GPT-3 tech reports** — n-gram decontamination of GSM8K, MMLU, HumanEval; report contaminated-subset scores separately.
- **Min-K% Prob (Shi et al., 2023) & MIMIR** — token-probability membership inference used to audit pretraining sets for benchmark leakage.
- **Data Contamination Quiz / "Rephrased samples" (Yang et al., 2023)** — show that paraphrased GSM8K/MMLU evade n-gram filters yet still inflate scores, motivating LLM-judge and semantic-overlap detectors.
- **Dynamic/private benchmarks (LiveCodeBench, GPQA, SWE-bench-Verified, time-gated evals)** — collect items after the model's training cutoff to be contamination-proof.

## Common mistake

Assuming n-gram/exact-match decontamination makes a benchmark clean. It catches verbatim copies but misses paraphrases, translations, and reformatted variants — and surface-form overlap underestimates true leakage. Conversely, a non-zero overlap rate does not by itself prove inflated scores; you need the clean-vs-dirty performance gap to attribute uplift.

## See also
- [[decontamination]] — the filtering pipeline meant to prevent contamination
- [[membership-inference-training-data-extraction]] — the core detection primitive (was this example trained on?)
- [[benchmark-saturation-dynamic-private-benchmarks]] — dynamic/post-cutoff evals as the structural fix
