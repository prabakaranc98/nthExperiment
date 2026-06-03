# Memorization vs Generalization

**One-liner:** LLMs provably store verbatim training data — extractable via prompting, scaling super-linearly with duplication and model size — a distinct phenomenon from learning generalizing structure, with sharp privacy and copyright stakes.

## The key insight

Memorization and generalization coexist; they are not opposite ends of one axis. A sequence is **k-eligible / extractable** if a prompt of the right prefix elicits the verbatim continuation under greedy decoding (Carlini et al. 2021/2022 "Quantifying Memorization"). Empirical scaling:

- Memorized fraction grows ~log-linearly with **model size** (bigger = memorizes more).
- Grows with **duplication count** of a sequence in the corpus (a string seen 1000x is far more extractable than one seen once) — the strongest single predictor.
- Grows with **prefix length** available to the attacker.

Capacity estimate (Morris et al. 2025, "How much do LMs memorize"): GPT-family models store roughly **~3.6 bits per parameter** of memorized data; once dataset bits exceed capacity the model is forced to generalize — the grokking/double-descent transition. Distinguish **unintended verbatim memorization** (privacy/copyright risk) from **necessary memorization** of facts (a feature, not a bug).

## Where it appears

- Training Data Extraction attacks (Carlini 2021) — recovered PII, code, URLs verbatim from GPT-2 by prompting + membership scoring.
- "Scalable Extraction" (Nasr et al. 2023) — divergence attack made aligned ChatGPT emit training data; basis for NYT-vs-OpenAI copyright evidence.
- Deduplication (Lee et al. 2021) — removing near-duplicates from C4/Pile cuts memorization ~10x with no quality loss; now standard pretraining hygiene.
- Differential privacy / DP-SGD and machine unlearning — mitigations bounding or removing per-example influence.

## Common mistake

Treating memorization as pure overfitting that more data or regularization eliminates. It is largely orthogonal to test loss: well-generalizing frontier models still memorize duplicated/rare sequences, and you cannot infer extraction risk from the train/val gap.

## See also
- [[membership-inference-training-data-extraction]] — the attacks that operationalize and measure memorization
- [[near-duplicate-deduplication]] — the primary corpus-level mitigation
- [[grokking]] — the memorization-to-generalization phase transition during training
