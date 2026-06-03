# Model-Based Quality Filtering

**One-liner:** Train a lightweight classifier to score web documents and keep only high-quality ones (FineWeb-Edu's educational-value regressor, DCLM's fastText reference-model filter) — the dominant 2024+ pretraining-data quality lever, replacing hand-tuned heuristics.

## The recipe

1. **Get labels.** Prompt a strong LLM (Llama-3-70B, GPT-4o) to score a sample of N docs (~hundreds of thousands) on a target axis — e.g. FineWeb-Edu asks for an integer 0–5 "educational value" score. Or label by *source*: positive = curated/instruction-like text (OpenHermes, ELI5), negative = random Common Crawl (DCLM's contrastive setup).
2. **Train a cheap scorer.** Fit a fastText linear classifier on n-gram features, or distill the LLM scores into a small BERT/embedding + linear head (FineWeb-Edu uses Snowflake-arctic-embed + linear regression).
3. **Threshold & keep.** Score the full corpus (trillions of tokens — must be cheap), keep docs above a cutoff. FineWeb-Edu keeps score ≥ 3.

The scorer must run at >>1M docs/sec/node, hence fastText/embedding-linear rather than an LLM pass over the whole web.

## Why it beats heuristics

Heuristic filters (length, symbol ratio, stopword counts, blocklists) are coarse and uncorrelated with downstream benchmark gains. A model-based filter learns the actual signal. FineWeb-Edu (1.3T tokens) matched or beat much larger raw FineWeb on MMLU/ARC; DCLM's fastID filter was the single largest contributor in their 2024 ablation suite.

## Where it appears

- **FineWeb-Edu (2024)** — LLM-scored educational classifier; the canonical open recipe, drove the FineWeb → FineWeb-Edu quality jump.
- **DCLM-Baseline (2024)** — fastText classifier trained to separate OH-2.5/ELI5 from raw Crawl; benchmark showed it dominated other filtering choices.
- **Llama 3, phi (textbook-quality), Nemotron-CC** — quality classifiers / model-judged filtering as a core curation stage.

## Common mistake

Pushing the threshold too high to maximize average doc quality — this collapses diversity and over-filters domains (code, multilingual, niche knowledge) the LLM labeler underrates, hurting broad benchmarks. Aggressive single-classifier filtering also amplifies the labeler's biases. The goal is the *downstream eval*, not the average score; tune the cutoff against ablations and combine with dedup, not as a standalone maximizer.

## See also
- [[synthetic-data-web-rephrasing]] — the other 2024 quality lever; often paired with filtering
- [[pretraining-data-curriculum]] — quality-filtered shards feed mid-training/annealing phases
- [[near-duplicate-deduplication]] — the complementary curation stage; quality + dedup together
