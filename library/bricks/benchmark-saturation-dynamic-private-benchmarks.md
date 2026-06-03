# Benchmark Saturation & Dynamic/Private Benchmarks

**One-liner:** Static public benchmarks get topped and contaminated within months, so evaluation moves to held-out private test sets, periodically refreshed live leaderboards, and deliberately harder frontier suites whose scores then saturate again — a perpetual treadmill.

## The key insight

Two failure modes drive saturation:
1. **Ceiling saturation** — headroom 1 − score → 0, so the metric stops discriminating between models (gap is within eval noise). Once SOTA passes ~90%, the remaining points are mostly label errors and ambiguity.
2. **Contamination/Goodhart** — once a test set is public, it leaks into pretraining corpora and gets optimized against, so reported score = true capability + memorization + train-on-test. The benchmark "ceases to be a good measure" (Goodhart).

Mitigations, by construction:
- **Held-out private split:** maintainer keeps test labels secret; you submit predictions (Kaggle-style). Blocks training-on-test but allows score-chasing via repeated submissions → use **submission limits** + a final once-only blind set.
- **Dynamic / live refresh:** new items drawn from a stream *after* a model's training cutoff, so contamination is provably impossible by date. Report score on items with `release_date > model_cutoff`.
- **Canary strings + n-gram overlap audits** to detect leakage post hoc (see decontamination).
- **Adversarial / human-vs-model loops:** items selected because current models fail them (Dynabench, ARC-AGI-style).
- **Pairwise human preference** (Chatbot Arena) — no fixed gold answer to memorize; ranks via Elo/Bradley-Terry on fresh live votes.

A benchmark's useful lifetime ≈ time until SOTA crosses the discrimination ceiling; mid-2020s this is often < 12 months.

## Where it appears

- **GPQA / "Humanity's Last Exam" / FrontierMath** — built deliberately hard (Google-proof, expert/research-level) to restore headroom after MMLU, GSM8K, HumanEval saturated.
- **LiveCodeBench / LiveBench / SWE-bench (+ verified/live variants)** — timestamped problems released continuously; only post-cutoff items count, defeating contamination.
- **Chatbot Arena (LMArena)** — live crowd-sourced pairwise battles, Elo updated continuously; the de facto dynamic ranking when static scores cluster at the top.
- **ARC-AGI-1/2** — private eval set; public training set only, to test generalization not recall.
- **Kaggle / ImageNet-style hidden test servers** — held-out labels, capped submissions.

## Common mistake

Reading a near-ceiling delta (e.g. 91.2 vs 90.8 on MMLU) as a real capability gap. Near saturation, the headroom is dominated by mislabeled/ambiguous items and the gap is well inside eval noise — confidence intervals overlap. Also: assuming a high public-benchmark score reflects capability rather than possible contamination; always check whether the test set predates the model and whether a contamination audit was run.

## See also
- [[benchmark-contamination]] — the leakage mechanism that motivates private/live test sets
- [[decontamination]] — n-gram/canary pipelines used to detect and scrub leaked test items
- [[elo-online-rating-for-model-ranking]] — the scoring backbone of live pairwise arenas
