# Topic 11 · Data, Tokenization, Benchmarking & Training at Scale

*The most underrated frontier lever: how the corpus is built, segmented, mixed, and consumed — data quality and tokenization as first-class statistical design choices, not preprocessing — plus the benchmark suites and evaluation tooling you measure scaled models against.*

**Papers:** 25 · **Pace:** ~12 days at 2/day

---

## Paper Log

| # | Paper | Authors | Year | Status | Note | Blog | Exp |
|---|-------|---------|------|--------|------|------|-----|
| 1 | Neural Machine Translation with Subword Units (BPE) | Sennrich, Haddow & Birch | 2016 | queued | Byte-pair merges as fix for open vocabulary | — | — |
| 2 | SentencePiece | Kudo & Richardson | 2018 | queued | Reversible, language-agnostic tokenization | — | — |
| 3 | Subword Regularization | Kudo | 2018 | queued | Sampling segmentations as regularization | — | — |
| 4 | Exploring the Limits of Transfer Learning (T5 / C4) | Raffel et al. | 2019 | queued | Text-to-text frame + web corpus built by explicit filtering | — | — |
| 5 | The Pile | Gao et al. | 2020 | queued | 800GB diverse corpus; composition drives capability | — | — |
| 6 | Deduplicating Training Data Makes LMs Better | Lee et al. | 2021 | queued | Dedup improves perplexity and cuts memorization | — | — |
| 7 | Scaling Laws and Interpretability of Learning from Repeated Data | Hernandez et al. | 2022 | queued | Repetition triggers double-descent-like collapse | — | — |
| 8 | Beyond Neural Scaling Laws (Data Pruning) | Sorscher et al. | 2022 | queued | Pruning to the right examples beats power-law scaling | — | — |
| 9 | A Pretrainer's Guide to Training Data | Longpre et al. | 2023 | queued | Controlled study of data age, quality filtering, toxicity | — | — |
| 10 | DoReMi | Xie et al. | 2023 | queued | Domain-mixture weights from distributionally-robust optimization | — | — |
| 11 | SemDeDup | Abbas et al. | 2023 | queued | Semantic-embedding dedup removes redundancy cheaply | — | — |
| 12 | Textbooks Are All You Need (phi-1) | Gunasekar et al. | 2023 | queued | Curated "textbook-quality" data substitutes for raw scale | — | — |
| 13 | RefinedWeb | Penedo et al. | 2023 | queued | Properly filtered web-only data matches curated corpora | — | — |
| 14 | Scaling Data-Constrained Language Models | Muennighoff et al. | 2023 | queued | Scaling law when tokens, not compute, are the bottleneck | — | — |
| 15 | MEGABYTE | Yu et al. | 2023 | queued | Multiscale patches model million-byte sequences | — | — |
| 16 | DataComp-LM (DCLM) | Li et al. | 2024 | queued | Model-based filtering dominates dataset quality | — | — |
| 17 | The FineWeb Datasets | Penedo et al. | 2024 | queued | Filtering/dedup ablations behind 15T-token open web corpus | — | — |
| 18 | Byte Latent Transformer (BLT) | Pagnoni et al. | 2024 | queued | Dynamic entropy-based byte patches replace fixed tokenization | — | — |
| 19 | What Do BPE Tokenizers Reveal About Their Training Data? | Hayase et al. | 2024 | queued | Merge order leaks data-mixture proportions | — | — |
| 20 | Muon Is Scalable for LLM Training | Moonshot/Kimi Team | 2025 | queued | Matrix-orthogonalized updates; ~2× compute efficiency over AdamW | — | — |
| 21 | Measuring Massive Multitask Language Understanding (MMLU) | Hendrycks et al. | 2020 | queued | Broad knowledge/reasoning benchmark | — | — |
| 22 | Beyond the Imitation Game (BIG-bench) | Srivastava et al. | 2022 | queued | 200+ tasks probing where capability breaks | — | — |
| 23 | Challenging BIG-Bench Tasks (BBH) | Suzgun et al. | 2022 | queued | The hard subset and how prompting changes the measured number | — | — |
| 24 | Evaluating LLMs Trained on Code (HumanEval) | Chen et al. | 2021 | queued | Code benchmark and pass@k estimator | — | — |
| 25 | Lessons from the Trenches on Reproducible Evaluation (lm-eval) | Biderman et al. | 2024 | queued | Why benchmark numbers swing on prompt/normalization choices | — | — |

---

## Synthesis Notes
## Blog Post
## Experiments
