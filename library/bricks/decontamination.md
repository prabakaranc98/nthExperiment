# Decontamination (Train-Test Overlap Removal)

**One-liner:** Detect and strip pretraining documents that overlap eval benchmarks (n-gram / substring / embedding / canary matching) so reported scores reflect generalization, not memorized test items leaked through the web crawl.

## The methods (in order of strictness)

1. **n-gram overlap (GPT-3/PaLM style):** flag a train doc if it shares an exact n-gram (typically n=8–13 tokens) with any eval example. PaLM dropped docs with ≥70% of their 8-grams found in any benchmark. Llama/most labs: remove the eval example or the train doc on a high overlap fraction.
2. **Substring / longest-match (The Pile, RedPajama):** suffix-array or MinHash lookup of the test string inside the corpus; flag if a long contiguous substring (e.g. ≥50 chars) matches.
3. **Embedding / fuzzy (2024+):** cosine similarity of doc embeddings > τ catches paraphrased or reformatted leakage that exact n-grams miss.
4. **Canary strings:** publishers embed a fixed GUID in the benchmark (e.g. BIG-bench). If the model can reproduce the canary, that benchmark was in training → exclude.

Decision: `contaminated = max_overlap(train_doc, eval_set) > threshold`. Either drop the train doc (clean the model) or drop the leaked eval items and re-score (clean the report).

## Where it appears

- **GPT-3 / GPT-4** — post-hoc "clean" eval splits: re-score only on eval examples with no n-gram overlap, report the gap vs. dirty scores.
- **Llama 2/3, OLMo, Dolma** — corpus-side filtering with documented n-gram + paragraph-overlap thresholds before pretraining.
- **GSM8K → GSM1k, LiveCodeBench, LiveBench (2024–2026)** — released *after* training cutoffs precisely so contamination is structurally impossible; a large gap to the static benchmark is the contamination signal.

## Common mistake

Treating decontamination as solved by exact n-gram matching. It catches verbatim copies but misses translated, reformatted, or LLM-paraphrased benchmark items — and is gamed by training on near-duplicates just below the threshold. Absence of n-gram overlap is *not* proof a benchmark is clean.

## See also
- [[benchmark-contamination]] — the failure mode decontamination defends against
- [[near-duplicate-deduplication]] — same MinHash/suffix-array machinery, applied corpus-internally
- [[membership-inference-training-data-extraction]] — canary/extraction probes used to detect contamination after the fact
