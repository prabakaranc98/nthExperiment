# Near-Duplicate Deduplication (MinHash / SemDeDup)

**One-liner:** Remove exact, fuzzy, and semantic duplicates from pretraining corpora via suffix-array substring matching, MinHash-LSH, and embedding-cluster pruning — one of the single largest levers on final quality, training efficiency, and memorization.

## The methods (three tiers)

**Exact / substring (suffix array):** Build a suffix array over the concatenated corpus; flag any duplicated substring longer than k tokens (Lee et al. 2021 "Deduplicating Training Data" use k=50). Removes verbatim repeats and boilerplate.

**Fuzzy (MinHash + LSH):** Shingle each doc into n-grams. For r independent hash functions, the MinHash is `h_i(doc) = min over shingles s of hash_i(s)`. Property: `P[h_i(A)=h_i(B)] = J(A,B) = |A∩B| / |A∪B|` (Jaccard). Stack r hashes into a signature, split into b bands of rows-per-band each; docs sharing any identical band collide → candidate pair. Tune (b, rows) so the LSH S-curve threshold `t ≈ (1/b)^(1/rows)` matches the target Jaccard (e.g. t=0.8). Used by GPT-3, The Pile, RefinedWeb, C4, FineWeb.

**Semantic (SemDeDup, Abbas et al. 2023):** Embed docs (e.g. with a sentence/CLIP encoder), k-means cluster, then within each cluster drop points whose cosine similarity to a kept neighbor exceeds threshold. Catches paraphrases and templated content that share little surface n-gram overlap. Pruned 50% of LAION with no loss; SemDeDup + density pruning ("D4") beat random on LLM pretraining.

## Where it appears

- RefinedWeb / FineWeb — aggressive MinHash-LSH dedup is the headline reason web-only data matched curated corpora
- Lee et al. 2021 — exact substring dedup cut memorized verbatim emission ~10x and improved perplexity
- SemDeDup / D4 (Meta) — embedding dedup as a data-efficiency multiplier, ~20% fewer steps to same loss
- The Pile, C4, Dolma, Llama 3 pipelines — multi-stage exact→fuzzy dedup as standard practice

## Common mistake

Conflating deduplication with decontamination. Dedup removes train↔train redundancy for quality/efficiency; decontamination removes train↔test overlap to keep evals honest — they use overlapping tooling (n-gram/MinHash) but solve different problems. Also: dedup too hard and you delete legitimately rare-but-repeated high-value content (code license headers, math identities); the threshold is a quality knob, not "more is always better."

## See also
- [[decontamination]] — same matching machinery, but against eval sets not the train set
- [[memorization-vs-generalization]] — dedup is the primary lever for suppressing verbatim memorization
- [[data-constrained-scaling-repetition-laws]] — what dedup leaves determines how many epochs of unique tokens you actually have
