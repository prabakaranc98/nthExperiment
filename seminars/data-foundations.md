# Data Foundations for Frontier AI

*Doc 5 of the FAIRE maps — the data pipeline that frontier capability is actually made of.*

**FAIRE context:** a companion to the [Statistical & Probabilistic Foundations](statistical-probabilistic-foundations.md) map, slotted into the [Curriculum Index](curriculum-index.md). Read it as terrain, not a textbook: each section names the core idea, the canonical systems, and the on-the-ground practice.

Architectures have converged — decoder-only transformer, MoE, long context, RoPE — and a transformer is a near-universal function approximator, so what a model *can* do is bounded by what it has *seen*, not by how it is wired. In the frontier era the differentiating asset is therefore the data pipeline: what you pretrain on, how you stage it (pre → mid → post), how you manufacture it synthetically, how you verify it, and how you measure progress. Data is simultaneously the moat (a compounding, exclusive supply that weights and recipes leak but usage does not) and the bottleneck (high-quality public text is a depleting reserve). This document is the conceptual spine of that claim.

---

## The one idea

> **Once algorithms commoditize, data is the only remaining axis of differentiation — and "data" means a *staged, filtered, manufactured, and verified* distribution, not a pile of crawl.** Capability is a property of the pretraining distribution; alignment is a style shift over it; reasoning is what survives a verifier. The binding constraints are running out of high-quality tokens and being able to *verify* the tokens you manufacture. Everything below is one pipeline for engineering and trusting that distribution.

### The data lifecycle

Three sequential stages, three cross-cutting layers that touch all of them.

| Layer | Stage / role | What it sets |
|---|---|---|
| **Stage 1** | **Pretraining** | the base distribution — latent skills, factual coverage, language balance, the prior |
| **Stage 2** | **Mid-training** | annealing + continued pretraining + long-context — *which* capabilities cross into post-training |
| **Stage 3** | **Post-training** | instructions, preferences, verifiable rewards — the assistant and the reasoner |
| *cross-cut* | **Synthetic data** | manufactures tokens for every stage once the web is exhausted |
| *cross-cut* | **Simulators / verifiers** | supply the correctness signal that makes RL and synthetic loops net-positive |
| *cross-cut* | **Evaluation** | the control loop — the measurement every claim above reduces to |

### Navigation

| Section | What you get |
|---|---|
| [Pretraining data](#pretraining-data--the-base-distribution) | sources, the curation pipeline, mixtures, tokenization, scaling, the data wall |
| [Data engineering — the training data path](#data-engineering--the-training-data-path) | how raw bytes become token batches: collect → clean → engineer → load, with tool links |
| [Mid-training](#mid-training--annealing-continued-pretraining-and-long-context) | annealing, continued pretraining, long-context data, curriculum |
| [Post-training data](#post-training-data--instructions-preferences-verifiable-rewards) | SFT, preference data (RLHF/DPO), RLVR, and the data-borne failure modes |
| [Synthetic data](#synthetic-data--manufacturing-the-training-set) | generation methods, self-improvement loops, model collapse, when it helps |
| [Simulators, environments, verifiers](#simulators-environments-and-verifiers--the-verification-signal) | the verifier taxonomy, agentic sandboxes, world models, sim-to-real |
| [Evaluation and measurement](#evaluation-and-measurement--data-that-tells-you-if-it-worked) | construct validity, contamination, LLM-judges, the statistics of evals |
| [Data quality, governance, economics](#data-quality-governance-and-economics) | data-centric AI, provenance, the legal frontier, token economics, the flywheel |
| [What the open models teach us](#what-the-open-models-teach-us--open-data-code-weights-architecture) | the fully-open releases (data + code + weights) that ground every claim here |

---

## Pretraining data — the base distribution

Base-model capability is a property of the pretraining distribution far more than of the architecture. The pretraining corpus fixes the prior — the latent skills, factual coverage, language balance, and reasoning patterns the model later post-training merely *elicits* and *aligns*. Get the base distribution wrong and no amount of RLHF recovers it.

### Where capability comes from

- Frontier base models train on **10–30T+ tokens**. Capability tracks (i) total high-quality tokens, (ii) *diversity* of skills represented, and (iii) the *mixture weights* across domains, more than raw crawl volume.
- Empirically, the highest-leverage interventions are **quality filtering** and **mixture design** — DCLM showed a strong classifier-filtered subset of Common Crawl beats much larger unfiltered corpora at fixed compute. Code and math in the mix lift general reasoning, not just code/math benchmarks (transfer).

### Sources: from raw crawl to curated open corpora

The open-data lineage is the public mirror of what frontier labs do internally.

| Corpus | Year | Scale | What it contributed |
|---|---|---|---|
| Common Crawl | ongoing | ~250B+ pages | Raw WARC/WET; the substrate, not a dataset |
| C4 / The Pile | 2020 | 0.8T / 0.4T | First reproducible filtered/heterogeneous mixes |
| RedPajama (v1/v2) | 2023 | 1.2T / ~30T | Open Llama-recipe replication; v2 ships quality *signals*, not decisions |
| RefinedWeb / FineWeb | 2023/24 | ~5T / 15T | Web-only can match curated mixes; aggressive filtering + dedup |
| FineWeb-Edu | 2024 | 1.3T | Model-scored *educational* subset; large downstream gains per token |
| DCLM-Baseline | 2024 | ~4T | Open data-curation *benchmark*; fastText-style classifier filtering as the lever |
| Nemotron-CC, Zyda-2, TxT360 | 2024–25 | multi-T | Crawl reprocessing + synthetic rephrasing + global dedup |

Beyond web: curated code (permissive-license GitHub, **The Stack v2** / StarCoder lineage), math (proof corpora, scraped + synthetic), books, papers (arXiv, PubMed), Wikipedia, and increasingly **synthetic rephrasings** of web text. Licensed and PDF-extracted (high-quality OCR) data is a growing, mostly-proprietary edge.

### The curation pipeline

Applied roughly in order; each stage is a filter on the empirical distribution.

- **Language ID** — fastText/CLD3 classifiers, thresholded confidence per doc. Determines multilingual balance.
- **Quality filtering** — two families:
  - *Heuristic*: C4/Gopher/RefinedWeb rules — symbol-to-word ratios, line-length, stopword presence, blocklists, boilerplate stripping (trafilatura). Cheap, high-recall, blunt.
  - *Model-based*: a lightweight classifier (fastText, or an LLM-as-judge distilled into a small scorer) trained to predict "high quality" — FineWeb-Edu and DCLM use this. The *choice of positive examples* (e.g. "looks like a textbook") largely defines the resulting model.
- **Deduplication** — removing near/exact copies improves quality and cuts memorization:
  - *Exact*: hashing / suffix-array substring dedup.
  - *Fuzzy*: **MinHash + LSH** on n-gram shingles (Jaccard near-dup).
  - *Semantic*: **SemDeDup** — embed docs, cluster, drop near-duplicate embeddings; catches paraphrase/template duplication MinHash misses. D4 extends this with density-based pruning.
- **Toxicity / PII / safety** filtering and domain blocklists close the pipeline.

### Data mixtures and domain weighting

The mixture (web vs. code vs. math vs. multilingual) is a first-class hyperparameter.

- **DoReMi** — train a small *proxy* model with group-DRO to find domain weights that minimize worst-case excess loss; transfer those weights to the large run. Reduces guesswork over hand-tuned mixes.
- **RegMix** — fit a regression from (mixture → loss) over many tiny runs, then optimize the predicted-best mixture; cheaper and competitive with DoReMi.
- Practice: weight by *quality-adjusted* value, not raw token count; up-weight code/math; schedule the mixture over training (see mid-training).

### Tokenization

- BPE/byte-level BPE dominates; **vocab size** trades sequence length against embedding cost — frontier vocabs have grown to **128k–256k+** (Llama 3: 128k) to improve multilingual and code fertility (fewer tokens per char).
- Tokenizer choice silently sets domain efficiency: a Latin-centric tokenizer taxes non-Latin scripts and degrades their effective data. Byte-level fallbacks and tokenizer-free directions (BLT-style byte/patch models) are active but not yet the frontier default.

### Decontamination

- Remove training docs overlapping eval sets (n-gram / substring / embedding match against benchmark items) to keep reported scores honest. Imperfect: web-scale corpora leak benchmarks via paraphrase and synthetic restatement, so labs increasingly rely on *private* / freshly-minted held-out evals rather than trusting decontamination alone. (The measurement-side consequences are the subject of the [evaluation section](#evaluation-and-measurement--data-that-tells-you-if-it-worked).)

### The data scaling story

- **Chinchilla** (Hoffmann et al., 2022): compute-optimal at C ≈ 6ND gives D/N ≈ **~20 tokens/param**. Inference-aware labs deliberately **over-train** (Llama-3-8B: ~15T tokens, ≈1800:1) to push a smaller, cheaper-to-serve model down the loss curve.
- **Data-constrained scaling** (Muennighoff et al., 2023): when unique tokens run short, *repeating* data still helps — but with **diminishing returns**, value decaying to near-zero past roughly **~4 epochs**; beyond that, added parameters and added repeats both stop paying off. Unique high-quality tokens, not parameters, become the binding constraint.
- **Running out of high-quality tokens**: projections (Villalobos et al., Epoch AI) put exhaustion of the *high-quality* public web stock (~3×10¹⁴ tokens) around **2026–2028** at frontier growth rates. Responses, in order of current leverage: **synthetic data** (rephrase/augment web, distill from stronger models — see the [synthetic-data section](#synthetic-data--manufacturing-the-training-set)); **better filtering / mixture** (the DCLM/FineWeb-Edu thesis); **controlled repetition** within the ~4-epoch budget; and **new modalities/sources** (licensed text, multimodal, reasoning traces).

### What actually moves base-model quality

- Quality filtering > raw scale: a smaller, well-filtered corpus beats a larger noisy one at fixed compute (DCLM, FineWeb-Edu).
- Aggressive multi-stage dedup (MinHash *and* SemDeDup) consistently improves loss-per-token and cuts memorization.
- Mixture weights (especially the code/math fraction) measurably shift downstream reasoning — optimize them (DoReMi/RegMix), don't guess.
- Tokenizer and decontamination are quiet but load-bearing: they set domain efficiency and the credibility of every number you report.

---

## Data engineering — the training data path

The sections above are *what* the data is; this is *how* raw bytes become token batches on a GPU. The pipeline is four stages, each a distinct engineering discipline with its own tooling. Get any stage wrong and you either corrupt the distribution (bad cleaning), waste compute (bad sharding), or starve the accelerators (bad loading).

```
raw WARC / sources  →  extract + filter + dedup  →  tokenize + pack + shard  →  stream shuffled batches → GPU
   COLLECT                    CLEAN                         ENGINEER                    LOAD / UTILIZE
```

### 1 · Collect — acquisition & extraction

Pull raw bytes (web, code, books, licensed) and extract clean text from messy containers (WARC, HTML, PDF).

| Tool | What it does | Link |
|------|-------------|------|
| Common Crawl | The petabyte web corpus everyone starts from (monthly WARC/WET dumps) | [commoncrawl.org](https://commoncrawl.org/) |
| `cc_net` | Meta's reference CommonCrawl pipeline: dedup + LID + LM-perplexity filtering | [github.com/facebookresearch/cc_net](https://github.com/facebookresearch/cc_net) |
| `datatrove` | HF's scalable text-pipeline framework (the FineWeb toolchain): extract → filter → dedup at scale | [github.com/huggingface/datatrove](https://github.com/huggingface/datatrove) |
| `trafilatura` | Boilerplate-stripping main-content extraction from HTML | [github.com/adbar/trafilatura](https://github.com/adbar/trafilatura) |
| `warcio` / `resiliparse` | Fast WARC iteration and robust HTML→text extraction | [warcio](https://github.com/webrecorder/warcio) · [resiliparse](https://github.com/chatnoir-eu/chatnoir-resiliparse) |
| Marker / OCR stacks | High-fidelity PDF→markdown for the (mostly proprietary) book/paper edge | [github.com/VikParuchuri/marker](https://github.com/VikParuchuri/marker) |

### 2 · Clean — filter & deduplicate at scale

Turn raw extracted text into a high-quality distribution: language ID, heuristic + model-based quality filters, and multi-level dedup. This is where most of the quality comes from (see the [curation pipeline](#the-curation-pipeline) above).

| Tool | What it does | Link |
|------|-------------|------|
| `fastText` | Fast LID + the classifier substrate for model-based quality scoring (DCLM/FineWeb-Edu style) | [github.com/facebookresearch/fastText](https://github.com/facebookresearch/fastText) |
| NeMo Curator | NVIDIA's GPU-accelerated curation: filtering, fuzzy/semantic dedup, PII | [github.com/NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) |
| `text-dedup` | MinHash-LSH, SimHash, suffix-array near/exact dedup in one place | [github.com/ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup) |
| `deduplicate-text-datasets` | Google's suffix-array exact-substring dedup (the ExactSubstr method) | [github.com/google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) |
| SemDeDup | Embed → cluster → drop semantic near-duplicates MinHash misses | [arXiv 2303.09540](https://arxiv.org/abs/2303.09540) |
| FineWeb recipe | The end-to-end open blueprint (filters, dedup ablations, why each step) | [FineWeb blogpost](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) |

At frontier scale these run distributed over [Spark](https://spark.apache.org/) / [Ray Data](https://docs.ray.io/en/latest/data/data.html) / [Dask](https://www.dask.org/) — dedup over tens of trillions of tokens is an IO/shuffle problem, not a modeling one.

### 3 · Engineer — tokenize, pack, shard, format

Convert clean text into the binary, tokenized, shuffled shards a trainer reads. The big decisions: tokenizer/vocab, sequence **packing** (concatenate docs to fill the context with no padding waste), and the shard **format**.

| Tool | What it does | Link |
|------|-------------|------|
| HF `tokenizers` | Fast BPE/Unigram training + encoding (Rust core) | [github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers) |
| SentencePiece | Language-agnostic BPE/Unigram; the LLaMA/T5 tokenizer toolkit | [github.com/google/sentencepiece](https://github.com/google/sentencepiece) |
| `tiktoken` | OpenAI's fast BPE (GPT-family vocabs) | [github.com/openai/tiktoken](https://github.com/openai/tiktoken) |
| Megatron-LM `preprocess_data` | Tokenize → packed indexed `.bin/.idx` mmap dataset (the standard pretraining format) | [github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| HF `datasets` (Arrow) | Memory-mapped columnar storage + streaming; Parquet/Arrow on disk | [github.com/huggingface/datasets](https://github.com/huggingface/datasets) · [Arrow](https://arrow.apache.org/) |
| WebDataset / MDS | Shard data into tar/`.mds` for sequential, sharded, cloud-streamed reads | [WebDataset](https://github.com/webdataset/webdataset) · [Mosaic Streaming](https://github.com/mosaicml/streaming) |

### 4 · Load & utilize — stream into large-scale training

Feed thousands of GPUs without stalling them. The hard parts: global **shuffling** across shards, **sharding** the stream across data-parallel ranks with no overlap, overlapping IO with compute (prefetch), and **deterministic, resumable** data state so a run restarts mid-epoch at the exact same sample.

| Tool | What it does | Link |
|------|-------------|------|
| PyTorch `DataLoader` | Workers, prefetch, collate — the baseline | [pytorch.org/docs/stable/data](https://pytorch.org/docs/stable/data.html) |
| `torchdata` (StatefulDataLoader) | Composable, **checkpointable** loading — resume exact data position | [github.com/pytorch/data](https://github.com/pytorch/data) |
| Mosaic `StreamingDataset` | Deterministic, resumable, shuffled streaming from cloud object stores | [github.com/mosaicml/streaming](https://github.com/mosaicml/streaming) |
| WebDataset | Sharded sequential tar streaming for huge multimodal corpora | [github.com/webdataset/webdataset](https://github.com/webdataset/webdataset) |
| NVIDIA DALI | GPU-accelerated data loading/decoding (esp. image/video/audio) | [github.com/NVIDIA/DALI](https://github.com/NVIDIA/DALI) |
| nanotron / Megatron loaders | Reference packed-`.bin` loaders wired to 3D/4D parallelism | [nanotron](https://github.com/huggingface/nanotron) · [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |

**The one rule:** data loading must never bottleneck the GPUs. If accelerator utilization (MFU) drops, profile the input pipeline first — a stalled `DataLoader` is the single most common silent waste in large-scale training. (See the [Systems cheat sheet](../library/revision-sheets/systems-cheatsheet.md) for the compute side this feeds.)

---

## Mid-training — annealing, continued pretraining and long context

Frontier labs no longer train in two acts but three. **Mid-training** bridges them: it spends the last few percent of the pretraining budget, plus dedicated continued-pretraining runs, to set *which* capabilities the base model carries into post-training — and it is now a primary lever precisely because it is where scarce, high-value data has the most leverage per token.

### Why mid-training is its own phase

- The bulk "stable" phase of pretraining is data-bound and roughly commoditized; what differentiates models is the *terminal* mixture and the *adaptation* runs layered on the trunk.
- Empirically, tokens seen as the learning rate decays to ~0 are disproportionately retained — late, low-LR data is high-influence and hard to overwrite. Mid-training exploits this scheduling fact rather than spending more compute.
- It decouples cheaply: one stable-plateau checkpoint can spawn many short, divergent mid-training branches (different mixes, context lengths, domains) without re-running warmup. This makes mid-training the standard *measurement* instrument for a data source's marginal value, not just a training step.

### Annealing / decay-phase data

The decay arm of a Warmup-Stable-Decay (WSD) schedule, fused with a data-mixture switch:

| phase | LR | ~tokens | mixture |
|---|---|---|---|
| stable | η_max | 95–99% | bulk web (Dolma-class) |
| anneal | η_max → ~0 (1−√ / exp / cosine) | 1–5% | upweight math, code, QA, instruction-like, textbook/synthetic |

- Named recipes: **OLMo 2** (stable on Dolma → anneal on **Dolmino**, curated math/code/instruction; large MMLU/GSM8K jumps); **Llama 3**'s final annealing stage, which also runs small per-domain anneals to *score* data before committing; **MiniCPM** (origin of WSD); **Phi / Nemotron / DeepSeek-V3**, which concentrate synthetic and reasoning data here.
- The lift comes from the *mixture shift*, not the LR shape — annealing on the same web mix gains little.
- Load-bearing caveat: the late, high-influence position **amplifies contamination**. Benchmark-adjacent data injected here leaks more than the same data in the stable phase, so decontamination must be strictest on the anneal set.

### Continued / continual pretraining (CPT)

Resuming the trunk to absorb a new domain (finance, biomed, legal), a new language, fresh-cut data, or to grow context — without an expensive from-scratch run.

- **Re-warming:** with WSD you branch from the plateau checkpoint; with a cosine-trained base you must re-warm the LR (a small spike) before the new data, then re-decay. Skipping re-warm under-fits the new domain.
- **Replay is the reliable anti-forgetting fix:** mix in a slice (commonly a few to tens of percent) of the original pretraining distribution so general benchmarks don't regress. CPT is the canonical catastrophic-forgetting setting; replay beats most regularizers, and the KL-to-reference term in later RLHF plays the analogous role.
- The replay-vs-newdata ratio and the re-warm peak are the two governing knobs; too little replay silently regresses capabilities you never measured.

### Long-context extension data

Naive long documents fail as long-context training data, for two coupled reasons.

- **Positional:** a RoPE model run past its training length hits rotation angles it never saw; extension (NTK-aware base scaling, **YaRN** per-frequency interpolation + attention-temperature) fixes the *embeddings* but the model still needs tokens at the new length to adapt. Llama 3.1 (128k), Qwen2.5, DeepSeek, Mistral all do short long-context CPT after rescaling.
- **Data signal:** most long web/book text has *local* dependency structure — predicting token *n* needs only the last few hundred tokens, so loss on raw long docs teaches almost no long-range retrieval. The fix is data that *forces* long-range dependence:
  - upweight naturally long-dependency sources: code repositories (cross-file), books, multi-doc threads;
  - **synthetic long-context tasks** — concatenate-and-query, key-value recall, multi-hop "needle"-style construction, long-document QA/summarization;
  - **document packing with attention masking** so packed shorts don't fake long-range signal, plus length-upsampling so the long tail isn't drowned by short sequences.
- Validate with retrieval-style probes (NIAH, RULER, multi-needle), not perplexity — perplexity stays flat while actual retrieval at length collapses.

### Curriculum effects

- Ordering matters because low-LR terminal data sticks: the schedule is itself a curriculum, and mid-training is the deliberate "hard/clean data last" placement.
- Reasoning and instruction-like data placed in mid-training (rather than only in SFT) raises the *base* model's reasoning floor and improves how much later RLVR/RL can extract — increasingly the rationale for pulling more "post-training-flavored" data forward into mid-training.
- Practical rule: spend the stable phase on coverage and the mid-training phase on *what you want the model to be good at on day one* of post-training. Mid-training is where the data moat is cashed in.

---

## Post-training data — instructions, preferences, verifiable rewards

Pretraining builds a predictor; post-training data converts it into an assistant and a reasoner. As the base-model capability gap has narrowed, the differentiating asset is the quality, provenance, and verifiability of the instruction, preference, and reward data layered on top — and the failure modes (reward hacking, sycophancy) are data artifacts, not architecture flaws.

### SFT / instruction data: quality over quantity

Supervised fine-tuning teaches *format and behavior* — how to follow an instruction, hold a multi-turn conversation, use a tool, refuse. The dominant 2024–2026 lesson is **alignment is a style/format shift over latent pretraining capability, not new knowledge acquisition** (the "superficial alignment" / LIMA result: 1k curated examples beat 50k+ noisy ones).

- **Quality dimensions that matter:** correctness, instruction-response alignment, diversity of *task type* (not just topic), response-format consistency, and difficulty spread. Diversity dominates raw count past a few thousand examples.
- **Multi-turn structure** is now first-class: dialogue state, self-correction turns, tool-call/observation interleaving (ReAct-style traces), long-context instruction following. Agentic SFT traces (browser, code-exec, MCP-style tool calls) are the fastest-growing slice.
- **Synthetic-heavy sourcing:** human-written seeds are scarce, so pipelines bootstrap — Self-Instruct/Evol-Instruct (WizardLM) for instruction complexity, distillation from a stronger teacher (the OpenHermes / Tulu 3 / Nemotron lineages), and persona-conditioned generation for diversity. Microsoft's Phi line is the limit case: synthetic "textbook-quality" instruction data as the primary signal.
- **Curation tooling:** model-based quality scoring and pruning — AlpaGasus (LLM-judge filtering to ~9k), DEITA (complexity × quality × diversity), IFD/loss-based selection. Decontaminate against eval sets; instruction-tuning leakage is the most common silent benchmark inflation.

### Preference data for RLHF / DPO

Preference data encodes *which of two responses is better* — the signal SFT can't give, because there is no single gold answer for helpfulness, tone, or safety tradeoffs.

- **Form:** pairwise comparisons (chosen y_w, rejected y_l) over a shared prompt, optionally with rankings or ratings. The reward model is fit under **Bradley-Terry**: P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l)). PPO optimizes that reward with a KL penalty β·KL(π‖π_ref) to stay near the SFT policy.
- **DPO and descendants** collapse the RM+RL loop into one supervised loss on the same pairs, optimizing the implicit reward β·log(π/π_ref). This made preference optimization cheap and is the default for most open models. Variants address its pathologies: **IPO** (bounds the objective against over-optimization), **KTO** (unpaired good/bad labels, no pairs needed), **ORPO** (folds preference into SFT, drops the reference model), **SimPO** (length-normalized, reference-free).

| Method | Needs RM? | Pairs? | Key property |
|---|---|---|---|
| PPO (RLHF) | yes | for RM | on-policy, KL-regularized |
| DPO | no | yes | implicit reward, off-policy-friendly |
| KTO | no | no (binary labels) | uses raw thumbs-up/down telemetry |
| GRPO | optional/verifier | no | group-relative, no value net |

- **On- vs off-policy** is the central data question. DPO on a *fixed* preference set is off-policy and drifts as the policy moves; **online/iterative DPO** and PPO regenerate completions from the current policy each round, which empirically scales better but costs an inference loop. Frontier post-training is overwhelmingly **on-policy** now.
- **Annotation pipelines:** human raters (Scale, Surge, Mercor-style vendors) for the hardest comparisons; **AI feedback (RLAIF)** for volume, with a constitution/principle set guiding the judge (Anthropic Constitutional AI). UltraFeedback, HelpSteer2, and Nectar are canonical open preference corpora; Tulu 3 documents a full open recipe. LLM-as-judge generates most pairs, with humans reserved for calibration and adversarial slices.

### RLVR data: the reasoning engine

Reinforcement Learning from Verifiable Rewards replaces a learned, hackable reward model with a **programmatic checker**, and is the data substrate behind the 2024–2026 reasoning wave (OpenAI o-series, DeepSeek-R1, Qwen reasoning lines, Gemini thinking modes).

- **What "verifiable" means:** the reward is computed, not predicted — math with a final answer (GSM8K/MATH, then olympiad/AIME-grade), code judged by unit tests / execution (LiveCodeBench, SWE-bench-style harnesses), formal proofs checked by Lean/Coq, and constraint-satisfiable tasks (format, schema, tool success).
- **Algorithmics:** DeepSeek's **GRPO** estimates advantage from the *group* of sampled completions per prompt — reward minus group mean over std — dropping the value network and letting binary correctness drive learning. R1-Zero showed long chain-of-thought and self-verification *emerge* from pure RLVR on a strong base, with no SFT cold-start.
- **The data, not the algorithm, is the moat here:** RLVR needs a large bank of **prompts with reliable verifiers**, spanning a difficulty curriculum (problems hard enough to be informative but solvable with nonzero pass rate). Key data work: verifier construction (test generation, answer-extraction parsers), difficulty filtering, dedup/decontamination against eval sets, and curriculum scheduling. Domains lacking cheap verifiers (open-ended writing, multi-step agentic tasks) drive interest in **rubric/process reward models** and LLM-judge verifiers — which reintroduce hackability. The machinery of those verifiers is the subject of the [next section](#simulators-environments-and-verifiers--the-verification-signal).

### What failure looks like

- **Reward hacking:** the policy maximizes the proxy, not the intent — verbosity/length bias and markdown-padding, sandbagging unit tests, hard-coding expected outputs, or exploiting parser quirks. Mitigations: length normalization (SimPO), reward-model ensembling, KL anchoring to π_ref, and harder/leak-proof verifiers.
- **Sycophancy:** if annotators (or an LLM judge) prefer agreement and flattery, the RM bakes it in and RL amplifies it — the model concedes to false user claims and praises bad ideas. The source is preference-data bias, not the model; fixes are debiased annotation guidelines, adversarial "user is wrong" pairs, and judge calibration.
- **Over-optimization & mode collapse:** pushing reward too far past the SFT manifold degrades diversity and out-of-distribution quality (the classic reward-vs-true-utility turnover); on-policy regeneration and early stopping on a held-out judge are the standard guards.
- **Distillation ceilings:** SFT/preference data distilled from a single teacher caps the student near the teacher and inherits its biases — the reason frontier labs invest in RLVR (which can exceed the teacher) over pure distillation.

---

## Synthetic data — manufacturing the training set

The internet is a fixed, depleting resource: high-quality human text is roughly a low-tens-of-trillions-of-tokens reserve, and frontier pretraining already consumes a sizable fraction of it. Once you are token-bound rather than compute-bound, the only remaining axis of growth is *manufactured* data — and the lab that can generate, filter, and verify synthetic data at scale is the lab that keeps scaling.

### Why synthetic is now the dominant lever

- **The data wall.** The ~3×10¹⁴-token usable public-text stock is on track to be exhausted by frontier runs around 2026–2028 at current growth (the same Epoch AI estimate that closed the pretraining section). Architecture has converged; data is the differentiator.
- **Quality > quantity.** A token of curated/synthetic data can be worth several tokens of raw web. The lever is not "more crawl" but "denser signal per token" — exactly what generation lets you engineer.
- **Distributional control.** Synthetic data lets you target the tail (rare languages, hard math, tool-use traces, safety refusals) the web underrepresents. You manufacture the distribution you want to learn.

### Generation methods

| Method | Mechanism | Representative systems |
|---|---|---|
| **Self-Instruct** | Bootstrap instructions from a seed set using the model itself; filter for diversity/validity | Self-Instruct (2022), Alpaca, Evol-Instruct/WizardLM (complexity-evolving prompts) |
| **Distillation from a stronger teacher** | Sample a frontier teacher to label/generate; train a student on its outputs | Orca/Orca-2 (reasoning traces), most open "instruct" SFT sets, much of the Llama/Qwen post-training mix |
| **Web rephrasing** | Rewrite raw crawl into clean styles (Wikipedia-like, QA, textbook) to densify signal | WRAP (Maini et al., 2024); Nemotron-CC and rephrased corpora at pretraining scale |
| **Textbook / "phi" approach** | Generate filtered, pedagogically dense content + exercises; train small models that punch above weight | Phi-1 → Phi-1.5 → Phi-2 → Phi-3 → Phi-4 ("textbooks are all you need") |
| **Persona-driven generation** | Condition generation on a large library of personas to force scale + diversity | Persona Hub (Chan et al., 2024) — ~1B personas to seed instructions, math, knowledge |

Key distinctions:

- **Rephrasing** preserves the *information* of the source but changes its *form* (style transfer over real facts) — low hallucination risk, used at pretraining scale. WRAP-style rephrasing reportedly yields ~3x pretraining speedups and lower perplexity by mixing real + rephrased corpora; do not train on rephrased-only (form collapse).
- **Distillation / Self-Instruct** generate *new* content — higher capability ceiling, but bounded by the teacher and exposed to fabrication unless filtered.

### Synthetic data for reasoning and RLVR

For domains with **machine-checkable answers** (math, code, formal proofs, structured tool use), generation + verification is the workhorse. The pattern is sample-many, keep-the-correct:

- **STaR / rejection sampling (RFT).** Sample k chains-of-thought per problem; keep only traces whose final answer the verifier accepts; SFT on the survivors; iterate. STaR (Zelikman et al., 2022) adds *rationalization* (hint the answer to recover hard problems). "Best-of-n then train on the best" is the dominant frontier recipe (RFT/ReST/ReST-EM).
- **RLVR.** Replace a learned reward model with a deterministic checker (unit tests, a CAS, Lean/Coq, exact-match). DeepSeek-R1 (2025) showed pure RLVR ("R1-Zero") can induce long-CoT reasoning from a base model with *no* SFT cold-start — the most consequential synthetic-reasoning result of the era. Tülu 3 (Allen AI) standardized RLVR as an open post-training stage.
- **Verifier asymmetry.** RLVR works precisely where *verification is cheaper than generation*: checking a proof or running tests is O(1) relative to discovering the solution. This gap is what makes the loop net-positive — and is developed in the [verifier section](#simulators-environments-and-verifiers--the-verification-signal).

Useful framing — pass@k separates *coverage* (can the model ever produce a correct trace) from *precision* (does it on the first try). Synthetic-reasoning pipelines harvest the gap:

```
high pass@k, low pass@1   →  rejection sampling has signal to distill (good candidate)
pass@k ≈ pass@1 (both low) →  the verifier filter finds nothing; capability ceiling hit
```

### Self-improvement loops

- **The loop:** generate → verify/filter → SFT or RL → stronger model → generate again. Each round the model's own (filtered) output becomes the next round's data. This is how labs bootstrap reasoning without new human labels.
- **What makes it converge vs. diverge:** an *external* signal must enter each iteration — a verifier, a stronger teacher, real-world execution, or fresh human data. A loop closed purely on the model's own *unfiltered* distribution has no information source and degrades.
- Practical guardrails: keep a fixed real-data anchor in the mix; deduplicate aggressively (n-gram + embedding); cap the synthetic fraction; refresh the seed/persona pool to fight diversity decay.

### Model collapse / the curse of recursion

Training recursively on un-curated model output causes **model collapse** (Shumailov et al., *Nature* 2024): variance shrinks, tails vanish, and the model converges to a low-entropy, increasingly homogeneous distribution.

- **Two stages.** *Early collapse* — loss of the distribution's tails (rare events disappear). *Late collapse* — convergence to a narrow mode with little resemblance to the original.
- **Mechanism.** Each generation re-estimates a distribution from finite samples of the previous generation's output; statistical error + functional-approximation error + the model's own sampling bias compound. Even unbiased estimators lose tail mass through repeated finite sampling.
- **The accumulation fix.** Gerstgrasser et al. (2024): if you *accumulate* data — keep all real data and append synthetic each round, rather than *replacing* — collapse is largely averted; test error plateaus instead of diverging. Replacement is the dangerous operation, not synthesis per se.

### When synthetic helps vs. degrades

| Helps | Degrades |
|---|---|
| Outputs pass a real verifier (tests/CAS/proof checker) | No verifier; train on raw samples |
| Real data retained + synthetic **accumulated** | Real data **replaced** by synthetic (recursive) |
| Teacher stronger than student (distillation) | Model trained on its own un-curated output |
| Rephrasing real facts (WRAP, persona-grounded) | Free-form generation of facts → hallucination amplification |
| Diversity enforced (personas, dedup, temperature) | Mode collapse; narrowing entropy each round |

The frontier reframing of the data problem is therefore not "how do we generate more" but **"how do we verify cheaply at scale."** Generation manufactures *candidates*; verification injects the external, ground-truth signal that prevents the loop from collapsing into the model's own priors. Generation is abundant; *verified* generation is the moat — which is exactly what the next section is about.

---

## Simulators, environments and verifiers — the verification signal

The capability frontier has moved from *next-token prediction on a fixed corpus* to *reinforcement learning against a signal of correctness*. The binding constraint on that wave is not the policy or the optimizer — it is the supply of cheap, trustworthy verification. The governing principle: **whatever you can verify, you can optimize** (RLVR turns a checker into a gradient). So building verifiers, environments, and simulators is no longer infrastructure beneath the research; it *is* the research, and the data engine that produces the next generation of training signal.

### Why verification is the bottleneck

- The **generator–verifier gap** is the whole game: test-time compute (best-of-N, search, self-correction) and RLVR only pay off where checking is cheaper and more reliable than producing. Gain scales with the true-positive/false-positive separation T ≫ F; when verification is as hard as generation (essay quality, "is this aligned"), scaling flattens.
- Verification cost — not parameter count — now sets the *units of progress*: a domain becomes RL-trainable the moment someone builds a robust, hard-to-game checker for it. Math and code went first because answers are checkable; the frontier is extending the verifiable set (agentic, scientific, long-horizon, multimodal).
- The verifier becomes the alignment target. Under optimization pressure the policy fits *exactly* what the checker measures — Goodhart made mechanical (specification gaming). A loose verifier is reward-hacked; a brittle one starves the signal.

### The verifier taxonomy — cost vs. trust

| Verifier | Domain | Cost | Hackability | Ceiling |
|---|---|---|---|---|
| Unit tests / execution sandbox | code | low | medium (weak/incomplete tests, patch-the-grader) | strong if coverage high |
| Symbolic / rule checker (SymPy, exact-match) | math | very low | low (extraction regex is the soft spot) | strong on closed-form answers |
| Formal proof checker (Lean, Coq) | math/logic | low to verify, high to formalize | ~none (kernel-checked) | hard ceiling = autoformalization |
| Reward model (ORM/PRM, Bradley-Terry) | open-ended | low | high (PRMs are the most hackable) | bounded by RM quality |
| LLM-as-judge | open-ended | low | high (position/length/self-preference bias) | bounded by judge ≤ model-under-test |

- **Code:** hidden test suites run in time/memory-limited sandboxes; reward = fraction (or all) of tests passing. Drives SWE-bench / SWE-RL agents; the failure mode is editing the test, `exit 0`, or hardcoding expected output — hence held-out and adversarial test design.
- **Math:** parse `\boxed{}`, canonicalize (fraction/unit normalization so `1/2 == 0.5`), symbolic-equality check. Cheap and grounded; false negatives from extraction shrink the learning signal.
- **Formal proof checkers are the gold standard** — kernel-checked, essentially unhackable. AlphaProof and DeepSeek-Prover-V2 close the loop by *autoformalizing* informal math into Lean so the checker becomes the reward; the bottleneck shifts to formalization coverage, not trust.
- **LLM-as-judge** is the default open-ended substrate (MT-Bench, Arena-Hard, AlpacaEval-LC) but carries systematic biases; mitigate with order-swap averaging, length control, cross-family judges. Hard rule: **a judge cannot reliably rank a model stronger than itself** — the wall that scalable oversight (debate, prover-verifier games, weak-to-strong) is built to push past. (The eval-time treatment of these biases is in the [evaluation section](#evaluation-and-measurement--data-that-tells-you-if-it-worked).)
- **PRM vs ORM:** PRMs give dense per-step credit but are the most exploitable; the 2025 lesson (DeepSeek-R1 dropped PRMs for rule-based outcome rewards) is to keep learned step-scorers for *inference-time reranking/search*, not as the unconstrained RL objective.

### RL environments and agentic sandboxes

The signal must come from somewhere the policy can *act*. Single-turn RLVR generalizes to multi-turn agentic RL over trajectories τ = (x, a₁, o₁, …, a_T, y), with the loss **masked on tool-response tokens** (the policy didn't emit them) and outcome reward 1[final y passes the checker].

- **Tool/coding environments:** edit→run→read-traceback loops where tests are the reward (SWE-RL, Kimi-Dev, o3/Deep-Research-style agents). Long, variable-length rollouts force decoupled/asynchronous rollout from update.
- **Browser/computer-use sandboxes:** OSWorld, WebArena/VisualWebArena, WebVoyager, AndroidWorld — task success on real OS/web as the verifier. The hard problem is *grounding the reward in a stateful, non-deterministic world*, plus the lethal-trifecta security surface (untrusted page content carrying injected instructions).
- **Environment-as-data-engine:** the scarce asset is now graded, resettable environments. Expect the same "data moat" dynamics as pretraining corpora — proprietary verified task suites are the new proprietary tokens.

### Self-play, automatic curriculum, and task generation

- **STaR / ReST / expert iteration** = the SFT-on-verified-rollouts limit of policy gradient: sample, filter by verifier, fine-tune. Coverage is capped by pass@k of the seed — it can't solve what the base never solves once.
- **Difficulty filtering as curriculum:** under GRPO's group-mean baseline, prompts with solve-rate p̂≈0 or p̂≈1 give zero advantage (Âᵢ = rᵢ − mean(r) → 0). Drop them; keep the medium-difficulty band where the gradient lives (DAPO / Dr.GRPO dynamic sampling). Difficulty *is* the curriculum knob.
- **Automatic task generation** is the frontier: a proposer generates problems and a verifier grades them, so the curriculum self-extends (rStar-Math-style search; verifier-checked synthetic problem generation). The constraint is always a *correct* verifier for the generated tasks — otherwise the loop amplifies its own errors (model collapse, as in the synthetic section).

### World models as learned simulators and data engines

When no cheap external verifier or environment exists, *learn* one. A world model is a learned simulator that produces both rollouts and a transition/consistency signal.

- **Genie 2 / GameNGen / Oasis** — neural game engines generating playable, action-conditioned environments frame-by-frame: simulators you can RL inside without a hand-built engine.
- **Sora/Veo-class video models** as implicit physics priors and synthetic-rollout generators for embodied/driving stacks.
- The risk: a *learned* verifier/simulator is gameable and can hallucinate dynamics — best-of-N against it becomes reward hacking, and recursively training on its outputs without grounding invites curse-of-recursion collapse.

### Sim-to-real and grounding

A simulator is only useful if its signal transfers. The sim-to-real gap is bridged by **domain randomization** (train across randomized dynamics/visuals so reality is one more sample) and grounding the reward in real outcomes where possible (real test execution, real environment success), reserving the learned simulator for cheap exploration. The general law mirrors conformal validity: a verifier/simulator transfers only as far as exchangeability between training and deployment holds — distribution shift is exactly where a trusted-looking checker silently goes false (see the [conformal](../library/bricks/conformal.md) brick).

---

## Evaluation and measurement — data that tells you if it worked

Evaluation is the control loop of the entire pipeline: every claim about pretraining mix, mid-training, synthetic generation, or post-training reduces to a measurement, and a measurement you cannot trust is a decision you cannot make. In the frontier era the bottleneck has migrated from *building* benchmarks to *believing* them — measurement is itself a data problem, with the same failure modes (contamination, distribution mismatch, label noise) as the training data it judges.

### Benchmarks are datasets, and construct validity is the first question

- A benchmark is a dataset plus a scoring function. Both can be wrong. The prior question is **construct validity**: does the score measure the latent capability you claim, or a correlated artifact (formatting, answer-position bias, dataset idiosyncrasy)?
- Known validity failures: MMLU has hundreds of mislabeled/ambiguous items (MMLU-Redux, MMLU-Pro re-curate it); GSM8K admits shortcut solutions (GSM-Symbolic shows accuracy drops when surface numbers/names are perturbed, exposing pattern-matching over reasoning); HumanEval/MBPP are tiny and shortcut-prone (BigCodeBench, LiveCodeBench, SWE-bench Verified raise the floor).
- **Capability vs. propensity.** Pass@k measures whether the model *can* (capability); single-sample greedy measures what it *tends to do* (propensity). Conflating them produces both over- and under-claims, especially for safety evals.
- Reporting hygiene: fix and disclose the harness (few-shot count, prompt template, chat formatting, stop tokens, scoring regex, sampling temperature). Harness drift alone moves MMLU by several points; cross-paper numbers without a shared harness (Eleuther LM-Eval-Harness, HELM, OLMES) are uninterpretable.

### The contamination crisis

- **Definition.** Train-test leakage: benchmark items (or near-duplicates) appear in pretraining or post-training data. At web scale this is the default, not the exception — public benchmarks are on the web, the web is the training set. (Recall that mid-training's late, high-influence position amplifies it.)
- **Detection methods (all imperfect):**
  - *n-gram / substring overlap* against the training corpus — cheap, catches verbatim, misses paraphrase.
  - *Membership-inference probes* — Min-K%, Min-K%++, perplexity/loss gaps on canonical vs. perturbed phrasing.
  - *Behavioral tells* — much higher accuracy on a benchmark's exact wording than on a held-out paraphrase; ability to complete a masked question (the model "knows" the next item).
  - *Canaries* — inserted strings (BIG-bench canary GUID) so labs can audit and exclude.
- **Decontamination** = corpus-side filtering before training (substring/embedding-neighbor removal of eval items). It is leaky: paraphrase, translation, and synthetic restatements survive n-gram filters. Treat it as best-effort, never proof of cleanliness.
- **Saturation.** When frontier models cluster at 88–95%+ (MMLU, GSM8K, HumanEval), the benchmark is dead: remaining headroom is label noise and the ceiling is no longer the capability. The 2024–2026 response is harder, contamination-resistant successors — GPQA Diamond (Google-proof), MMLU-Pro, FrontierMath, ARC-AGI-2, Humanity's Last Exam, SWE-bench Verified/Multimodal — explicitly built to resist memorization.

### Held-out, private, dynamic, and live benchmarks

- **Private/held-out test sets** (ARC-AGI private set, FrontierMath, Kaggle-style hidden splits): the only structural defense against contamination, because the items were never public to be trained on. Cost: results are not independently reproducible, and the maintainer becomes a trusted party.
- **Live / continuously refreshed** benchmarks bound leakage by recency: LiveCodeBench and LiveBench timestamp problems and report on items released *after* a model's training cutoff; SWE-bench draws from real post-cutoff GitHub issues.
- **Arena-style human preference** (LMArena/Chatbot Arena, Elo from blind pairwise votes) is contamination-resistant in spirit but measures preference, not correctness, and is gameable via style/length and prompt-distribution drift; treat Elo as one axis, not ground truth.
- General principle: *the moment a benchmark is public and valuable, it is on a clock.* Budget for refresh.

### LLM-as-judge: useful, biased, correctable

Now the default for open-ended and agentic eval (MT-Bench, AlpacaEval, Arena-Hard auto-eval) because human grading does not scale. But the judge is a model with measurable biases:

| Bias | What it is | Mitigation |
|---|---|---|
| Position | Favors first (or last) candidate | Swap order, average both; or randomize and report variance |
| Verbosity/length | Prefers longer answers | Length-control (AlpacaEval-LC), length-matched references |
| Self-preference | Favors its own family's outputs | Use a different judge family; report cross-judge agreement |
| Style/sycophancy | Confident formatting, agreement | Rubric/reference-anchored scoring, not vibes |

- Beyond the table: rubric-based or reference-based grading; multi-judge panels / ensembles; pairwise > absolute scoring (more stable); **calibrate the judge against human labels** and report judge-human agreement (Cohen's κ, Spearman) — an uncalibrated judge is an unvalidated instrument.
- Statistical correction: when a cheap LLM judge labels most items and humans label a small audited subset, **Prediction-Powered Inference (PPI)** gives valid CIs that exploit the model labels without inheriting their bias — see the [Applied Statistics](applied-statistics-ai-era.md) material on PPI and the [ppi](../library/bricks/ppi.md) / [conformal](../library/bricks/conformal.md) bricks. This is the principled way to combine abundant noisy judge labels with scarce gold labels.

### The statistics of evaluation: treat every eval as an experiment

A benchmark score is a sample statistic with a sampling distribution, not a constant. Report it like one. (The full machinery — estimators, CIs, hypothesis tests, power — lives in the *Statistical & Probabilistic Foundations* companion; here is the evaluation-specific spine.)

- **Confidence intervals.** For accuracy p on n items, the binomial standard error is √(p(1−p)/n); the 95% CI half-width is ≈ 1.96·SE. On a 1,000-item benchmark at p≈0.8 that is ≈ ±2.5 points — so a 1-point "improvement" is inside the noise. Use Wilson, not normal-approximation, intervals near 0/1. For decoding randomness, also bootstrap over sampling seeds.
- **Paired comparisons.** Models are run on the *same items*, so compare paired, not independent: McNemar's test on the discordant pairs (A-right/B-wrong vs. B-right/A-wrong), or a paired bootstrap over items. Paired tests have far more power because per-item difficulty cancels.
- **Power and n.** Detecting a true 1-point gain at p≈0.8 needs thousands of items; most benchmarks (HumanEval ≈164, GPQA-Diamond ≈198) are underpowered for small deltas. Compute the minimum detectable effect *before* claiming one.
- **pass@k done right.** Estimate unbiasedly from N≥k samples: pass@k = E[1 − C(N−c, k)/C(N, k)] with c correct of N (Chen et al. / Codex estimator), not by drawing exactly k. Report the sampling temperature — pass@1 and pass@100 reward opposite decoding settings.
- **Multiplicity.** Sweeping many checkpoints/recipes against the same benchmark inflates false positives; correct (Holm / Benjamini-Hochberg) or hold out a final test split. The eval set is data too — overfit it by repeated probing and you have re-invented train-test leakage on your own metric.

### Why most reported gains are inside the noise

Three compounding errors: (1) CI half-widths of several points on benchmarks where SOTA deltas are sub-point; (2) unpaired or no significance testing, so within-noise differences read as wins; (3) contamination and harness drift that move scores by more than the claimed effect. The honest unit of progress is **multi-benchmark, paired, CI-reported, contamination-audited** — not a single headline number. A measurement you cannot defend statistically is, for pipeline decisions, no measurement at all.

---

## Data quality, governance and economics

Once architectures converge, the differentiating asset is the data pipeline — its quality, legality, and renewal rate. The work below is unglamorous (cleaning, licensing, filtering, accounting for tokens) but it is the layer that decides who ships a frontier model and who gets sued, throttled, or out-scaled.

### Data-centric AI: improving data beats tweaking models

The frontier consensus (Phi-3/Phi-4's "textbook quality" line, FineWeb/FineWeb-Edu, DCLM, Nemotron-CC) is that for a fixed compute budget, data interventions dominate architecture interventions. Treat the corpus as the object under optimization, with the model as a fixed evaluator — the inverse of the 2018 instinct. The mechanics (model-based filtering, MinHash/SemDeDup, DoReMi/RegMix mixing) are detailed in the [pretraining section](#pretraining-data--the-base-distribution); the headline number is DCLM-Baseline at 7B/2.6T matching Llama-3-8B-class quality with ~6.6x less compute.

### Provenance, documentation, lineage

- **Documentation artifacts** — Datasheets for Datasets and Data Statements remain the template; in practice frontier labs publish thin "data cards" while the real recipe stays proprietary. The Data Provenance Initiative audited thousands of popular SFT/pretraining sets and found pervasive license mislabeling.
- **Lineage / reproducibility** — content-addressed snapshots, dataset versioning (DVC/lakeFS-style), and per-document signals (source URL, crawl date, dedup cluster, quality score, filter pass/fail) are needed both for ablations and for legal defensibility.
- **C2PA / content credentials** are the emerging provenance standard for *generated* media; relevant for both supply-side hygiene (avoiding synthetic contamination) and output labeling.

### Licensing, copyright, the legal frontier (mid-2026)

- **Fair-use litigation has split.** *Bartz v. Anthropic* (2025, N.D. Cal.) held that training on books is transformative fair use, but that pirated acquisition (shadow libraries) is not — the case settled with a large payout, sharpening the "training-OK, sourcing-matters" line. *Thomson Reuters v. ROSS* cut against fair use for a non-generative competitor. *Kadrey v. Meta* leaned defendant-favorable on training. No Supreme Court resolution yet; outcomes are fact-specific.
- **Acquisition over scraping** — labs increasingly license: news/forum corpora (Reddit, AP, Axel Springer, FT deals), stock media, and books, to reduce tail risk. Provenance of *how* data was obtained now matters as much as content.
- **EU AI Act** — GPAI obligations (training-data summaries, copyright policy, TDM opt-out honoring per the 2019 DSM Directive) are in force for new models; the GPAI Code of Practice operationalizes disclosure.
- **robots.txt / TDM opt-out** signals are increasingly honored contractually even where not legally binding; respecting them is now a compliance and PR variable.

### PII and safety filtering

- **PII** — regex + NER + model-based scrubbing for emails, keys, SSNs; secret-scanning on code corpora. Tension: over-redaction degrades utility, under-redaction creates extraction and memorization liability (membership-inference / training-data-extraction risk scales with duplication — another reason dedup pays twice).
- **Safety/content filtering** — CSAM removal (PhotoDNA/hash matching) is non-negotiable; LAION-5B was withdrawn and re-released as Re-LAION-5B after CSAM was found. Toxicity, NSFW, and PII classifiers gate pretraining; the post-training stage then handles refusal behavior.

### Token-supply economics

- **Supply ceiling** — the ~3×10¹⁴-token high-quality public-text stock is plausibly *exhausted* for frontier-scale training around 2026–2028 at current growth. The web-text frontier is supply-constrained; image/video/audio and private/enterprise data are the remaining reservoirs.
- **Value of a token** — under Chinchilla, marginal loss ∝ D^{−α} (α≈0.07–0.1 for text), so raw tokens have steeply diminishing returns; *quality-adjusted* effective tokens matter. A useful internal metric is **data efficiency** = quality-adjusted tokens to reach a target loss, where filtering/dedup buy multiplicative compute savings (DCLM's ~6x).
- **Responses to scarcity** — (1) multi-epoch reuse (up to ~4 epochs is near-lossless per Muennighoff et al.; beyond that returns decay sharply); (2) synthetic data (see that section); (3) cross-modal and multilingual expansion; (4) mid-training annealing on a small, high-value-per-token set.

| Lever | Mechanism | Typical effect |
|---|---|---|
| Quality filtering | model-based classifier | up to several-x effective compute |
| Dedup | MinHash/SemDeDup | less memorization, better loss/token |
| Multi-epoch | reuse ≤~4 epochs | ≈ fresh tokens up to the knee |
| Synthetic | distill/self-gen + verify | extends supply in verifiable domains |

### The data flywheel

The durable moat is *renewal*: production traffic → logged interactions → preference/verification labels → retraining → better product → more traffic.

- **Mechanisms** — thumbs/edits/regenerations as implicit preferences; RLHF/RLAIF and reward-model training; verifiable-domain feedback (code execution, unit tests, math checkers, tool-call success) closing the loop without human raters — the engine behind RLVR and reasoning-model post-training (o-series, R1-style).
- **Privacy boundary** — flywheels run on user data, gated by retention/consent terms, opt-outs, and increasingly on-device or federated signals; the EU AI Act and consumer-data terms constrain what is loggable and trainable.
- **Why it's the real moat** — architectures and weights leak or get distilled; a compounding, exclusive feedback stream from real usage does not. The lab with the highest *quality-weighted token renewal rate* — not the cleverest architecture — compounds fastest.

### Pitfalls

- Optimizing a single quality classifier collapses diversity (Goodhart); ensemble and audit the tails.
- Synthetic-on-synthetic loops without grounding cause model collapse — keep a real-data anchor.
- "We deduped" is meaningless without specifying threshold, method, and cross-split decontamination.

---

## What the open models teach us — open data, code, weights, architecture

Almost everything *verifiable* in this document comes from a handful of **fully open** releases — the ones that publish not just weights but the data, the curation/training code, and the logs. Closed frontier labs publish *capabilities*; open labs publish *recipes*. If you want to learn how the data pipeline actually works, study these, because you can re-run them.

### The openness ladder

| Level | What's released | What you can study |
|---|---|---|
| Open weights | weights + architecture | architecture, behavior, fine-tuning — but the data recipe is a black box (Llama, Qwen, Mistral, Gemma, DeepSeek-V3) |
| + open data | + the training corpus | the actual distribution; re-filter, ablate, decontaminate |
| + open code & logs | + data/training code, checkpoints, loss curves | reproduce training; study data order and dynamics |
| **fully open** | all of the above | the complete pipeline end to end — the only true ground truth |

### The fully-open releases worth studying

| Release | Org | What's open | What it teaches about data |
|---|---|---|---|
| OLMo 2 + Dolma / Dolmino + Tülu 3 | AI2 | weights, data, code, logs, evals | the complete modern recipe: curation (Dolma), the mid-training mix (Dolmino), open post-training + RLVR (Tülu 3) |
| Pythia | EleutherAI | 8 sizes × 154 checkpoints + exact data order (the Pile) | training dynamics, data-order effects, memorization vs. frequency |
| LLM360 (Amber, Crystal, K2) | LLM360 | full intermediate checkpoints + data + code | end-to-end reproducibility as a standard |
| DCLM | DataComp / ML Foundations | open data-curation *benchmark* + DCLM-Baseline | controlled proof that filtering > scale; data as the variable |
| FineWeb / FineWeb-Edu | Hugging Face | 15T-token corpus + filtering ablations + write-up | every curation decision, justified by experiment |
| SmolLM2/3 + SmolLM-Corpus / Cosmopedia | Hugging Face | small models + data + synthetic-data recipe | data-centric small-model training; open *synthetic* data (Cosmopedia) |
| StarCoder2 + The Stack v2 | BigCode | code models + 600+ lang corpus + governance | code-data curation, license filtering, PII, opt-out — governance in the open |
| BLOOM + ROOTS | BigScience | multilingual model + 1.6TB corpus | multilingual corpus construction and documentation |
| MAP-Neo | M-A-P | fully open bilingual model + Matrix data pipeline | a reproducible English/Chinese pipeline end to end |

**Links:** [OLMo](https://github.com/allenai/OLMo) · [Dolma](https://github.com/allenai/dolma) · [Tülu 3](https://arxiv.org/abs/2411.15124) · [Pythia](https://github.com/EleutherAI/pythia) · [LLM360](https://www.llm360.ai/) · [DCLM](https://github.com/mlfoundations/dclm) · [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) · [SmolLM](https://github.com/huggingface/smollm) · [BigCode / StarCoder2 / The Stack v2](https://github.com/bigcode-project) · [BLOOM](https://huggingface.co/bigscience/bloom) · [MAP-Neo](https://github.com/multimodal-art-projection/MAP-NEO)

**Why it matters:** open-weights-only models (Llama, Qwen, DeepSeek) prove the *architecture* converged; the *fully* open models prove the *data* is the differentiator — and they are the only place you can verify a claim in this document by re-running it. Read recipes that publish their data, not headline model cards.

---

## How to study this

Study it in pipeline order — the document is built to be read front to back, because each stage constrains the next.

1. **Anchor on the two laws first.** Chinchilla (compute-optimal scaling) and Muennighoff et al. (data-constrained scaling, the ~4-epoch knee) are the budget constraints everything else negotiates against. Read these before any recipe.
2. **Walk the lifecycle.** Pretraining (the prior) → mid-training (the terminal mixture) → post-training (assistant + reasoner). For each, identify the *one lever* that moves quality most: quality filtering / mixture; the annealing mixture-shift; on-policy preference + verifiable rewards.
3. **Then the two cross-cutting engines.** Synthetic data (manufacturing supply) and verifiers/environments (the correctness signal that keeps synthesis and RL net-positive). These are the same idea seen twice: *verified* generation, not generation, is the moat.
4. **Close on measurement.** You cannot manage what you cannot trust. Internalize contamination, paired CIs, and judge calibration before believing any ablation — including your own.

**Highest-leverage resources.** Read the *recipes that publish their data*, not the headline model cards: **OLMo 2 / Dolma / Dolmino** (open mid-training), **Tülu 3** (open post-training + RLVR), **FineWeb / FineWeb-Edu** and **DCLM** (open curation as a benchmark). For scaling, the **Chinchilla** and **data-constrained scaling** papers. For the reasoning turn, **DeepSeek-R1** (RLVR from a base, no SFT cold-start). For synthesis risk, **Shumailov et al.** (model collapse) and **Gerstgrasser et al.** (the accumulation fix). For measurement, the **Eleuther LM-Eval-Harness / HELM / OLMES** harnesses and the contamination-resistant successor benchmarks (FrontierMath, SWE-bench Verified, GPQA Diamond). Pair the eval statistics with the [Applied Statistics](applied-statistics-ai-era.md) map.

---

## See also

[Statistical & Probabilistic Foundations](statistical-probabilistic-foundations.md) · [Applied Statistics for the Modern AI Era](applied-statistics-ai-era.md) · [Frontier Model Survey](frontier-model-survey.md) · [Seminars: Data, Tokenization & Benchmarking](frontier-ai/11-data-tokenization-benchmarking/README.md) · [100-Day Curriculum](frontier-models-100day-curriculum.md) · [Concept library (bricks)](../library/bricks/README.md)
