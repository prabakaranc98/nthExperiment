# Capstone: Causal Foundation Model (cFM) — an in-context causal inference model

**Capstone ID:** R1 (research / exploratory) · cross-listed FAIRE 501 (probabilistic & causal foundations) + 502 (generative/foundation models)
**FAIRE Subject:** 501 / 502 (research elective)
**Status:** not started
**Report:** report.md
**Repo:** src/

---

## What this capstone proves

Building, from scratch, a small foundation model that does *amortized in-context causal inference*: a 300–500M-parameter transformer pretrained on synthetic Structural Causal Models that answers interventional, counterfactual, and effect-estimation queries in a single forward pass — with no per-dataset retraining. Completing it demonstrates working mastery of Prior-Fitted Networks, SCMs, transformer architecture, synthetic-prior design, and rigorous causal evaluation (calibration, coverage, identifiability-aware uncertainty). The defensible novelty is the *jump to mid-scale*: one unified causal FM whose single query head spans interventional / counterfactual / effect estimation, with conformal, identifiability-aware uncertainty — a regime no published causal PFN currently occupies.

---

## The idea & the three directions

The owner floated three framings for the capstone. All three are "a transformer that does causal inference in one forward pass," but they differ enormously in what data they need, what fits on a few GPUs, and what is genuinely open in mid-2026. This section scores each, then commits to a spine and a stretch, and shows the path that connects them.

**Recommendation up front:** the spine is a PFN-style *causal FM for any distribution* (framing C below); the killer stretch application is *virtual-cell perturbation prediction* (framing B). Build C to a real, calibrated artifact first; treat B as the capstone's killer demo and A (causal dynamics) as a paper-2 generalization.

### The three framings, scored

| Framing | What it amortizes | Data needed | 300-500M solo? | Novelty (mid-2026) | Anchor work |
|---|---|---|---|---|---|
| **A. Causal dynamics FM** | World/transition dynamics under `do()` in temporal/RL settings | Trajectory rollouts from simulators or offline RL logs; must *include interventions* | Risky — needs an env/sim harness + RL infra on top of the model | High, but crowded and ill-defined eval | Causal world models; Meta-Causal Graph (NeurIPS 2025) |
| **B. Causal FM + virtual cell** | Effect of `do(gene=KO)` / drug on single-cell expression | Perturb-seq: Norman 2019, Replogle 2022 (genome-wide, >2.5M cells), plus drug screens | Borderline — public data is real but biased toward control state; eval is contested | High *application* novelty; methodologically derivative | GEARS (Nat. Biotech 2023); scGPT, scFoundation (~100M) |
| **C. Any-distribution causal PFN** | In-context interventional/counterfactual queries over a synthetic SCM prior | **Self-generated** — you sample the SCMs; data is infinite and free | **Yes** — the only one where data and eval are fully under your control | Medium-high; the live frontier line, room to push scale + counterfactuals | Do-PFN (arXiv 2506.06039); CausalPFN (arXiv 2506.07918); PFN/TabPFN |

### Why C is the spine

- **Data is a solved problem.** C generates its own training distribution by sampling SCMs (graph prior × mechanism prior × noise prior). No scraping, no licensing, no batch-effect hell. For a solo builder on a few GPUs this is decisive — you spend compute on the model, not on a data pipeline.
- **Ground truth is free.** Because *you* wrote the SCM, you know the true interventional and counterfactual answers exactly. Evaluation is unambiguous (RMSE on ATE/CATE, PEHE, counterfactual error), unlike B where "correct" perturbation response is itself a research question.
- **The frontier is reachable and unfinished.** Do-PFN and CausalPFN (both NeurIPS 2025) are recent, small (TabPFN-scale, ~tens of M params), and stop at *interventional effect estimation*. They do **not** push to 300–500M, do not do full **counterfactual** (abduction–action–prediction) queries, and use limited SCM priors. Each of those is a concrete, defensible delta.
- **It is the honest base layer.** Per the APSL framing (Pillar 4 → 5), C *is* the causal foundation model; B and A are C-with-a-domain and C-over-time. Building C first means B and A reuse the same architecture and pretraining loop.

A loses on tractability and eval clarity (you must build a sim/RL harness *and* the model, and "did it learn the right dynamics" has no clean metric). B loses on methodological novelty and a brutal external fact: the 2025 Nature Methods benchmark (s41592-025-02772-6) found that deep perturbation models *do not yet beat simple linear baselines* — so a from-scratch 300–500M virtual cell is a high-effort bet against a moving, skeptical field.

### Recommendation

- **PRIMARY spine — C, the any-distribution causal PFN.** Pretrain a 300–500M transformer on synthetic SCMs to answer association → intervention → counterfactual queries in-context. This is the buildable, evaluable, novel-at-scale core.
- **STRETCH extension — B as the killer application.** Once C trains and the eval harness is trusted, port it to virtual-cell perturbation: treat `do(gene=KO)` as an intervention node, fine-tune (or condition) the pretrained C on Norman/Replogle Perturb-seq, and benchmark against GEARS and a linear baseline. The stretch is to *beat the linear baseline honestly* on held-out perturbations — a clear, falsifiable win condition.

### Why this ordering composes cleanly

The spine is deliberately the most general object, so the others are restrictions/extensions, not rewrites:

| Layer | Relation to spine C | What changes | What is reused |
|---|---|---|---|
| **C (base)** | — | SCM prior + in-context head | everything |
| **B (app)** | C with a domain-specific prior | gene-gene KG prior over graphs; expression-shaped mechanisms; real-data fine-tune | architecture, pretraining loop, eval scaffolding, counterfactual head |
| **A (temporal)** | C unrolled over time | SCM → dynamic SCM / rollout prior; add time/RoPE over steps; intervention = `do()` at a timestep | tokenization of (vars, values, mask), the do-conditioning mechanism, the loss |

Concretely: the intervention-conditioning mechanism (how a `do(X=x)` token reshapes attention / masks the parent edges) is identical in all three. C defines it on a static table; B specializes the prior to biology; A repeats the static SCM block per timestep and adds temporal position. So the recommended build order — **C → B → A** — front-loads the part that is cheapest to train and easiest to verify, and each later step inherits a working, calibrated forward pass rather than starting over.

---

## Background: what a causal foundation model is

This capstone sits at the intersection of two amortized-inference threads: **prior-fitted networks** (train once on samples from a prior, do Bayesian inference in-context at test time) and **amortized causal inference** (learn the observation→graph or observation→effect map from synthetic SCMs). The lineage below is the direct technical scaffolding — what to reuse, what to extend.

### The PFN substrate (the engine we inherit)

- **Transformers Can Do Bayesian Inference (Müller, Hollmann, Pineda Arango, Grabocka, Hutter, 2021)** — [arXiv:2112.10510](https://arxiv.org/abs/2112.10510). Founds the PFN idea: restate posterior approximation as supervised learning over (train-set, query) → masked-label prediction; train on draws from any *prior you can sample*. Near-perfectly mimics GP posteriors with >200× speedups. **This is the loss and the training loop you copy verbatim.** Prior: GPs / BNNs. Arch: set-transformer over (x,y) tokens. Query: posterior predictive p(y|x, D).
- **TabPFN (Hollmann, Müller, Eggensperger, Hutter, 2022)** — [arXiv:2207.01848](https://arxiv.org/abs/2207.01848). PFN over a prior of *random SCMs/BNNs* generating tabular tasks; single forward pass beats tuned GBMs on small data with ~70× speedup. **Establishes that an SCM-shaped prior is learnable in-context** — the seed of the whole causal extension.
- **TabPFN v2 (Hollmann et al., Nature 2025; vol. 637, 319–326)** — new architecture, richer SCM-style prior, per-feature/per-sample attention, handles regression + missing/categorical data up to ~10k rows; explicitly framed as a *tabular foundation model* (generation, density estimation, embeddings, fine-tuning). **Your closest architectural reference for a 300–500M PFN.** (Companion analyses: *A Closer Look at TabPFN v2* [arXiv:2502.17361]; *TabICL* [arXiv:2502.05564] for scaling to larger n.)

### Amortized causal *structure* learning (the graph side)

- **CSIvA — Learning to Induce Causal Structure (Ke et al., 2022)** — [arXiv:2204.04875](https://arxiv.org/abs/2204.04875). Supervised transformer mapping a dataset of obs+interventional samples → graph distribution; attention alternates over samples and variables. **Shows full-graph recovery is amortizable**, but OOD-brittle when test mechanisms/noise differ from the prior.
- **AVICI (Lorch, Sussex, Bauer, Krause, 2022)** — [arXiv:2205.12934](https://arxiv.org/abs/2205.12934), NeurIPS 2022. Variational lower-bound on MI(structure; data), likelihood-free, permutation-equivariant axial attention, acyclicity via predicted-adjacency spectral radius. **The cleanest blueprint for a permutation-equivariant encoder over (n samples × d vars)** — reuse this axial-attention block.
- **Demystifying Amortized Causal Discovery with Transformers (Montagna et al., 2024)** — [arXiv:2405.16924]. Critical study: these models lean heavily on the prior's noise/sortability statistics. **Read this before designing your prior** — it dictates the invariances you must inject to avoid shortcut learning.

### Amortized causal *effect* estimation (the direct ancestors)

| Model | Prior trained on | Query answered | Headline result |
|---|---|---|---|
| **Do-PFN** ([arXiv:2506.06039](https://arxiv.org/abs/2506.06039)) | SCMs with sampled interventions; obs + do-data pairs | p(Y \| do(X), context) **without the graph** | Accurate interventional outcomes vs. backdoor/IV baselines that need the graph or unconfoundedness |
| **CausalPFN** (Balazadeh et al., [arXiv:2506.07918](https://arxiv.org/abs/2506.07918)) | Large library of DGPs satisfying ignorability | ATE & CATE, single forward pass | SOTA-avg on IHDP / Lalonde / ACIC; competitive uplift modeling; HF weights `vdblm/causalpfn` |
| **CausalFM** (Ma, Frauen, Javůrek, Feuerriegel, [arXiv:2506.10914](https://arxiv.org/abs/2506.10914), **ICLR 2026**) | Causality-inspired BNN priors over SCMs; formal validity criteria | back-door / front-door / IV adjustment in-context | Matches task-specific baselines; **the most principled prior-construction recipe — adopt its validity criteria** ([code](https://github.com/yccm/CausalFM)) |
| **FairPFN** ([arXiv:2407.05732](https://arxiv.org/abs/2407.05732); v2 [arXiv:2506.07049](https://arxiv.org/abs/2506.07049)) | Synthetic causal-fairness SCMs with protected attrs | counterfactually-fair prediction (removes protected-attr causal effect) | Removes protected-attr effect without the true graph; shows counterfactual queries are PFN-learnable |

### Freshest follow-ups (2026)

- **Interventional Time Series Priors for Causal Foundation Models (Thumm & Chen, 2026)** — [arXiv:2603.11090]. *CausalTimePrior*: temporal SCMs with lagged DAGs, nonlinear AR mechanisms, regime-switching, and hard/soft/time-varying interventions, yielding paired obs+interventional series. PFNs trained on it do in-context effect estimation on held-out TSCMs. **Relevant if you stretch the capstone to dynamics.**
- *Iterative Amortized Inference* [arXiv:2510.11471] and *amortized SCM-mechanism inference via fixed-point iterations* [arXiv:2410.06128, TMLR 2025] — note the recurring finding that **interventional sampling works far better than full counterfactual generation** in current amortized models.

### Synthesis: established vs. open

**Established:** that a single transformer pretrained on synthetic SCMs performs in-context *effect* estimation (ATE/CATE, back/front-door, IV) in one forward pass, matching task-specific estimators with no per-dataset retraining (Do-PFN, CausalPFN, CausalFM) — and that the *prior*, not the architecture, is the dominant design lever (CausalFM's validity criteria; the amortized-discovery critique). Architecturally, permutation-equivariant axial attention over (samples × variables) with v2-style per-feature embeddings is the consensus backbone, and 300–500M is comfortably in reach on a few GPUs.

**Open:** (1) genuine *counterfactual* (abduction-level) queries remain shakier than interventional ones; (2) robustness when real data leaves the synthetic prior's support (noise model, sortability, hidden confounding strength) is unsolved; (3) all published causal PFNs are *narrow*-scale (≤~10–20 vars, small n) — no one has shown the TabPFN-v2 → causal jump *at* 300–500M with a unified obs/interventional/counterfactual query interface; (4) honest uncertainty/identifiability signaling (telling the user when the effect is *not* identified from the context) is essentially untouched. **The capstone's defensible novelty is exactly (3)+(4): one mid-scale causal FM with a single query head spanning interventional/counterfactual/effect estimation, plus calibrated identifiability-aware uncertainty.**

---

## Architecture — a 300-500M causal PFN

The model is a **table-in, query-in, posterior-out** transformer: feed it an `n × d` dataset plus a causal query, get back a calibrated posterior predictive over the queried outcome in one forward pass. We borrow TabPFN v2's cell-tokenization and two-axis attention wholesale, and add a thin but load-bearing **do-conditioning** layer. Call it **Do-FM**.

### Input representation: the dataset as a grid of cells

- **One token per cell.** Each value `x_{ij}` (row `i`, variable `j`) becomes a token of width `d_model`, embedded as `value_proj(x_{ij}) + col_emb_j + type_emb(type_j)`. Mixed types: continuous → a small MLP on the (quantile-normalized) scalar; categorical → a learned codebook; missing → a dedicated mask vector. This is TabPFN v2 / TabICL cell embedding, unchanged.
- **Two-axis ("cell") attention** alternates per block (no positional encodings on either axis):
  - **Sample attention** — within a column, tokens attend across the `n` rows. Permutation-invariant over rows by construction (full attention, no row positions); this is the in-context "fit" step.
  - **Feature attention** — within a row, tokens attend across the `d` variables. This is where causal structure gets discovered: the variable-to-variable attention pattern is the model's implicit, amortized stand-in for the SCM's edges.
- **Train/query split.** Context rows carry observed `(features, outcome)`; query rows carry features with the outcome slot held out (a learned `[QRY]` embedding). The posterior predictive is read off the query rows — identical mechanics to TabPFN's in-context regression.
- **Variable-count handling.** Column count `d` is a sequence length on the feature axis, not a fixed input width, so the model is *equivariant* to adding/removing/permuting columns up to `d_max`. Column identity is carried only by a permutation-equivariant `col_emb` (sampled per-dataset at train time, à la TabPFN, so the model can't memorize column order).

### Encoding interventions and counterfactuals

This is the only genuinely new block versus TabPFN v2. We tag the **role** of each variable and the **regime** of each cell, then bias attention so the do-operator is respected.

- **Per-variable role channel** (added to every cell in a column): `{covariate, treatment T, outcome Y}`. Lets one model answer any `(T, Y)` pairing in a dataset without retraining.
- **Per-cell regime channel:** `{observational, do(·)=set}`. For an interventional query `do(X=x)`, the query row's `X` cells get the `do` tag and their value clamped to `x`.
- **Do-mask on feature attention.** The defining property of `do(X=x)` is *graph surgery*: incoming edges to `X` are cut. We implement this as an **attention bias on the feature axis** — for `do`-tagged cells, keys from would-be parents are masked, so `X` cannot be "explained" by other covariates; it only acts as a cause downstream. This is the architectural encoding of the do-operator, the piece Do-PFN/CausalPFN realize through the *prior* (intervened samples) rather than through attention masking; we do both — see below.
- **Counterfactuals (abduction–action–prediction).** A counterfactual query carries (i) a factual evidence row, (ii) a do-tagged intervention, and a `[CF]` query slot. The factual row is appended to the *context* with full attention (abduction: it pins down the latent noise/unit), while the intervened row sits in the query set under the do-mask (action + prediction). No explicit latent-noise inference; the abduction is amortized in-weights, conditioned on the paired factual row.
- **Query slot.** A typed `[QRY]`/`[CF]` token requests one of: posterior predictive `p(Y | do(X=x), covariates)`, **ATE** `E[Y|do(x_1)] − E[Y|do(x_0)]`, or **CATE** `τ(c) = E[Y|do(x_1),c] − E[Y|do(x_0),c]`.

### Output head and calibrated uncertainty

- **Riemann / bar-distribution head** (TabPFN v2's regression head): discretize `Y` into `K≈1000` adaptive bins and output a categorical over bins → a full predictive density, not a point estimate. Train with cross-entropy against the bin of the true (noised) outcome.
- **Effect estimates by differencing densities.** ATE/CATE are computed by running the head twice (treated vs. control query rows) and differencing the predictive means; the *distribution* of the effect comes from the paired predictive densities, giving an interval for free.
- **Calibration for free + a guarantee.** The Bayesian posterior-predictive objective yields well-calibrated densities directly (TabPFN's selling point). Wrap **split-conformal** on a held-out calibration slice of the context for a distribution-free coverage guarantee on `Y` / on the CATE under exchangeability — cheap, and turns "calibrated-ish" into a finite-sample bound.

### Backbone sizing — landing at ~380M

Target the middle of the band. With cell tokenization the context length is `L = n_max × d_max` *tokens*, which dominates memory, so we cap the grid and lean on depth/width for capacity.

| Hyperparameter | Value | Note |
|---|---|---|
| `d_model` | 1024 | |
| Layers | 24 | each = 1 sample-attn + 1 feature-attn + MLP |
| Heads | 16 (head dim 64) | |
| MLP ratio | 4× (hidden 4096) | SwiGLU |
| `n_max` rows | 4096 | context grid is capped, not the table |
| `d_max` vars | 128 | covariates + T + Y |
| `K` (Y bins) | 1024 | regression head |

- **Parameter count (non-embedding):** per layer ≈ attention `4·d²` + MLP `2·4·d²` = `12·d² ≈ 12.6M`; two attention axes share the MLP but each axis adds its own `4·d²`, so ≈ `16·d² ≈ 16.8M`/block × 24 ≈ **403M**. Trim to 22 layers or `d_model=960` to sit nearer 350M. Cell/value embeddings and the `K`-way head add ~5–10M.
- **Compute/memory implications:**
  - **Attention is axis-factored, not full `L²`.** Sample attention costs `O(d · n²)`, feature attention `O(n · d²)`. At `n=4096, d=128` that's ~`128·4096²` ≈ 2.1B vs. a naive `(n·d)²` ≈ 2.7e11 — a **~125×** saving; this factorization is what makes 4k×128 grids tractable on a few GPUs.
  - **The grid cap is the real ceiling.** A single 4096×128 cell grid in bf16 activations is the memory unit; batch in *datasets*, not rows. Expect to fit only a handful of grids per 80GB GPU at 24 layers — use activation checkpointing on the feature/sample blocks and FlashAttention along each axis.
  - **Bigger tables at inference:** subsample/bag rows (ensemble over row-subsets, TabPFN-style) and, for `d>128`, do feature bagging — the equivariant column embedding makes this sound.

### Borrowed vs. new

| Component | Source | Status |
|---|---|---|
| Cell tokenization, mixed types, missing | TabPFN v2 (Nature 2025) / TabICL (arXiv 2502.05564) | borrowed |
| Two-axis sample/feature attention, no positions | TabPFN v2 / TabICL | borrowed |
| Riemann bar-distribution regression head | TabPFN v2 | borrowed |
| In-context posterior-predictive objective | Müller et al., *PFNs* (2022); TabPFN v1 (arXiv 2207.01848) | borrowed |
| Effect estimation from an SCM-with-interventions prior | Do-PFN, CausalPFN (both NeurIPS 2025) | borrowed (prior side) |
| Role channel (treatment/outcome) + regime channel (obs/do) | — | **new** |
| **Do-mask on feature attention** (graph surgery in-architecture) | — | **new** |
| Counterfactual `[CF]` slot with factual-row abduction | — | **new** |
| Conformal wrapper on `Y`/CATE for coverage | Conformal prediction (see [[conformal]]) | new-for-causal-PFN |

The bet: Do-PFN and CausalPFN get the do-operator entirely through the **prior** (they pretrain on SCMs and sample post-intervention data, then treat effect estimation as ordinary ICL). We keep that prior but *additionally* encode the surgery as an **attention mask**, so the inductive bias is explicit rather than purely learned — cheaper to train to the same coverage on a few GPUs, and it makes counterfactual abduction a first-class input rather than a prompt-engineering trick.

---

## The synthetic SCM prior & data generator — the heart of a PFN

A PFN never sees a real causal dataset during training. It sees millions of synthetic worlds, and its entire test-time behavior is the Bayesian posterior under whatever distribution generated those worlds. The prior **is** the model. For a Causal FM, "the prior" means a sampler over structural causal models *plus* a sampler over the causal queries you want answered. Get this right and the rest is engineering; get it wrong and no amount of scale or attention-fiddling recovers it. This section specifies the generator concretely.

### The generative recipe: one training example

Each training example is a tuple `(D_obs, query, answer)`. The generator is a hierarchical sampler — draw a world, draw data from it, draw a question, compute the truth analytically because *you built the world and know it exactly*. This last point is the whole trick: ground truth for interventions/counterfactuals is free in simulation and impossible in the wild.

```
sample_example():
    G        ~ p(DAG)                      # structure: nodes, edges
    F, U     ~ p(mechanisms, noise | G)    # an SCM M = (G, F, p(U))
    D_obs    = simulate(M, n_rows)         # observational table
    query    ~ p(query | M)                # do(X=x)? counterfactual? ATE/CATE?
    answer   = oracle(M, query)            # exact, from the known SCM
    return (D_obs, query, answer)
```

`oracle` is the part real data can never give you. For an interventional query you mutilate `G` (cut incoming edges of the intervened node, fix its value) and either solve the resulting SCM in closed form (linear-Gaussian) or Monte-Carlo the truncated factorization with many fresh noise draws. For a counterfactual you run **abduction → action → prediction**: infer the posterior over exogenous `U` given the factual row, then re-simulate under the intervention holding `U` fixed. Cache the per-query oracle so you are not re-simulating inside the training loop.

### Sampling the DAG

Concrete, defensible defaults for an initial 300–500M build:

| Knob | Suggested prior | Why |
|---|---|---|
| Node count `d` | `~ U{3, ..., 20}` early; widen to ~40 later | curriculum; attention cost grows with `d` |
| Edge density | Erdős–Rényi `p` s.t. expected in-degree ∈ [1,3], OR scale-free (Barabási–Albert) | mix of "civilized" and hubby graphs matches real tabular causality |
| Acyclicity | sample a random topological order, only allow forward edges | guarantees a DAG by construction, zero rejection |
| `n_rows` | `~ U{20, ..., 1024}` (log-uniform) | model must work in small-data regimes, where it earns its keep |

Always feed columns to the model in **scrambled order** (not topological), or it learns to read structure off position. Train with variable `d` and `n` so the transformer is permutation-/size-agnostic at test time, exactly as TabPFN does (Müller et al., *Transformers Can Do Bayesian Inference*, arXiv 2112.10510; Hollmann et al., *TabPFN*, arXiv 2207.01848).

### Mechanism & noise families

The mechanism prior controls the function class the model can fit. Mix them per-node so a single SCM is heterogeneous:

| Family | Form `xᵢ = fᵢ(pa(i), uᵢ)` | Coverage it buys |
|---|---|---|
| Linear-Gaussian | `wᵀ pa + uᵢ`, `u ~ N` | identifiable baseline; closed-form oracle |
| Additive-noise MLP | `MLP(pa) + uᵢ` | smooth nonlinear, ANM-identifiable direction |
| GP-drawn | `f ~ GP(pa)` (RFF approx) | rich nonparametric shapes, cheap to sample |
| Post-nonlinear | `g(h(pa) + uᵢ)` | breaks pure ANM; tests robustness |
| Discrete/CPT | tabular conditionals | categorical columns, mixed types |

Noise: don't ship only Gaussian. Mix `N`, heavy-tailed (`t`, Laplace), skewed, and heteroscedastic (`uᵢ` scale a function of parents) — real residuals are not symmetric. Randomize weight scales and add a standardization step, because the model will key on marginal scale otherwise.

### Confounding & hidden variables — the part that makes it *causal*

A regressor can be trained on `D_obs`; a *causal* model must be forced to confront the gap between `P(Y|X)` and `P(Y|do(X))`. Bake this into the prior:

- Sample SCMs **with** latent confounders (common cause of treatment and outcome), then **delete those columns** from `D_obs` before the model sees it. The oracle still computes the true interventional answer using the full SCM. This is what teaches the model that conditioning ≠ intervening.
- Vary the confounding strength continuously, including the unconfounded special case, so the model learns to *detect* how much adjustment is warranted from the observational signal alone (Do-PFN and CausalPFN, both NeurIPS 2025, follow this graph-free, prior-over-SCMs route).
- Optionally inject selection bias and measurement noise as further hidden mechanisms — but add these only after the confounded version trains stably.

### The training objective (the PFN loss)

Standard PFN posterior-predictive NLL. With params `θ`, over the prior `p`:

`L(θ) = E_{M ~ p} E_{D_obs, q, a ~ M} [ −log q_θ(a | D_obs, q) ]`

Minimizing this makes `q_θ` converge to the true Bayesian posterior predictive `p(a | D_obs, q)` **averaged over the prior** — no per-dataset retraining, one forward pass at test time. Practicalities: discretize continuous answers into a Riemann/bar distribution (TabPFN's trick) so a scalar effect becomes a calibrated predictive *distribution*, giving you intervals for free; for ATE/CATE you can regress the answer directly or predict the post-intervention outcome distribution and read the contrast off it. The loss is a proper scoring rule, so calibration is trained, not bolted on (cf. [[calibration]] and the repo's [[conformal]] note for the complementary distribution-free view).

### Curriculum & the sim-to-real gap

- **Easy → hard:** start small `d`, low density, linear-Gaussian, no confounding; anneal toward large `d`, nonlinear mixes, heavy confounding. This stabilizes early optimization and mirrors APSL's "prior design determines test-time coverage."
- **Coverage rule of thumb:** the model can only answer query types and SCM shapes it saw in training. If real targets are CATE on mixed-type tables with unobserved confounders, the prior must routinely emit exactly that. Audit by sampling from the prior and eyeballing marginals/effect-size histograms against your target real datasets (IHDP, Lalonde, ACIC).
- **The gap is a prior-mismatch gap, nothing more.** Sim-to-real failure = your `p(M)` puts low mass where reality lives. Fixes are all prior edits: richer mechanisms, realistic noise, plausible confounding rates, correlated/structured covariates instead of clean independent draws.

### Pitfalls

- **Prior misspecification is silent.** A confidently-wrong, well-calibrated-on-prior model that never saw a regime is the failure mode. There is no in-distribution signal warning you; only out-of-prior evaluation reveals it.
- **Identifiability bounds what is learnable.** If a query is non-identifiable from observational data over your prior (e.g. ATE under arbitrary unobserved confounding with no instrument), the *Bayes-optimal* answer is the prior-averaged posterior — the model will hedge, and that hedging is correct, not a bug. Don't expect point accuracy where the math forbids it. Counterfactuals are even stricter: they need the full functional form, so a counterfactual-capable prior must commit to specific mechanism families and the model inherits exactly those assumptions.
- **Leakage via ordering/scale.** Column order, marginal scale, and row count can all become shortcuts; scramble, standardize, and randomize them.
- **Oracle cost.** Counterfactual abduction and MC interventions dominate generation time — precompute and shard the synthetic corpus rather than sampling inside the training step.

---

## Build plan, compute & evaluation

The PFN bet is that *training cost is one-time and bounded by synthetic-data generation + a fixed pretraining run*; there is no per-dataset fitting. So the plan is staged not by data acquisition (it's all synthetic) but by **prior richness** and **model size**. Build the loop tiny, prove calibration, then scale the prior and the model. Each phase ends with a go/no-go gate.

### The build

**What I am building:** a 300–500M-parameter "Do-FM" — a Prior-Fitted Network over synthetic SCMs that, given an `n × d` table plus a typed causal query, returns a calibrated posterior predictive for interventional (`p(Y|do(X=x))`), counterfactual (`[CF]` abduction–action–prediction), and effect-estimation (ATE/CATE) targets in a single forward pass, with a split-conformal wrapper for distribution-free coverage and identifiability-aware uncertainty.

**The "ships when" line:** the capstone **ships** when, on a frozen bank of ≥1000 held-out SCMs spanning the full prior, the model (1) matches or beats DML/doubly-robust and a strong amortized baseline (CausalPFN-class) on ATE error and √PEHE; (2) achieves 90% predictive-interval coverage in [0.87, 0.93] with non-degenerate widths; **and** (3) on ≥3 of {IHDP, ACIC, Lalonde/Jobs, Twins} lands within a small, pre-registered margin of published amortized-PFN results — all *in a single forward pass with zero per-dataset retraining*.

**Key design decisions:**
- *Spine = the most general object (any-distribution causal PFN).* Virtual-cell (B) and dynamics (A) are restrictions/extensions that reuse the same architecture, loss, and eval scaffolding — so build C first.
- *Do-conditioning is both prior-side and architecture-side.* Keep the Do-PFN/CausalPFN trick of sampling post-intervention data from the prior, but *additionally* encode graph surgery as a feature-axis attention mask so the inductive bias is explicit and cheaper to train.
- *Calibration is graded, not decorative.* A trained proper-scoring-rule head (Riemann/bar) plus a split-conformal wrapper; coverage collapse on out-of-prior SCMs is the canary metric, reported at every phase gate.
- *Identifiability-aware hedging is a feature.* Include non-identifiable regimes in the prior and verify the model *widens intervals / abstains* there rather than confidently guessing.
- *The data engine, not the GPU, is the bottleneck.* Generate SCMs on the fly in a sharded CPU producer pool; precompute/cache oracle answers (counterfactual abduction + MC interventions dominate).

### Phase 0 — Prove the loop (tiny PFN, ~10–50M, ~1–2 weeks on 1 GPU)
- **Prior:** linear-Gaussian SCMs only. Sample a random DAG (Erdős–Rényi or scale-free, 3–12 nodes), linear mechanisms with Gaussian noise, one designated treatment `T` and outcome `Y`, optional linear confounders into both.
- **Task & loss:** in-context interventional prediction. Context = `n` observational rows `(X, T, Y)`; query = a row with `do(T=t)`; target = the *true* `P(Y | do(T=t))` from the simulator. Train with a Riemann/bar-distribution head (TabPFN-style discretized output) under NLL so you get a predictive distribution, not a point — this is what makes calibration measurable from day one.
- **Architecture:** TabPFN v2-style per-cell tokenization, attention over rows (datapoints) with feature-axis attention, *permutation-invariant over rows*, treatment/intervention encoded as a special column flag. No causal-attention masking yet.
- **Ground truth advantage:** because the SCM is known, you can compute exact ATE and the true interventional density per dataset — supervise directly and measure error exactly.
- **Gate:** on held-out *linear* SCMs, ATE error and interventional NLL beat a linear-regression-with-backdoor baseline, **and** 90% predictive intervals have empirical coverage in ~[0.86, 0.94]. If coverage is off, the head/loss is wrong — fix before scaling.

### Phase 1 — Scale mechanisms + DAG complexity (~2–4 weeks, 1–2 GPUs)
- **Prior enrichment (one axis at a time, re-check coverage after each):** nonlinear mechanisms (MLP/GP/neural-net draws), heteroscedastic and non-Gaussian noise, mixed continuous/categorical variables, more nodes (up to ~50–100), hidden confounders (so backdoor adjustment is *not* always available), and varying overlap/positivity.
- **Add counterfactual supervision:** sample noise `U`, generate factual + counterfactual under the same `U` (abduction-action-prediction in the simulator), train an ITE/counterfactual head alongside the interventional one.
- **Architecture upgrades:** longer context (more rows, more features), the do-mask on feature attention, CSIvA-style two-axis attention; consider feeding an inferred-graph token but keep the model *graph-free at test time* (à la Do-PFN/CausalPFN).
- **Gate:** no catastrophic regression on linear SCMs; calibration holds across the *enriched* prior; PEHE on held-out nonlinear+confounded SCMs is below a doubly-robust / DML baseline.

### Phase 2 — Scale to 300–500M with the full prior (~2–4 weeks of training, a few GPUs)
- Freeze the prior spec, scale width/depth to 300–500M, train one long run with WSD schedule, bf16, FlashAttention, gradient checkpointing, μP-transferred hyperparameters from Phase 1.
- **Data engine is the real bottleneck:** generate SCMs + simulate on the fly in a sharded CPU producer pool feeding the GPU; cache nothing for the table itself (infinite fresh prior samples beat reuse), but precompute the expensive oracle answers. Target keeping GPUs >50% MFU; if the simulator starves them, precompute a large rolling buffer.

| Phase | Params | Prior | Hardware | Wall-clock |
|---|---|---|---|---|
| 0 | 10–50M | linear-Gaussian | 1× A100/H100 | days |
| 1 | 50–150M | nonlinear, confounded, mixed-type | 1–2× A100/H100 | 1–3 weeks |
| 2 | 300–500M | full prior + counterfactuals | 4–8× A100/H100 | 1–3 weeks |

**Rough compute:** a 300–500M model at C ≈ 6ND over ~50–100B effective synthetic tokens is roughly a few× 10²⁰ FLOPs — order **2–4 GPU-weeks on 8×H100**, i.e. days of training plus a few weeks of iteration. This is days-not-months; the dominant ongoing cost is the SCM simulator, so budget engineering time there, not GPU dollars.

### Evaluation (statistical rigor is graded)
- **Synthetic held-out SCMs (primary, because ground truth exists):**
  - **Effect error:** absolute ATE error and √PEHE (CATE) vs. simulator truth.
  - **Interventional-prediction error:** NLL / CRPS of predicted `P(Y|do(t))` vs. true interventional density.
  - **Calibration & coverage:** empirical coverage of nominal 80/90/95% predictive intervals, reliability diagrams, and ECE on the predictive head. Report coverage *and* interval width (over-wide intervals are a failure). This is the differentiator vs. point-estimate causal ML — connect to [[conformal]] for a distribution-free wrapper and [[calibration]] for proper-scoring-rule checks.
- **Real benchmarks (transfer, no retraining):** IHDP (√PEHE), ACIC 2016/2018 (ATE error across DGPs), Lalonde/Jobs (policy/ATT error vs. experimental benchmark), Twins (CATE + AUC on mortality). Match the protocol Do-PFN/CausalPFN report so numbers are comparable; expect to be in the neighborhood of CausalPFN, not necessarily SOTA, for a first build.
- **Ablations (the science):**
  1. **Prior richness:** train on linear-only vs. full prior, test on nonlinear held-outs → shows the prior, not the architecture, drives generalization.
  2. **With/without confounding:** include vs. exclude hidden confounders in the prior; measure test bias on confounded held-outs (does the model learn to *not* assume ignorability?).
  3. **In-context dataset size:** sweep `n` (e.g. 20→2000 rows) → expect monotone error decrease and shrinking intervals; flat curves mean the model ignores context.
  4. **Do-mask on/off** and **counterfactual-head on/off**; optional graph-given vs. graph-free.
- **Statistical hygiene:** bootstrap CIs over the held-out SCM population for every metric ([[bootstrap-confidence-intervals-for-eval-metrics]]); fix seeds for the eval SCM bank; never tune on the real benchmarks.

### Key risks
- **Prior misspecification (the central risk):** test SCMs outside the training prior → silent, confident, wrong answers. Mitigate by deliberately widening the prior and reporting OOD coverage; treat coverage collapse as the canary.
- **Confounding / identifiability:** if the prior implies effects are always identifiable, the model learns to assume ignorability and is biased on real confounded data. Explicitly include unidentifiable regimes and *check the model abstains/widens intervals* there.
- **Data-engine starvation:** a slow simulator caps MFU and silently turns "days" into "weeks." Profile and parallelize the SCM sampler first.
- **Calibration drift on scale-up:** the discretized head can miscalibrate as the prior/model grows; re-run reliability diagrams at every phase gate, not just at the end.
- **Benchmark leakage / overfitting the test bank:** rotate the held-out SCM seeds and keep real benchmarks strictly untouched until the final gate.

---

## Build log
*(append entries as you work)*

## Rubric self-assessment
| Axis | Score (1-5) | Evidence |
|------|-------------|---------|
| Correctness & reproducibility | — | — |
| Statistical rigor | — | — |
| Depth / from-scratch-ness | — | — |
| Engineering quality | — | — |
| Communication | — | — |

**Pass threshold:** >=4 on Correctness + Communication, >=3 on rest.

## Defense notes
**What I would push back on if challenged:**
- *"This is just TabPFN with extra labels."* No — the do-mask makes graph surgery an explicit inductive bias rather than something purely learned from intervened samples, and the unified head spanning interventional/counterfactual/effect at 300–500M is a regime no published causal PFN occupies. The architecture-side encoding is a testable claim (the do-mask ablation isolates it).
- *"Synthetic evaluation is self-fulfilling."* The primary metric is on held-out SCMs the model never saw, *and* we deliberately include out-of-prior and non-identifiable regimes where the Bayes-optimal answer is to hedge — so the eval rewards honest uncertainty, not memorized point accuracy. Real benchmarks (IHDP/ACIC/Lalonde/Twins) are the external check.
- *"Calibration is a side feature."* It is the central deliverable: a proper-scoring-rule head plus split-conformal gives a finite-sample coverage guarantee under exchangeability, which is exactly what point-estimate causal ML lacks.

**The weakest part of this work:** prior-misspecification / sim-to-real. The model is only ever as good as `p(M)`; failures on real data are silent (well-calibrated-on-prior, confidently-wrong-off-prior) and there is no in-distribution signal that warns you. Counterfactual queries are the most exposed, since they inherit the exact mechanism families committed to in the prior.

**What I would do differently:**
- Invest earlier in an *out-of-prior stress suite* (target-dataset marginal/effect-size audits) rather than treating real benchmarks as the only OOD check — the canary needs to fire before the final gate.
- Consider learning an explicit identifiability/abstention signal as a separate head, rather than relying on interval width alone to communicate "not identified from this context."

## See also
[APSL — Algorithmic Probabilistic Structure Learning](../../seminars/apsl-structure-learning.md) · [Statistical & Probabilistic Foundations](../../seminars/statistical-probabilistic-foundations.md) · [Data Foundations](../../seminars/data-foundations.md) · [AIDO — AI-Driven Digital Organism (virtual-cell stretch)](../../seminars/foundation-models/12-virtual-cells/aido-digital-organism.md) · [Concept library: causal bricks](../../library/bricks/README.md)
