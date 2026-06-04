# APSL v3: Algorithmic Probabilistic Structure Learning
### Graduate Course — Full Syllabus with Foundation Models Integration

**Tagline:** *"Learning distributions over structured objects, efficiently — from Bayesian networks to causal worlds to foundation models."*

**Level:** Advanced MS/PhD (Year 1–2)
**Format:** 18-week semester · 2 lectures/week · 1 paper discussion/week · 4 problem sets · 1 final project
**What changed in v3:** A dedicated **Pillar 5: Foundation Models as Structure Learners** is added — not as "LLMs with graphs bolted on" but as the natural next step: *what happens when the inference machine (Pillar 2) scales to billion parameters and is pre-trained on the entire structured world?*

***

## The Conceptual Map (v3)

All six pillars answer a single question from a different vantage point:

> **"How do you define, represent, and efficiently sample from a distribution over structured objects — and what happens when the inference machine itself is a foundation model?"**

```
Pillar 1: Foundations
  → What IS a structured object? How do we put probability on it?
        ↓
Pillar 2: Inference Algorithms
  → Exact DP · MCMC · VI · GFlowNets · PFNs
        ↓
Pillar 3: Generative Models over Structures
  → AR · Flow · Diffusion · EBM · GFlowNet
        ↓
Pillar 4: Causal ML + Structure Learning
  → SCMs · Amortized Discovery · CRL · Deep Causal Models
        ↓
Pillar 5: Foundation Models as Structure Learners   ← NEW
  → Graph FMs · LLM-as-structure-learner · GFlowNet-LLM · Causal FMs
        ↓
Pillar 6: Research Frontiers + Projects
  → Open problems, positioning, execution
```

***

## PILLAR 1 — Foundations (Weeks 1–3)

### Week 1: Structured Objects and Their Representations

**Lecture 1A — What Is a Structured Object?**
- Flat vectors in \( \mathbb{R}^d \) vs. graphs, trees, DAGs, sets, permutations, programs
- The combinatorial explosion: why enumeration is impossible
- Running examples set up for all 18 weeks: molecules, causal DAGs, programs, treatment-outcome tables, token sequences as latent structure
- Constraints as first-class citizens: acyclicity (DAGs), planarity (graphs), validity (programs)

**Lecture 1B — Data Structures as Modeling Decisions**
- Adjacency matrices (dense/sparse), edge lists, adjacency sets
- CPDAGs and equivalence class representations[1]
- Tries, parse forests, CYK tables
- Canonical orderings and symmetry breaking as generative modeling choices
- Incremental scoring: sub-quadratic BIC/BDe recomputation after one edge change[2]
- **Key framing:** the data structure you choose for the structured object determines the computational complexity of every inference algorithm you run on it

**Paper Discussion:** Koller & Friedman, *PGMs* Ch. 1–3[3]

***

### Week 2: Probability over Structured Objects

**Lecture 2A — Distributions on Combinatorial Spaces**
- Distributions over graphs, permutations, partitions: foundations
- Exchangeability, symmetry, invariance; de Finetti's theorem
- Exponential families over structures: sufficient statistics as clique features
- Partition function intractability: the central challenge

**Lecture 2B — BNP + Causal Priors**
- Dirichlet Process, IBP, Pitman-Yor, Gaussian Process
- SCMs as structured priors: TabPFN's prior is a distribution over SCMs with preference for simple causal structures[4][5]
- Pearl's causal ladder as evaluation criterion: association → intervention → counterfactual[6]
- What does it mean to put a prior over causal *mechanisms* vs. causal *graphs*?

**Paper Discussion:** Pearl, *Causality* Ch. 1–2 + Orbanz & Teh, *Bayesian Nonparametric Models*

***

### Week 3: Statistical Learning Theory for Structured Spaces

**Lecture 3A — Learnability of Structured Hypothesis Classes**
- VC dimension, Rademacher complexity, covering numbers for structured classes
- PAC learning over combinatorial spaces

**Lecture 3B — PAC-Bayes and Causal Connections**
- PAC-Bayes bounds: \( \text{KL}(Q \| P) \) as structural complexity
- MDL and algorithmic information theory
- Connection to PFNs: training on a prior over SCMs is equivalent to choosing a structural complexity penalty[7]
- **Foreshadowing:** foundation model pre-training is PAC-Bayes at scale — the pre-training distribution is the prior \( P \), and the in-context dataset is the evidence that updates to \( Q \)

**Paper Discussion:** McAllester (2003), *PAC-Bayes Bounds*

***

## PILLAR 2 — Inference Algorithms (Weeks 4–7)

### Week 4: Exact Inference as Dynamic Programming

**Lecture 4A — Variable Elimination + Belief Propagation**
- Message passing as DP on trees (exact) and graphs (loopy, approximate)
- Treewidth as the structural complexity measure[8]
- Junction tree: converting arbitrary graphs to trees for exact inference

**Lecture 4B — The DP Landscape**
- Inside-outside algorithm for PCFGs; CYK as DP over parse forests
- Viterbi = MAP; inside = partition function; outside = marginals
- **Key insight:** every exact inference algorithm is DP over a structural decomposition — the decomposition is an algorithmic design choice with direct complexity consequences

**Paper Discussion:** Wainwright & Jordan, *Graphical Models, Exponential Families, VI* (Ch. 2–3)

***

### Week 5: Approximate Inference — MCMC + VI

**Lecture 5A — MCMC for Structured Spaces**
- MH over graphs: edge flip, node swap, subgraph replacement
- Mixing time problem: the core limitation motivating amortization
- Specialized DAG moves preserving acyclicity

**Lecture 5B — Variational Inference**
- ELBO, mean-field, structured VI families
- Reparameterization trick; amortization gap[9]

**Paper Discussion:** Blei et al. (2017), *Variational Inference: A Review*

***

### Week 6: GFlowNets — Amortized Samplers

**Lecture 6A — GFlowNet Foundations**
- Construction trajectories, state DAG, flow conservation as Kirchhoff's laws[10][11]
- Training objectives: TB, DB, FM; off-policy replay buffers
- Continuous GFlowNets; Non-acyclic GFlowNets (ICML 2025)[12][13]

**Lecture 6B — GFlowNets × VI × RL: Unified View**
- Reward-proportional sampling ≠ reward maximization[14]
- VI–GFlowNet equivalence under specific conditions[15][9]
- GFlowNet as amortized MCMC without sequential correlation
- **Foreshadowing:** GFlowNet fine-tuning of LLMs (Pillar 5) is this same framework applied at billion-parameter scale[16]

**Paper Discussion:** Bengio et al. (2023), *GFlowNet Foundations*; Malkin et al. (2022), *GFlowNets and VI*[11][9]

***

### Week 7: Prior-Fitted Networks — Amortized Bayesian Inference

**Lecture 7A — The PFN Framework**
- Core idea: train once on synthetic data from a prior → approximate the posterior predictive in one forward pass[17]
- Formal guarantee: PFN approximates \( P(y^* \mid \mathcal{D}_{\text{train}}) \) for any \( \mathcal{D}_{\text{train}} \) drawn from the prior[17]
- TabPFN: SCM-based prior → state-of-the-art tabular classification in 2.8 seconds[5][4]
- TabPFN v2 (Nature, 2025): pre-trained on 130 million synthetic datasets from SCM priors; supports density estimation, data generation, fine-tuning[18][19][20]

**Lecture 7B — PFN Landscape + The Amortization Trinity**
- LC-PFN (NeurIPS 2023): learning curve extrapolation in one forward pass[21][22]
- PFNs4BO: PFNs as flexible surrogates for Bayesian optimization[23]
- In-context learning as Bayesian inference: transformers learn the prior from meta-training[24][25]
- **The amortization trinity:** GFlowNet (amortizes sampler) · PFN (amortizes Bayesian update) · Foundation Model (amortizes the entire reasoning process at scale)
- **Key design question:** the prior over SCMs is an *algorithmic decision* with direct empirical consequences — it determines test-time coverage

**Paper Discussion:** Müller et al. (2022), *PFNs: Transformers Can Do Bayesian Inference*; Hollmann et al., *TabPFN*[5][17]

***

## PILLAR 3 — Generative Models over Structured Objects (Weeks 8–11)

### Week 8: Autoregressive Models over Structures

**Lecture 8A — The Ordering Problem**
- Graph generation: GraphRNN (node-by-node), GRAN (edge-by-edge)[26]
- AR over trees and programs: PCFGs as special case; grammar-constrained decoding
- Causal AR: generating variables in topological order of a DAG

**Lecture 8B — Permutation-Invariant + Masked AR**
- Order-agnostic training; BERT-style masked autoencoding as partial AR
- Equivariant architectures respecting structural symmetries

**Paper Discussion:** You et al. (2018), *GraphRNN*[26]

***

### Week 9: Flow, Score-Based, and Diffusion Models

**Lecture 9A — Flows on Structured Spaces**
- Flow matching, Riemannian flow matching, CTMCs as discrete flow[27][28]
- Causal normalizing flows: parameterizing SCM mechanisms as flows for exact counterfactual inference[29][30]
- Causally consistent flows: observational + interventional equivalence guarantees[31][32]

**Lecture 9B — Discrete Flow Matching + Graph Diffusion**
- Edit flows: variable-length generation via CTMC edit operations[33]
- Graph diffusion (Cometh); constrained graph generation (ConStruct)[34][35]
- Generator Matching: unified theory of discrete/continuous flows[27]

**Paper Discussion:** Gat et al. (2024), *Discrete Flow Matching*[33]

***

### Week 10: EBMs + Combinatorial Optimization

**Lecture 10A — EBMs and Contrastive Divergence**
- EBMs for structured prediction; CD; Langevin dynamics[36]
- Energy Matching (2025): unifying EBMs + flow matching[37]

**Lecture 10B — Generative NCO**
- Diffusion-based NCO: distributions over combinatorial solutions[38]
- Generation as search operator (GenSCO); MCTS + neural heuristics[39][40]

**Paper Discussion:** DIFUSCO[38]

***

### Week 11: GFlowNets over Molecules, DAGs, Programs

**Lecture 11A — GFlowNets for Structured Generation**
- Molecules: atom-by-atom with Atomic GFlowNet (ICML 2025)[41][42]
- Hierarchical GFlowNets (crystal generation)[43]
- Interpretability: sparse autoencoders + concept attribution on GFlowNet policies[44]

**Lecture 11B — GFlowNets for Causal Structure Learning**
- Bayesian structure learning with GFlowNets[2]
- CPDAG-GFN: equivalence class data structures inside GFlowNets[1]
- Amortized causal structure learning (NeurIPS 2022)[45][46]

**Paper Discussion:** Deleu et al. (UAI 2022); Lorch et al. (NeurIPS 2022)[45][2]

***

## PILLAR 4 — Causal ML + Causal Structure Learning (Weeks 12–13)

### Week 12: Deep SCMs + Amortized Causal Inference

**Lecture 12A — DSCMs + Pearl's Ladder**
- SCM formalism; do-calculus; abduction-action-prediction triplet[30][6]
- Deep SCMs: parameterizing mechanisms with flows, VAEs, diffusion[47][30]
- Causally consistent normalizing flows[32][31]

**Lecture 12B — PFNs for Causal Effect Estimation**
- Do-PFN (NeurIPS 2025): causal effect estimation without knowing the causal graph[48][49][50]
- CausalPFN (NeurIPS 2025): amortized effect estimation via in-context learning on SCM-generated data; SOTA on IHDP, Lalonde, ACIC[51][52]
- FairPFN (ICLR 2025): causal fairness via PFN pre-training[53]
- **Key design question:** what prior over SCMs gives good test-time coverage?

**Paper Discussion:** Pawlowski et al. (NeurIPS 2020), *Deep Structural Causal Models*; Do-PFN[49][30]

***

### Week 13: Causal Representation Learning + Causal World Models

**Lecture 13A — Causal Representation Learning**
- CRL problem: recover latent \( Z \) and causal graph \( G \) from observations \( X \)[54]
- Score-based CRL: score differences across environments encode the latent DAG[54]
- Grouping-based identifiability without interventions (ICML 2024)[55]
- General identifiability + achievability (Google Research)[56]
- Generative Intervention Models (GIMs, ICML 2025): perturbation → intervention in causal model[57]

**Lecture 13B — Causal World Models**
- Causal World Model Induction (CWMI)[58]
- Meta-Causal Graph (NeurIPS 2025): encoding shifts of causal structure across world states[59]
- WhatIfGAN: generative models arranged by causal graph, trained with unobserved confounders[60]
- **Synthesis:** a causal world model is a generative model over SCMs, learned via amortized inference, supporting all three rungs of Pearl's ladder at test time[47][58]

**Paper Discussion:** CWMI survey; Meta-Causal Graph[58][59]

***

## PILLAR 5 — Foundation Models as Structure Learners (Weeks 14–16)

**(New — the frontier pillar)**

### Week 14: What Is a Foundation Model for Structured Objects?

**Lecture 14A — The Foundation Model Paradigm, Reframed**

The standard framing of foundation models is: *"large model pre-trained on broad data, adapted to many tasks."* The APSL framing is different and sharper:

> A **foundation model for structured objects** is an amortized inference machine pre-trained on a distribution over structured objects (or data generated by them), such that at test time it performs approximate Bayesian inference — or exact structured generation — in a single forward pass.

Under this framing, three classes emerge:[61][62][7]

| Class | Pre-trained on | Test-time task | Example |
|---|---|---|---|
| **PFN-style** | Synthetic datasets from a prior over SCMs/structures | Posterior predictive for new dataset | TabPFN v2, Do-PFN, CausalPFN |
| **Graph Foundation Models** | Large-scale multi-domain graph corpora | Node/edge/graph-level tasks, zero-shot transfer | AutoGFM, GFM-RAG, GraphBFF |
| **LLM as structure reasoner** | Natural language + structured data | In-context structure learning, causal graph elicitation | GPT/Claude + BSL, LLM-constructed BNs |

The NeurIPS 2025 workshop *Structured Probabilistic Inference and Generative Modeling* explicitly asked: *"Is probabilistic inference still relevant in the era of foundation models?"* — this lecture answers that question from first principles.[63]

**Lecture 14B — ICML Workshop on Foundation Models for Structured Data (FMSD)**

The 1st ICML 2025 FMSD workshop unified the tabular and time-series communities around foundation models for structured data. The 2nd FMSD workshop runs at ICML 2026. The core questions driving this research community:[64][62][65][66][61]

- Can a single foundation model generalize across *heterogeneous* structured domains?
- What is the right pre-training objective for structured objects?
- When does in-context learning fail for structured inference, and why?
- How do structural inductive biases (graph symmetry, causal acyclicity) get encoded in transformer weights?

**Paper Discussion:** ICML FMSD workshop 2025 overview papers[65][61]

***

### Week 15: Graph Foundation Models

**Lecture 15A — From Task-Specific GNNs to Graph Foundation Models**
- The graph FM problem: a single model pre-trained on large-scale multi-domain graph corpora (social, web, academic, molecular) that generalizes via fine-tuning, prompting, or zero-shot inference[67]
- Key challenges vs. LLMs: no natural tokenization for graphs; structural heterogeneity; domain shift in edge semantics
- AutoGFM (ICML 2025): automated graph foundation model with adaptive architecture selection; learns shared knowledge across domains while dynamically combining modular components per task[68]
- GFM-RAG (NeurIPS 2025): graph foundation model for retrieval-augmented generation; GNN reasoning over graph structure for complex multi-hop QA[69]
- GraphBFF (Meta): first billion-parameter graph foundation model framework; KL-batching for representative multi-domain pre-training[70]

**Lecture 15B — Self-Supervised Pre-training on Graphs**
- What is the right pre-training signal for graphs? Contrastive (GraphCL), predictive (masked node/edge), generative (diffusion/flow)
- Connection to Pillar 3: graph generative models are *the pre-training objective* for graph FMs
- Scaling laws for graph models: does performance improve predictably with more parameters and more graph data?
- Zero-shot transfer across domains: what structural features are universal vs. domain-specific?
- **Key insight:** a graph FM is a generative model that has internalized a prior over graph distributions from multiple domains — it is a PFN at graph scale[70][67]

**Paper Discussion:** AutoGFM (ICML 2025); Wang et al., *Awesome-Foundation-Models-on-Graphs* survey[68][67]

***

### Week 16: LLMs as Structure Learners

**Lecture 16A — LLMs + Bayesian/Causal Structure Learning**

LLMs encode enormous amounts of relational and causal knowledge from text. Three ways this is being exploited:

**1. LLM-informed priors for BSL**[71]
LLMs are queried for domain knowledge to define an informed prior over causal graph structures, replacing uninformed uniform priors in Bayesian structure learning. The resulting LLM-informed prior improves posterior concentration and edge recovery on standard benchmarks.[71]

**2. LLM-constructed Bayesian Networks**[72]
PromptBN and ReActBN use LLMs as both structure elicitation agents (generating the DAG) and probabilistic parameterization engines (estimating CPTs from world knowledge). In zero-data settings, LLM-constructed BNs match or exceed hand-engineered models.[72]

**3. Efficient Causal Graph Discovery via LLMs**[73]
LLMs perform full causal graph discovery using a breadth-first search approach, requiring only a linear number of queries. Observational data can be incorporated to improve accuracy. LLMs serve as a proxy for human domain knowledge in causal analysis.[73]

**ECAM (2025):** a plug-and-play causal attention mechanism that integrates local causal graphs into transformer attention computation, enabling intervention and counterfactual reasoning within standard foundation models — task-agnostic and applicable to both NLP and vision.[74]

**Lecture 16B — GFlowNet Fine-tuning of LLMs**

The deepest integration: treating the LLM itself as a GFlowNet policy.

- **Amortizing intractable inference in LLMs** (ICLR 2024 Oral): fine-tune LLMs via GFlowNet objectives to sample from intractable posteriors. Chain-of-thought is treated as a latent variable model; GFlowNet training finds the posterior over reasoning chains proportional to their correctness, enabling data-efficient adaptation to multi-step reasoning and tool-use tasks.[75][16]
- **Why GFlowNet ≠ RLHF:** RLHF maximizes reward → mode collapse. GFlowNet distributional matching → posterior diversity. The structured object is the *reasoning chain* (sequence of token groups forming a logical argument)[76][16]
- **Amortizing inference in diffusion models** (NeurIPS 2024): relative trajectory balance (RTB) objective for posterior sampling in diffusion models; asymptotically correct for Bayesian inverse problems[77][78]
- **Simons Institute Lecture (Malkin, 2025)**: LLMs as entropic policies; two-stage approach (GFlowNet fine-tuning + SFT on discovered reasoning chains); synergies between amortized probabilistic inference and foundation models as grounding[79]

**The key architectural framing:**
```
LLM-as-GFlowNet:
  Prior = pretrained LLM weights (knowledge of the world)
  Posterior = GFlowNet-finetuned policy (conditioned on task)
  Structured object = reasoning chain, program, causal graph
  Reward = task correctness, causal consistency, logical validity
```

This is **Pillar 2 (GFlowNets) applied at foundation model scale** — the same mathematics, the same algorithmic invariants, but the policy is an LLM and the structured object is a reasoning chain.[16][79]

**Paper Discussion:** Hu et al. (ICLR 2024), *Amortizing Intractable Inference in LLMs*; Malkin (2025), *Amortised Inference Meets LLMs* (Simons Institute)[75][79][16]

***

### Week 17: Causal Foundation Models + Unification

**Lecture 17A — Toward Causal Foundation Models**

What would a "causal foundation model" look like? It would need to:
1. Represent distributions over SCMs (Pillar 1)
2. Perform amortized posterior inference over causal graphs from observational data (Pillar 4 + PFNs)
3. Support in-context do-calculus: given a new dataset, answer interventional and counterfactual queries in one forward pass
4. Ground its structural priors in real-world knowledge (Pillar 5 + LLMs)

**Current state of the art:**
- Do-PFN + CausalPFN: amortized effect estimation, no graph needed[49][51]
- ECAM: causal attention inside a standard transformer[74]
- LLMs + BSL: LLM-informed priors for causal discovery[71]
- NeurIPS 2024 Workshop *Causality and Large Models*: four research directions — causality *in*, *for*, *with*, and *of* large models[80]

**Lecture 17B — The Full Unification**

The complete picture of APSL at its current frontier:

| What's amortized | Mechanism | Scale | Structured object |
|---|---|---|---|
| Sampler from posterior | GFlowNet | Small-medium | Molecule, DAG, program |
| Bayesian update | PFN | Small-medium | Tabular dataset → predictions |
| Causal effect | Do-PFN/CausalPFN | Small-medium | Observational data → ATE/CATE |
| Causal graph | Amortized CSL | Medium | Observational/interventional data → DAG |
| Reasoning chain | GFlowNet-LLM | Large (billions) | Token sequences → posterior |
| Graph representation | Graph FM | Large | Multi-domain graph → downstream task |
| Structural prior | LLM + BSL | Large | Domain knowledge → prior over graphs |

**The unifying principle across all rows:** replace expensive iterative computation (MCMC, VI, brute-force search) with a single neural forward pass, trained on the space of all possible inputs from the relevant prior.[79][7][17]

**Paper Discussion:** NeurIPS 2024 *Causality and Large Models* workshop overview; Malkin (2025), Simons slides[80][79]

***

## PILLAR 6 — Research Frontiers + Projects (Week 18)

### Week 18: Open Problems + Project Presentations

**Lecture 18A — Ten Open Problems Across All Pillars**

1. **Scaling causal structure learning to 100+ variables** — state space + scoring bottleneck[81][2][1]
2. **Prior design for causal PFNs** — what SCM family gives the best test-time coverage?[51][49]
3. **Identifiability in CRL without interventions** — grouping-based approaches open but incomplete[55][56]
4. **GFlowNet mixing theory** — analogue of MCMC mixing time for specific graph classes[13][82]
5. **Counterfactual foundation models** — abduction step in high dimensions at scale[47][58]
6. **Amortization gap for causal inference** — when does PFN fail to approximate the causal posterior?[83][84]
7. **Equivariant structured generative models** — enforcing acyclicity/planarity *inside* generation[35][13]
8. **Graph FM scaling laws** — do performance gains scale predictably with data/parameters for graphs?[67][70]
9. **GFlowNet-LLM for causal reasoning** — using GFlowNet fine-tuning to induce causal consistency in LLM reasoning chains[74][16]
10. **Inference compilation for structured programs** — learning message-passing schedules as structured generative models

**Project Presentations** (rest of Week 18):
Each presentation addresses the APSL quadruple: *What structured object? What data structure? What algorithm? What statistical/causal guarantee?*

***

## Complete Reading Stack

### Core Texts
- Koller & Friedman, *Probabilistic Graphical Models* (2009) — Pillars 1–2[3]
- Bengio et al., *GFlowNet Foundations* monograph (2023) — Pillars 2–3[11]
- Pearl, *Causality* (2009 ed.) — Pillar 4
- GFlowNet Tutorial (Milayb Notion) — Pillar 2 supplement[85]
- Awesome-GFlowNets GitHub — paper feed[86]
- Awesome-Foundation-Models-on-Graphs GitHub — Pillar 5 paper feed[67]
- Awesome-PFNs GitHub — Pillar 2/5 paper feed[87]

### Papers by Pillar

**Pillar 2 (Inference)**
- Malkin et al. (2022), *GFlowNets and VI*[9]
- Müller et al. (2022), *Transformers Can Do Bayesian Inference*[17]
- Hollmann et al. (2022/2025), *TabPFN v1/v2*[19][5]

**Pillar 3 (Generative Models)**
- You et al. (2018), *GraphRNN*[26]
- Deleu et al. (UAI 2022), *Bayesian SL with GFlowNets*[2]
- Gat et al. (2024), *Discrete Flow Matching*[33]
- Sun et al. (2023), *DIFUSCO*[38]
- Lahlou et al. (ICML 2023), *Continuous GFlowNets*[88]
- CPDAG-GFN (ICLR 2025)[1]
- Energy Matching (2025)[37]

**Pillar 4 (Causal ML)**
- Pawlowski et al. (NeurIPS 2020), *Deep SCMs*[30]
- Lorch et al. (NeurIPS 2022), *Amortized Causal SL*[45]
- Do-PFN (NeurIPS 2025)[49]
- CausalPFN (NeurIPS 2025)[51]
- Ke et al. (2023), *CSIvA*[83]
- Varıcı et al. (2025), *CRL Identifiability*[54]
- GIMs (ICML 2025)[57]
- Meta-Causal Graph (NeurIPS 2025)[59]

**Pillar 5 (Foundation Models)**
- Hu et al. (ICLR 2024 Oral), *Amortizing Intractable Inference in LLMs*[16][75]
- Malkin (2025), *Amortised Inference Meets LLMs* — Simons Institute slides[79]
- AutoGFM (ICML 2025)[68]
- GFM-RAG (NeurIPS 2025)[69]
- Jiralerspong et al., *Efficient Causal Graph Discovery with LLMs*[73]
- LLM-improved Bayesian Structure Learning (2025)[71]
- LLM-constructed Bayesian Networks (2025)[72]
- ECAM (2025)[74]
- NeurIPS 2024, *Causality and Large Models* workshop[80]
- ICML FMSD 2025 workshop overview[61][65]

***

## Assessment Structure (18-week)

| Component | Weight | Description |
|---|---|---|
| Weekly paper reviews (16) | 20% | Structured critique: object · data structure · mechanism · guarantee · gap |
| PS1 (Pillars 1–2) | 8% | Implement BP + VE; derive PAC-Bayes bound for graph-structured class |
| PS2 (Pillar 2–3) | 8% | Implement toy GFlowNet; prove flow conservation → correct marginals |
| PS3 (Pillars 3–4) | 7% | Implement one of: discrete FM on sequences / GFlowNet for small causal graphs / minimal DSCM |
| PS4 (Pillar 5) | 7% | Run TabPFN v2 + Do-PFN on a benchmark; analyze the SCM prior design; compare to GFlowNet-based structure learning on same dataset |
| Paper presentation (Week 17) | 10% | Present one paper with APSL quadruple critique |
| Final project | 40% | Research contribution (see categories below) |

### Final Project Categories

| Type | Description |
|---|---|
| **New method** | New mechanism × structure combination from the intersection matrix |
| **Theoretical analysis** | Sample complexity / convergence for existing method |
| **PFN design** | New prior over structured objects → amortized inference machine |
| **Graph FM** | Pre-training + adaptation of a graph foundation model on a new domain |
| **GFlowNet-LLM** | Apply GFlowNet fine-tuning to a structured reasoning task beyond existing work |
| **CRL application** | CRL method for a new data modality |
| **Causal FM design** | Architecture for a foundation model supporting interventional + counterfactual queries |
| **Unification** | Formal equivalence between two existing methods |
| **Benchmark** | Evaluation of existing methods on a new structured domain |

***

## Self-Study Track (18-Week)

| Phase | Weeks | Focus | Deliverable |
|---|---|---|---|
| **1: Language** | 1–3 | PGMs[3] + BNP + SCM basics (Pearl Ch. 1–2) | Implement BP on a Bayes net |
| **2: Inference Engine** | 4–7 | GFlowNet Foundations[11] + VI[9] + PFN[17] | Toy GFlowNet; run TabPFN v2[19] |
| **3: Generative Zoo** | 8–11 | 10 core Pillar 3 papers | Reproduce one structured generative model result |
| **4: Causal ML** | 12–13 | DSCMs[30] + Do-PFN[49] + CausalPFN[51] + CRL[54] | Run CausalPFN on a benchmark; replicate one key result |
| **5: Foundation Models** | 14–17 | GFlowNet-LLM[16] + Graph FMs[68][69] + LLM+BSL[71] + ECAM[74] | GFlowNet fine-tune a small LM on a structured reasoning task |
| **6: Research** | 18 | Identify one open problem; survey adjacent 5 papers | 4-page research proposal with: object, mechanism, data structure, theory gap, proposed contribution |

**The universal reading lens (your APSL quadruple) for every paper:**
1. What is the structured object?
2. What data structure represents it?
3. What algorithm searches/samples over it?
4. What is the statistical / causal guarantee?

At Pillar 5, add a 5th question: *What is the pre-training distribution, and what does it implicitly say about the prior over structured objects?*
