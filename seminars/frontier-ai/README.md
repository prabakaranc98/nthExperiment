# Frontier-AI Curriculum
## 16 Topics · 327 Papers

**Pace:** 2 papers/day → ~164 days. Compress to 3/day on topics you know well.
**Order:** Convergent — foundations first, then representation/generation, then the LLM frontier, then RL, causality, systems, perspective. Topics 11–15 are the frontier-systems layer (2023–25 heavy); Topic 16 is the architecture field guide threading under Topics 1, 4, 5, 6.

→ **[How to Read a Paper](../how-to-read.md)** — 8 angles, sequencing guide, three exits
→ **[FAIRE Program Handbook](../../faire-program.md)** — how this curriculum maps to FAIRE 501–505

---

## The Minimal Artifact Rule

**Every paper produces one artifact. No exceptions.**

The artifact is a **Cornell Note + One-Page Synthesis** — filed in the paper's topic folder, publishable as a log entry at [pracha.me/frontier](https://pracha.me/frontier/foundational-models/).

Template: `_templates/cornell-note.md`

```
seminars/frontier-ai/01-science-of-deep-learning/
└── notes/
    └── 01-resnet-he-2015.md       ← Cornell Note + One-Page Synthesis
    └── 02-adam-kingma-2014.md
    ...
```

**The five fields that make it publishable:**
- **The Claim** — what the paper asserts
- **The Mechanism** — how it works
- **The Key Result** — the number or finding that matters
- **Where It Breaks** — known failure modes and limits
- **Why It Matters** — what it enables or changes

---

## The Full Loop (per paper)

1. Read for the **one load-bearing idea**
2. Write the **Cornell Note + One-Page Synthesis** ← *mandatory*
3. Flag **Anki cards** from the note
4. When a cluster clicks, write a **blog post** connecting them
5. Spin up the **smallest experiment** that tests the central claim

Reading is input. The note is the minimum output. Everything else is optional but welcome.

---

## Seminar Format — Role-Playing Paper Reading

*Inspired by Colin Raffel's role-playing seminars.*

For each paper, one person takes a role. Roles:

| Role | What You Do |
|------|-------------|
| **Author** | Short, high-quality conference-style presentation of main contributions |
| **Archaeologist** | Trace the prior work — what papers does this build on? What did it displace? |
| **Researcher** | Strengths, what it opens up, follow-on work it enables |
| **Industry Practitioner** | Real-world applications, deployment considerations, what practitioners actually use |
| **Skeptic** | Limitations, what's overclaimed, where it breaks, what the authors didn't test |
| **Hacker** | Implement the core idea — the smallest reproduction that captures the mechanism |
| **Wild Card ×2** | Choose any of the above; coordinate so the two Wild Cards don't duplicate |

**If not presenting:** read both papers, submit ≥2 in-depth questions or observations before the session. Discussion is seeded from these.

**Synthesis note format:**
```
Paper: [title]
Role: [your role]
Load-bearing idea: [one sentence]
Mechanism: [how it works]
Statistical/optimization principle: [what makes it work theoretically]
Where it breaks: [known failure modes, limitations]
What it opens: [follow-on questions]
```

---

## Topic Index

| # | Topic | Papers | Folder |
|---|-------|--------|--------|
| 01 | The Science of Deep Learning | 20 | [→](01-science-of-deep-learning/README.md) |
| 02 | Statistical Foundations & Statistics for LLMs | 20 | [→](02-statistical-foundations/README.md) |
| 03 | Representation Learning | 20 | [→](03-representation-learning/README.md) |
| 04 | Generative Modeling | 20 | [→](04-generative-modeling/README.md) |
| 05 | Attention, Memory, Reasoning, Sequence Models | 20 | [→](05-attention-memory-reasoning/README.md) |
| 06 | AI — LLMs, Multimodal, Agents, Interpretability | 20 | [→](06-llms-multimodal-agents-interpretability/README.md) |
| 07 | Reinforcement Learning | 20 | [→](07-reinforcement-learning/README.md) |
| 08 | Causal & Statistical Inference | 20 | [→](08-causal-statistical-inference/README.md) |
| 09 | Algorithms & Systems for AI | 20 | [→](09-algorithms-and-systems/README.md) |
| 10 | Complexity, Cognition & First Principles | 20 | [→](10-complexity-cognition-first-principles/README.md) |
| 11 | Data, Tokenization, Benchmarking & Training at Scale | 25 | [→](11-data-tokenization-benchmarking/README.md) |
| 12 | Post-Training | 20 | [→](12-post-training/README.md) |
| 13 | Agents — Capabilities & Systems | 22 | [→](13-agents-capabilities-systems/README.md) |
| 14 | Reasoning & Inference-Time Compute | 20 | [→](14-reasoning-inference-time-compute/README.md) |
| 15 | Alignment, Evaluation & Safety | 20 | [→](15-alignment-evaluation-safety/README.md) |
| 16 | Architectures — From CNNs to Hybrids | 20 | [→](16-architectures-field-guide/README.md) |

**Total: 327 papers**

---

## Progress

| Topic | Done | Total | Blog | Experiments |
|-------|------|-------|------|-------------|
| 01 Science of DL | 0 | 20 | — | — |
| 02 Statistical Foundations | 0 | 20 | — | — |
| 03 Representation Learning | 0 | 20 | — | — |
| 04 Generative Modeling | 0 | 20 | — | — |
| 05 Attention, Memory, Reasoning | 0 | 20 | — | — |
| 06 LLMs, Multimodal, Agents | 0 | 20 | — | — |
| 07 Reinforcement Learning | 0 | 20 | — | — |
| 08 Causal & Statistical Inference | 0 | 20 | — | — |
| 09 Algorithms & Systems | 0 | 20 | — | — |
| 10 Complexity, Cognition | 0 | 20 | — | — |
| 11 Data, Tokenization, Benchmarking | 0 | 25 | — | — |
| 12 Post-Training | 0 | 20 | — | — |
| 13 Agents | 0 | 22 | — | — |
| 14 Reasoning & Inference-Time | 0 | 20 | — | — |
| 15 Alignment, Evaluation, Safety | 0 | 20 | — | — |
| 16 Architectures | 0 | 20 | — | — |
| **Total** | **0** | **327** | | |
