# Lecture Myself
### *A Self-Lectured Curriculum*

**FAIRE context:** → [Program Handbook](../faire-program.md) · Capstone for all 5 subjects — write the lecture to prove you can teach it

---

## The Idea

Professors learn by teaching. The act of preparing a lecture — writing out the derivation, finding the right example, designing the exercise that breaks the wrong intuition — forces understanding that passive reading never does.

This is that, applied to yourself.

For every topic in this curriculum, you are the professor. You write the lecture notes. You build the examples. You design the problem sets. You create the visual that makes the hard thing click. And then you ask: *can I explain this clearly to someone who doesn't already know it?* If the answer is no, the lecture isn't done.

This is the Feynman technique at university scale — not a summary, not flashcards, but actual teaching material: the kind a rigorous professor would hand to students.

---

## The Method

**Step 1 — Study the topic** *(from the seminars, papers, or textbooks)*
Understand it first. Don't start writing until you can hold the idea in your head.

**Step 2 — Write the lecture**
Explain it from scratch, in your own words, at the level of a strong upper-division undergrad. No hiding behind notation. No "it can be shown that." Every claim earns its place.

**Step 3 — Build the examples**
Find or construct 2–3 worked examples that make the abstract concrete. The best example is one that could break a naive understanding — something surprising.

**Step 4 — Write the exercises**
Design 3–5 problems. At least one should be easy (checks basic comprehension), one medium (applies the idea), one hard (breaks a wrong mental model or extends to a new case).

**Step 5 — The Feynman Check**
Read back what you wrote. Would a sharp student who has never seen this understand it? If not — find the gap, go back to the source, and rewrite.

**Step 6 — Teaching aids**
A diagram, an analogy, a visual, a code snippet. Something that makes the idea exist in a different modality than prose.

---

## Lecture Format

Each lecture lives in its own folder:
```
lecture-myself/
└── 01-ai/
    └── 01-llm-foundations/
        ├── lecture.md       ← the actual lecture notes
        ├── examples.md      ← worked examples
        ├── exercises.md     ← problem set with solutions
        └── aids/            ← diagrams, code, visuals
```

### `lecture.md` structure
```
# Lecture: [Topic]

## Motivation
[Why does this matter? What problem does it solve?]

## Prerequisites
[What should the student already know?]

## The Core Idea
[The load-bearing concept, explained clearly]

## The Formalism
[The math/notation, derived not stated]

## Intuition
[The mental model that makes the formalism make sense]

## Where It Breaks
[The failure modes, edge cases, known limits]

## Connections
[How this connects to other topics in the curriculum]

## Feynman Check
[The one-sentence version. If you can't write this, the lecture isn't done.]
```

---

## 10 Tracks

| # | Track | Lectures | Status |
|---|-------|----------|--------|
| 01 | [AI — LLMs, Agents, Interpretability](01-ai/README.md) | 0 | — |
| 02 | [Generative Modeling](02-generative-modeling/README.md) | 0 | — |
| 03 | [Representation Learning](03-representation-learning/README.md) | 0 | — |
| 04 | [Neural Networks & Deep Learning](04-neural-networks-and-deep-learning/README.md) | 0 | — |
| 05 | [Statistical & Probabilistic ML](05-statistical-and-probabilistic-ml/README.md) | 0 | — |
| 06 | [Reinforcement Learning](06-reinforcement-learning/README.md) | 0 | — |
| 07 | [Attention, Memory & Reasoning](07-attention-memory-reasoning/README.md) | 0 | — |
| 08 | [Causal Inference](08-causal-inference/README.md) | 0 | — |
| 09 | [Algorithms & Systems](09-algorithms-and-systems/README.md) | 0 | — |
| 10 | [Complexity & Cognition](10-complexity-and-cognition/README.md) | 0 | — |

---

## The Standard

A lecture here is not a summary. It is not a reading note. It is teaching material — something you could actually give to a student. If it reads like bullet points scraped from a paper, it is not done.

The test: *could a strong student who hasn't read the source paper learn the idea from this lecture alone?*
