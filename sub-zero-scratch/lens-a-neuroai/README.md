# Lens A — NeuroAI

**Type:** Lens
**Serves:** Deepens continual learning and world models with a rarer substrate — the biological implementation of the things AI is trying to do.

---

## Why This Lens

Most deep learning researchers learn biology as analogy. NeuroAI treats it as constraint and inspiration: the brain is the existence proof that intelligence, continual learning, and world modeling are possible. Understanding *how* it does them — at the computational level, not just the metaphor level — gives you a different vocabulary for thinking about what modern AI is and isn't doing.

The rarity is part of the value. Most AI researchers don't go here.

---

## What "From Scratch" Means Here
- Start with no assumption that biology directly maps to AI
- Build implementations of the key models: a predictive coding network, a Helmholtz machine, a simple active inference agent
- Test the claims: does the brain actually do this? Does the model capture what matters?

---

## Core Framework

### Complementary Learning Systems (CLS) Theory
The brain uses two systems: hippocampus for fast, specific memorization; neocortex for slow, distributed generalization. This is the biological solution to catastrophic forgetting — and it directly inspires replay-based continual learning.

**The claim to test:** Is the hippocampus-neocortex split a good architectural prior for artificial continual learning systems?

### Predictive Coding
The brain is not a bottom-up feature detector. It is a top-down prediction machine that only propagates *errors* — the difference between what was predicted and what arrived. Perception is inference.

**The claim to test:** Does a predictive coding network learn more efficiently than a standard feedforward network on distribution-shifted data?

### Active Inference (Free Energy Principle)
Karl Friston's framework: the brain minimizes "free energy" — a bound on surprise. Perception, action, and learning are all unified under one principle. Controversial but worth understanding deeply.

**The claim to test:** Can a simple active inference agent solve a control task that a standard RL agent can also solve — and do the failure modes differ?

---

## Build Progression

- [ ] **Step 1:** Implement a simple Helmholtz machine (generative + recognition networks) — the conceptual ancestor of VAEs
- [ ] **Step 2:** Implement a predictive coding network on a toy task. Visualize what gets propagated vs. what is suppressed.
- [ ] **Step 3:** Read CLS theory. Build a two-system continual learning experiment: fast learner (hippocampus proxy) + slow learner (neocortex proxy) with replay.
- [ ] **Step 4:** Build a minimal active inference agent on a grid world. Compare to PPO on the same task.
- [ ] **Step 5:** Write a technical note: "What NeuroAI tells us about continual learning that the AI literature doesn't"

---

## Essential Resources

- [ ] **Theoretical Neuroscience — Dayan & Abbott**
  - Ch. 1–2: Neural encoding and decoding
  - Ch. 6: Model neurons (rate coding, spiking)
  - Ch. 10: Reinforcement learning (the biological version)

- [ ] **Neuromatch Academy — Computational Neuroscience track**
  - W1D1–W1D4: Neural data, Generalized Linear Models
  - W2D4: Optimal control and active inference
  - W3D2: Continual learning and memory

- [ ] **Complementary Learning Systems — McClelland, McNaughton, O'Reilly (1995)**
  - Full paper — the original CLS theory

- [ ] **Predictive Coding — Rao & Ballard (1999)**
  - Full paper — hierarchical predictive coding in the visual cortex

- [ ] **The Free-Energy Principle — Karl Friston (2010)**
  - §1–3: The variational free energy framework
  - §5: Active inference

- [ ] **A tutorial on active inference — Parr, Pezzulo, Friston**
  - Ch. 2–4: Generative models, free energy, action

---

## Build Log

| # | What I Built | Claim Tested | Verdict | Link |
|---|-------------|-------------|---------|------|
| — | — | — | — | — |
