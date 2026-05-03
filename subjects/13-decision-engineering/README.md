# Decision Engineering
### *A Data Science for the Modern AI Era*

---

## The Proposal

Data Science, as it was defined and practiced, was built for a world of dashboards, predictions, and post-hoc analysis. It gave organizations the ability to see — to describe the past, detect patterns, forecast the near future. That was enough when the bottleneck was data and compute.

That world is over.

The bottleneck now is not prediction — it is *judgment*. Not "what will happen?" but "what should we do?" Not correlation but causation. Not a model that fits the data but a system that can reason about interventions, handle distribution shift, and operate reliably in novel environments. The models exist. The infrastructure exists. What's missing is the engineering discipline for building systems that actually make decisions well — under uncertainty, at scale, in the real world.

**Decision Engineering** is that discipline. It is the practice of designing, building, evaluating, and improving decision-making systems using the full stack of modern AI — causal inference, reinforcement learning, probabilistic reasoning, large language models, and human-AI interaction — grounded in the rigor of engineering and the depth of decision theory.

It is not a new name for data science. It is a reorientation: from prediction to action, from correlation to causation, from fitting the past to shaping the future.

---

## The Gap It Fills

| Classical Data Science | Decision Engineering |
|------------------------|----------------------|
| Predict what will happen | Determine what to do |
| Correlational models | Causal models + interventions |
| Offline, batch analysis | Online, adaptive systems |
| Maximize accuracy on held-out data | Maximize real-world outcomes under deployment |
| Human interprets output | System recommends or acts |
| Descriptive + predictive | Prescriptive + generative |
| Model the past | Reason about counterfactuals |

---

## Core Pillars

**1. Causal Reasoning as Foundation**
Decisions require understanding interventions, not just observations. Decision engineering is built on Pearl's causal hierarchy — not because causality is fashionable, but because without it, systems break the moment the environment changes.

**2. Decision Theory + Uncertainty**
A decision system that can't represent and communicate uncertainty is not safe to use. Expected utility, prospect theory, robust optimization, conformal prediction — these are engineering requirements, not academic luxuries.

**3. Human-AI Decision Loops**
Most consequential decisions involve humans. Decision engineering designs the interface: when does the system act, when does it defer, how does it explain itself, how does it fail safely? This is a design discipline as much as a technical one.

**4. Adaptive and Online Systems**
The world changes. A decision system that can only operate on the distribution it was trained on is brittle. Continual learning, domain adaptation, and distribution shift detection are first-class engineering concerns.

**5. Evaluation as a First-Class Problem**
How do you know a decision system is good? Not accuracy on a test set — but real-world outcomes, counterfactual fairness, calibration under shift. Offline evaluation of online decision systems is one of the hardest open problems in the field.

**6. LLMs as Decision Components**
Modern large models are increasingly embedded in decision pipelines — as reasoners, planners, evaluators, and communicators. Decision engineering asks: what are they good for, where do they fail, and how do you build around their failure modes?

---

## Core Claims to Investigate

These are propositions this subject exists to test, refine, and build:

- [ ] *Causal inference is not just for academic studies — it is a practical engineering tool for improving deployed decision systems.*
- [ ] *LLMs can be reliable components in decision pipelines if their role is correctly scoped and their failure modes are explicitly engineered around.*
- [ ] *The right way to evaluate a decision system is not accuracy but regret — how much worse is it than the optimal policy it could have followed?*
- [ ] *Human-AI decision systems fail not at the technical level but at the interface level — where authority is ambiguous and accountability is diffuse.*
- [ ] *Reinforcement learning, properly constrained, is the correct framework for sequential decision problems that current ML practice solves incorrectly with supervised learning.*
- [ ] *Distribution shift is not a bug to patch but a signal — it tells you something about your causal model that you got wrong.*

---

## The Research Agenda

### Near-term questions
- What does a "decision engineering stack" look like? What are the primitive components?
- How do you scope a decision problem correctly before building anything?
- What is the right evaluation framework for a human-AI decision loop?
- Where does causal inference add genuine value over well-calibrated prediction?

### Medium-term questions
- Can LLMs serve as causal reasoners — and under what conditions does that break?
- What does a decision system that learns continually in production look like?
- How do you build a decision system that knows when to defer to a human?

### Long-term questions
- Can decision engineering become a recognizable discipline with its own methods, tools, and training pipelines?
- What does a "decision engineering" role look like in an AI-native organization?

---

## Foundational Resources

**Causal and Decision Theory**
- [ ] **Causality — Judea Pearl** *(Ch. 1, 3, 4, 7 — the SCM framework)*
- [ ] **The Book of Why — Pearl & Mackenzie** *(readable foundation)*
- [ ] **Decision Theory: Principles and Approaches — Parmigiani & Inoue** *(Ch. 1–4)*
- [ ] **Reinforcement Learning: An Introduction — Sutton & Barto** *(Ch. 1–3, 13)*

**Human-AI Interaction and Systems**
- [ ] **Designing Human-AI Interaction — various** *(selected papers)*
- [ ] **Explainability and Interpretability — DARPA XAI papers** *(selected)*

**Uncertainty and Evaluation**
- [ ] **Probabilistic Machine Learning — Kevin Murphy** *(Vol. 1: Ch. 4–5 on uncertainty)*
- [ ] **Conformal Prediction: A Gentle Introduction — Angelopoulos & Bates**

**LLMs as Decision Components**
- [ ] **Language Agents: From Next-Token Prediction to Digital Automation — selected survey**
- [ ] **Calibration of Large Language Models Using Their Generations — Kadavath et al.**

**Economics and Strategy**
- [ ] **Thinking Strategically — Dixit & Nalebuff** *(game theory for decisions)*
- [ ] **Poor Charlie's Almanack — Munger** *(Talk 7: Psychology of Misjudgment)*

---

## Experiment Log

| # | Title | Type | Status | Dates |
|---|-------|------|--------|-------|
| — | — | — | — | — |

---

## Working Definition

> **Decision Engineering** is the discipline of designing systems that make good decisions — under uncertainty, in changing environments, involving both humans and machines — using the full stack of causal inference, probabilistic reasoning, reinforcement learning, and modern AI, evaluated not by prediction accuracy but by real-world outcomes.

*This definition is a hypothesis. It will be revised by the experiments in this subject.*
