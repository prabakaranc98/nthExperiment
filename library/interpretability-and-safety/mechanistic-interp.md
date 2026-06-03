# Mechanistic Interpretability

*Reverse-engineering what neural networks actually compute.*

---

## The question

A neural network is a mathematical function. It takes inputs, processes them through millions of parameters, and produces outputs. We know it works. We don't know *what it's doing* internally.

Mechanistic interpretability asks: **what algorithm is the network implementing?**

Not "what does it do on inputs?" (that's behavioral evaluation) but "what computation is happening inside?" — which circuit implements which capability, which features are stored where, and how information flows.

---

## The superposition hypothesis

A neuron in a neural network could, in principle, represent one feature. But in practice, models represent *far more features than they have neurons*. How?

**Superposition:** features are stored as *directions* in activation space, not individual neurons. Many features can share the same neurons by being represented in different linear combinations.

Analogy: like storing multiple signals on one radio channel using different frequencies — they overlap but can be separated.

**Evidence:** when you look at what activates individual neurons, they respond to multiple unrelated concepts. A neuron might respond to both "banana" and "yellow things" and "curved objects." It's not "the banana neuron" — it's participating in representing multiple features simultaneously.

---

## Circuits — the mechanistic view

A **circuit** is a specific computational subgraph of a neural network: a set of neurons/attention heads and the connections between them that together implement a specific algorithm.

**The induction head** — the first major discovered circuit:

An induction head is a two-layer attention mechanism that implements "copy previous context." If the sequence contains [A][B]...[A], an induction head learns to predict [B] after the second [A].

This is the mechanism behind **in-context learning**: the model learned to find patterns in the context and continue them. It's not "knowing" the answer — it's running a pattern-matching algorithm over the context.

---

## Activation patching — probing causality

How do you know if a circuit is *causally* responsible for a behavior, not just correlated?

**Activation patching:**
1. Run the model on two inputs (e.g., "The Eiffel Tower is in Paris" and "The Eiffel Tower is in London")
2. Identify where the model's predictions diverge
3. **Patch**: replace activations from input A with activations from input B at specific layers/positions
4. Observe whether the prediction changes

If patching activations at a specific location changes the output, that location is causally responsible.

---

## What we've found so far

| Circuit | What it does |
|---------|-------------|
| Induction heads | Pattern matching / in-context learning |
| Name-mover heads | Copying tokens from context |
| Negative name-mover heads | Suppressing certain tokens |
| IOI circuit | Indirect object identification in sentences |
| Docstring circuit | Completing function documentation |

We've found specific circuits, but we've characterized only a tiny fraction of what frontier models compute.

---

## The key limitation

Circuits work are mostly done on small models or specific narrow tasks. Scaling to full frontier models is the open challenge. This is what **Sparse Autoencoders (SAEs)** are designed to address.

---

## Why this matters for safety

If you understand the algorithm a model is running, you can:
- Detect when it's about to do something harmful (monitoring)
- Intervene to change specific behaviors (steering)
- Verify that alignment training actually changed the right computations

Mechanistic interpretability is the scientific foundation for making AI systems trustworthy — not just behaviorally well-tested, but *internally* understood.
