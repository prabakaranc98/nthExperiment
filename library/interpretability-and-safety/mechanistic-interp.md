# Mechanistic Interpretability

*Reverse-engineering what neural networks actually compute.*

---

## The question

A neural network is a mathematical function: inputs in, millions of parameters in the middle, outputs out. We know it works. We don't know *what it's doing* internally.

Mechanistic interpretability asks: **what algorithm is the network implementing?**

Not "what does it do on inputs?" (that's behavioral evaluation) but "what computation happens inside?" — which circuit implements which capability, which features live where, and how information flows.

---

## Superposition: more features than neurons

A neuron could, in principle, represent one feature. In practice, models represent *far more features than they have neurons*. How?

**Superposition:** features are stored as *directions* in activation space, not as individual neurons. Many features share the same neurons by occupying different linear combinations.

- **Analogy:** multiple signals on one radio channel at different frequencies — they overlap but separate cleanly.
- **Evidence:** individual neurons respond to multiple unrelated concepts. One neuron might fire for "banana," "yellow things," *and* "curved objects." It is not "the banana neuron" — it participates in representing several features at once.

This is why reading off single neurons fails, and why we need methods that recover the underlying directions.

---

## Circuits — the mechanistic view

A **circuit** is a specific computational subgraph: a set of neurons / attention heads and the connections between them that together implement an algorithm.

**Induction heads** — the first major discovered circuit:

A two-layer attention mechanism that implements "copy from previous context." Given a sequence containing `[A][B] ... [A]`, an induction head predicts `[B]` after the second `[A]`.

This is a core mechanism behind **in-context learning**: the model finds a pattern in the context and continues it. It is not "knowing" the answer — it is running a pattern-matching algorithm over the prompt.

---

## Activation patching — probing causality

How do you know a circuit is *causally* responsible for a behavior, not merely correlated?

**Activation patching:**

1. Run the model on two inputs (e.g., "The Eiffel Tower is in Paris" vs. "The Eiffel Tower is in Rome").
2. Identify where the predictions diverge.
3. **Patch:** replace activations from input A with those from input B at specific layers/positions.
4. Observe whether the output changes.

If patching at a location flips the output, that location is causally responsible. Related variants — attribution patching, path patching — trade exactness for the speed needed to scan many components.

---

## What we've found so far

| Circuit | What it does |
|---------|--------------|
| Induction heads | Pattern matching / in-context learning |
| Name-mover heads | Copying tokens from context |
| Negative name-mover heads | Suppressing certain tokens |
| IOI circuit | Indirect-object identification in sentences |
| Docstring circuit | Completing function documentation |

These are concrete wins, but they cover only a tiny fraction of what frontier models compute.

---

## The key limitation

Circuit analysis has mostly been done on small models or narrow tasks. Scaling to full frontier models is the open challenge. **Sparse Autoencoders (SAEs)** are the main tool aimed at it: they decompose activations into many sparse, monosemantic features, partly undoing superposition. By 2025-2026 this has matured into a standard workflow — train an SAE (or transcoder) on activations, label features at scale, then trace **attribution graphs** across features to recover end-to-end circuits in production-scale models.

See the [concept-library index](../bricks/README.md) for related entries on SAEs and feature steering.

---

## Why this matters for safety

If you understand the algorithm a model is running, you can:

- **Monitor** — detect when it is about to do something harmful.
- **Steer** — intervene on specific features or circuits to change behavior.
- **Verify** — check that alignment training actually changed the right internal computations, not just surface outputs.

Mechanistic interpretability is the scientific foundation for trustworthy AI: systems that are not only behaviorally well-tested, but *internally* understood.
