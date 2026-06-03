# Circuits

**One-liner:** A subgraph of model features/components (heads, MLP neurons, SAE features) wired by weights that implements a human-interpretable algorithm — the unit of mechanistic explanation for *how* a model computes an output, not just *what* it outputs.

## The key insight

A circuit is a sparse, end-to-end causal path through the network that, in isolation, reproduces a behavior. Components communicate by reading from and writing to the **residual stream** (the linear "bus"); weights between components define the edges. To validate a circuit you do causal surgery:

- **Activation patching / causal tracing:** replace a component's activation with one from a counterfactual run; if the metric flips, the component is causally involved.
- **Path patching:** patch a specific *edge* (head A → head B's query) to isolate which connection carries the signal.
- **Faithfulness check:** does running *only* the circuit (ablate everything else) recover the behavior? Does ablating the circuit destroy it?

The output: a labeled graph like "S-Inhibition heads → Name Mover heads → logit," with roles assigned to each node.

## Where it appears

- **IOI (Indirect Object Identification), Wang et al. 2022** — the canonical circuit: ~26 GPT-2 heads in classes (duplicate-token, S-inhibition, name-mover, backup) implementing "John gave a drink to Mary → John".
- **Induction heads, Olsson et al. 2022** — a 2-head circuit (previous-token head → induction head) doing in-context `[A][B]...[A]→[B]`; tied to the ICL phase change.
- **Anthropic attribution graphs (2025)** — transcoders + cross-layer features scale circuit discovery to production LLMs (Claude), tracing multi-step reasoning, planning, and refusal circuits.
- **EAP / attribution patching** — gradient-based approximation to find circuits at scale without per-edge ablation.

## Common mistake

Confusing correlation with causation: a component that *activates* on a behavior isn't necessarily part of the circuit. Only intervention (patching/ablation) establishes causal membership. Also: circuits are not unique or clean — backup heads and self-repair mean ablating one component is often silently compensated for, inflating apparent robustness.

## See also
- [[induction-heads]] — the foundational, fully reverse-engineered circuit
- [[activation-patching-causal-tracing]] — the core method for validating circuit membership
- [[residual-stream]] — the shared linear bus circuits read from and write to
