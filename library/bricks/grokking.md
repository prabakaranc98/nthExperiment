# Grokking

**One-liner:** A delayed generalization phenomenon where test accuracy jumps from chance to near-perfect long after training accuracy has saturated at 100% — a sharp phase transition driven by weight decay slowly replacing a memorizing solution with a generalizing circuit.

## The phenomenon (Power et al., 2022)

Train a small transformer/MLP on an algorithmic task (e.g. modular addition a+b mod p) with weight decay:

- Step ~10³: train acc → 100%, test acc ≈ chance. Network has **memorized**.
- Step ~10⁵: test acc suddenly → ~100%. Network has **grokked** — found the general rule.

The gap between memorization and generalization can be 100-1000× in steps. Generalization is not gradual; it is a phase transition.

## The mechanism

Two competing solutions exist for the data: a high-norm **memorizing** solution (fits train, fails test) and a low-norm **generalizing** circuit. Training cross-entropy is driven to ~0 quickly by memorization, after which the dominant remaining gradient signal is **weight decay** (the λ‖θ‖² term). Weight decay slowly pushes the network down the loss-equal manifold toward the lower-norm generalizing solution.

For modular arithmetic the grokked circuit is interpretable (Nanda et al., "Progress measures", 2023): embeddings become Fourier features, and the network computes via trig identities — cos/sin(ωₖa)·cos/sin(ωₖb) combined to read off (a+b) mod p. Progress measures (e.g. restricted loss, Gini of Fourier power) rise smoothly before the test-accuracy jump.

## Where it appears

- **Power et al. (2022)** — original; algorithmic datasets, the canonical grok curve.
- **Nanda et al. (2023)** — mechanistic interpretability: reverse-engineered the modular-addition circuit; introduced continuous progress measures showing the "phase change" is the cleanup of memorizing components.
- **Liu et al. (2022/23)** — grokking explained via the **L2 norm**; "Omnigrok" shows grokking induced/removed by tuning weight norm at init, framing it as a competition between norm and fitting.
- **Varma et al. (2023)** — efficiency argument: generalizing circuit has higher per-parameter "efficiency"; weight decay selects it once train loss is saturated.

## Common mistake

Believing grokking requires weird/algorithmic data and is unrelated to real training. The core ingredient is **regularization (weight decay) continuing to act after the loss is ~0**, which holds broadly. The deeper error is thinking the network "does nothing" during the plateau — the generalizing circuit is forming the whole time (visible in progress measures), it just isn't yet dominant enough to flip top-1 test accuracy.

## See also
- [[double-descent]] — both are non-monotone, over-parameterization-driven generalization surprises
- [[implicit-bias]] — grokking is implicit bias of weight decay toward low-norm/min-complexity solutions
- [[sparse-autoencoders]] — the interpretability lens used to extract the grokked circuit
