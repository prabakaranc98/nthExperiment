# Frontdoor Criterion & Mediation Analysis

**One-liner:** Identify P(Y | do(X)) by routing the effect through a fully-mediating variable M — using M's own (unconfounded) front-door path — when an unobserved confounder U blocks every valid backdoor adjustment set.

## The formula / definition

The canonical front-door graph: X → M → Y, with a hidden confounder U → X and U → Y. Backdoor fails (U is unobserved), but M satisfies the **front-door criterion** if (1) M intercepts every directed path X→Y, (2) no unblocked backdoor path from X to M, and (3) every backdoor path from M to Y is blocked by X. Then the effect is identified by chaining two adjustments:

P(Y | do(x)) = Σ_m P(m | x) · Σ_{x'} P(y | x', m) P(x')

Read it as: estimate X→M (clean, no U confounding into M), estimate M→Y adjusting for X (X blocks the U backdoor M←X←U→Y), then compose. Mechanically it is two backdoor applications glued by do-calculus rule 2.

**Mediation decomposition (Pearl, nested counterfactuals):** Total Effect = Natural Direct Effect + Natural Indirect Effect. NDE = E[Y_{x, M_{x*}}] − E[Y_{x*}] (vary X, hold M at its baseline-X value); NIE = E[Y_{x, M_x}] − E[Y_{x, M_{x*}}] (hold X, shift M along X). The mediation formula requires no unmeasured X–M, X–Y, or M–Y confounding — strictly stronger than front-door's hidden-U-into-X tolerance.

## Where it appears

- **DoWhy / Ananke / `causaleffect`** — the ID algorithm returns the front-door estimand automatically when backdoor is non-identifiable but M mediates fully
- **Baron–Kenny vs. modern mediation** — the linear product-of-coefficients a·b is the front-door estimand under linearity; counterfactual NDE/NIE generalize it to nonlinear / interaction settings
- **Activation patching & causal tracing in LLMs** — patching a mediating component (a head, an MLP, the residual at a layer) to measure its indirect effect on the output is empirical mediation analysis on the compute graph
- **Proxy / front-door RL and off-policy eval** — using a fully-mediating action representation to identify policy value under confounded logging

## Common mistake

Assuming a mediator that "mostly" mediates is good enough. The front-door requires M to intercept **all** directed X→Y paths — any direct X→Y edge bypassing M, or any unblocked X→M backdoor, breaks identification entirely (not just adds bias). Also: NDE ≠ "the coefficient on X with M added as a covariate" once there is X×M interaction or M–Y confounding.

## See also
- [[backdoor-criterion-confounding-adjustment]] — the complement: front-door is what you reach for precisely when no admissible backdoor set exists
- [[do-calculus]] — front-door is a two-step corollary of the three rewrite rules; the ID algorithm subsumes both
- [[activation-patching-causal-tracing]] — mediation analysis applied to neural network internals
