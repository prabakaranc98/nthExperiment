# Induction Heads

**One-liner:** A two-head circuit — a previous-token head writing token[i−1]'s identity into position i, then an induction head that K-composes on it to attend [A][B]…[A]→[B] and copy B — the mechanistic substrate of in-context copying whose formation drives the ICL loss phase change.

## The circuit (Olsson et al., 2022 / Elhage et al., 2021)

Two layers, composing through the residual stream:

1. **Previous-token head** (layer L): at position `i`, attends to `i−1` and copies its token embedding into the residual stream at `i`. Now position `i` "knows" what preceded it.
2. **Induction head** (layer L+1): its query reads the *current* token A; via Q/K composition it matches against the previous-token info, finding the earlier position where A also occurred; it then attends to the *next* token (B) at that prior site and its OV circuit copies B to the output logits.

Behaviorally: given `…[A][B]…[A]`, predict `[B]`. The match is on the **prefix** (the token before the candidate), not on A's value alone — this is "prefix matching" + "copying."

QK circuit = where to look (find prior A); OV circuit = what to write (boost the token that followed). Requires **K-composition**: layer L+1's keys read what layer L wrote.

## Where it appears

- **A Mathematical Framework for Transformer Circuits** (Elhage 2021) — formalized via QK/OV decomposition and head composition (Q/K/V-composition).
- **In-context Learning and Induction Heads** (Olsson 2022) — the per-token ICL loss drop coincides with a sharp bump in induction-head formation early in training; argued as the primary mechanism of ICL in small/mid models.
- **Function vectors / task vectors, "in-context heads"** — induction is the prototype; later work finds richer in-context heads doing fuzzy/semantic matching, not just literal copy.
- **Long-context & RoPE extension** — induction is the canonical probe for whether a model copies across extended positions; failures localize to these heads.

## Common mistake

Believing induction heads do general in-context *learning* or reasoning. They implement literal (and some fuzzy) **copying / pattern-completion**; that is necessary scaffolding for ICL but not the same as learning a new task in-context. Also: it is a *two-head* composed circuit — a single head cannot do it, since matching needs the previous-token head's output already in the stream.

## See also
- [[in-context-learning]] — induction heads are the mechanistic correlate of the ICL phase change
- [[qk-ov-circuits-head-decomposition]] — the QK (where) / OV (what) factorization the circuit is built from
- [[residual-stream]] — the shared channel through which the two heads K-compose
