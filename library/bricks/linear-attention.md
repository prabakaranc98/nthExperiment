# Linear Attention (Kernel-Feature / Associative Form)

**One-liner:** Replace softmax(QKᵀ)V with φ(Q)(φ(K)ᵀV) so attention is an associative sum computable in O(n) time with a constant-size recurrent state Sₜ = Σ φ(kᵢ)vᵢᵀ — trading the growing KV cache for a fixed outer-product memory.

## The key insight

Standard attention computes oᵢ = Σⱼ sim(qᵢ,kⱼ)vⱼ / Σⱼ sim(qᵢ,kⱼ). Softmax forces sim(q,k)=exp(qᵀk), which couples q and k inside a nonlinearity → you must materialize the n×n matrix. Replace exp with a **separable kernel** sim(q,k)=φ(q)ᵀφ(k):

oᵢ = φ(qᵢ)ᵀ (Σⱼ φ(kⱼ)vⱼᵀ) / (φ(qᵢ)ᵀ Σⱼ φ(kⱼ))

Reassociate: compute the d×d (key-value) state **once**, then query it. Two equivalent forms:

- **Parallel:** (φ(Q)φ(K)ᵀ)V = φ(Q)(φ(K)ᵀV), O(n·d²) instead of O(n²·d).
- **Recurrent (causal):** Sₜ = Sₜ₋₁ + φ(kₜ)vₜᵀ;  oₜ = φ(qₜ)ᵀSₜ / (φ(qₜ)ᵀzₜ), zₜ = zₜ₋₁ + φ(kₜ). State Sₜ ∈ ℝ^{d×d} is **constant size** — O(1) memory per step at inference.

Feature maps: φ(x)=elu(x)+1 (Katharopoulos), φ(x)=x (linear/no-norm in modern variants), or random features approximating softmax (Performer's FAVOR+).

## Where it appears

- **Linear Transformers** (Katharopoulos et al., 2020) — the original φ(x)=elu+1 recurrent formulation; "transformers are RNNs" at inference.
- **Performer** (Choromanski et al., 2021) — FAVOR+ random features unbiasedly approximate the softmax kernel.
- **GLA / Gated Linear Attention, RetNet, Mamba-2** — add data-dependent decay to the state update Sₜ = Gₜ⊙Sₜ₋₁ + kₜvₜᵀ; trained via the chunked/chunkwise-parallel form for hardware efficiency.
- **DeltaNet / delta rule** — replaces the additive update with a least-squares "remove-then-write" rule to fix capacity overwrite.

## Common mistake

Thinking plain linear attention matches softmax quality. It does not: the rank-bounded d×d state can only store ~d key-value associations, so it degrades on long-range associative recall, and the unnormalized φ(q)ᵀφ(k) can go negative/blow up. The fixes that made it competitive are **gating/decay and the delta rule**, not the kernel trick alone — and FAVOR+ approximates softmax but adds variance.

## See also
- [[gated-linear-attention-data-dependent-decay]] — adds the input-dependent forget gate that makes linear attention actually work
- [[delta-rule-fast-weight-update]] — the better state-update rule that fixes capacity overwriting
- [[associative-recall-the-recall-state-size]] — why the constant d×d state caps what it can remember
- [[chunked-chunkwise-parallel-form]] — how these models are trained efficiently on GPUs
