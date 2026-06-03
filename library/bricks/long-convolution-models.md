# Long-Convolution Models (Hyena / H3)

**One-liner:** Replace attention with data-controlled gating of an implicitly-parameterized, sub-quadratic global convolution whose filter spans the whole sequence (computed via FFT in O(N log N)); H3 is the canonical "why pure SSMs fail at recall" study.

## The key insight

A causal convolution `y = h * x` with a filter `h` as long as the sequence gives global token mixing. Done naively it costs O(N^2); done by FFT it costs O(N log N):

    y = iFFT( FFT(h_pad) * FFT(x_pad) )   # circular conv, zero-padded to 2N for causal

The trick is *not materializing* `h` as N free parameters. Instead `h` is **implicit**: a function of position evaluated by a small MLP, `h_t = window(t) * MLP(PositionalEmbedding(t))`. This decouples filter length from parameter count (extend to longer N for free) and keeps the filter smooth/learnable.

**Hyena** = recurrence of (long conv) interleaved with **data-controlled gating**. With projections u → (x1, x2, v):

    z_0 = v
    z_{i} = x_i ⊙ (h_i * z_{i-1})        # elementwise gate, then long conv, repeated order=N times
    y     = z_N

The elementwise multiply by an input-dependent `x_i` is what recovers attention-like data dependence — a static conv alone cannot do content-based routing.

**H3** (Hungry Hungry Hippos) is the SSM-flavored ancestor: two stacked SSMs (a *shift* SSM that acts as a local memory + a *diagonal* SSM) combined with multiplicative gating, designed specifically to fix the recall failure of vanilla S4.

## Where it appears

- **H3 (Dao, Fu et al. 2022)** — diagnosed that pure LTI SSMs (S4) cannot *compare tokens across the sequence* or recall earlier tokens; the synthetic *induction head* and *associative recall* tasks expose this. Fix: shift-SSM + multiplicative interaction → motivated FlashConv and ultimately Mamba's selectivity.
- **Hyena (Poli et al. 2023)** — first attention-free LM to match Transformers at sub-2k context on The Pile; implicit long filters + gating.
- **HyenaDNA / Evo / genomics** — million-token DNA context where O(N^2) attention is infeasible.
- **M2 / Monarch Mixer, conv backbones in hybrid stacks** — long-conv layers as cheap global-mixing primitives.

## Common mistake

Thinking a long convolution alone replaces attention. It does not: a (linear time-invariant) conv applies the *same* filter regardless of content, so it cannot do input-dependent routing or exact recall — exactly the gap H3 measured. The data-dependent **gating** (Hyena) or **selectivity** (Mamba) is the essential ingredient; the FFT conv is just the efficient global-mixing engine. Mamba later made the recurrence itself input-dependent, superseding most pure long-conv LMs.

## See also
- [[recurrence-convolution-scan-duality]] — long conv is the convolutional view of an LTI SSM; the FFT and scan are the same operator
- [[associative-recall-the-recall-state-size]] — the precise task on which pure SSMs/convs fail and gating is needed
- [[selective-state-space-models-mamba]] — input-dependent selectivity that subsumed long-conv gating
