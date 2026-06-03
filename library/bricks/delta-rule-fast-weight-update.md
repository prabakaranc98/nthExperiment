# Delta Rule / Fast-Weight Update (DeltaNet)

**One-liner:** Replace linear attention's pure-additive state write (S ← S + vkᵀ) with an error-correcting delta rule S ← S(I − βkkᵀ) + βvkᵀ, so the recurrent KV-state *overwrites* the value currently associated with key k instead of blindly accumulating, giving finite-state memory the ability to forget/update associations.

## The formula

Linear attention stores associations additively in a matrix state S (the "fast weights"):

  S_t = S_{t-1} + v_t k_tᵀ,  output o_t = S_t q_t

DeltaNet (Schlag et al. 2021, "Linear Transformers Are Secretly Fast Weight Programmers") instead does a single online gradient step on the loss ‖S k_t − v_t‖²:

  S_t = S_{t-1} − β_t (S_{t-1} k_t − v_t) k_tᵀ
     = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ

Read out the *old* value v̄_t = S_{t-1} k_t (assume ‖k_t‖=1), then write the corrected residual β_t(v_t − v̄_t)k_tᵀ. β_t ∈ (0,1) is a learned data-dependent write strength; β_t=1 is a hard overwrite, β_t=0 a no-op. The (I − βkkᵀ) factor is a rank-1 "Householder-like" deletion that erases the stale association before inserting the new one.

## The chunkwise parallel form (what made it trainable at scale)

The (I − βkkᵀ) product is sequential, so naive DeltaNet can't use matmuls. Yang et al. (2024, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length") rewrite the recurrence via the WY representation: within a chunk the cumulative product of rank-1 updates becomes a matrix inverse (I − tril(diag(β)KKᵀ))⁻¹ solved once per chunk, recovering hardware-efficient O(L·d²) chunked matmul training.

## Where it appears

- **DeltaNet** (Schlag '21; Yang '24) — the canonical delta-rule linear-attention RNN; chunkwise form is the reference kernel in the `flash-linear-attention` library.
- **Gated DeltaNet** (Yang et al. 2024) — adds a Mamba2-style scalar decay α_t to the delta rule: S_t = S_{t-1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ, combining global forgetting with targeted overwrite; backbone of NVIDIA's **Nemotron-H**-style and **Qwen3-Next** hybrid layers.
- **DeltaProduct** (2025) — uses multiple delta steps per token (products of Householders) to raise state-transition expressivity / extend the trackable state-machine class.
- **Hybrid attention–SSM stacks** — delta-rule layers interleaved with full attention (e.g. MiniMax-01-style, Qwen3-Next) for constant-memory recall with better associative-recall accuracy than vanilla linear attention.

## Common mistake

Thinking the delta rule just adds a forget gate. A scalar/diagonal decay (GLA, Mamba) shrinks the *whole* state uniformly; the delta rule selectively *replaces* the content stored at one key direction k via the rank-1 projection (I − βkkᵀ), leaving orthogonal associations untouched. That targeted overwrite — not global decay — is what fixes linear attention's "memory overflow" on associative recall.

## See also
- [[gated-linear-attention-data-dependent-decay]] — combine its data-dependent decay with the delta write to get Gated DeltaNet
- [[chunked-chunkwise-parallel-form]] — the WY/matrix-inverse trick that makes the sequential delta recurrence GPU-parallel
- [[associative-recall-the-recall-state-size]] — the benchmark axis where overwriting beats pure additive accumulation
