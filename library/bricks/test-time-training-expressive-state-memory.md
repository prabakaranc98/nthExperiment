# Test-Time Training / Expressive-State Memory (TTT, Titans)

**One-liner:** Reframe the RNN hidden state as a small learnable model (matrix or MLP) whose weights ARE the memory, updated by online (self-supervised) gradient descent on each incoming token — so the recurrent update is one inner-loop SGD step and "the sequence" is the training set.

## The formula / definition

Memory is a fast-weight model with state Wₜ (a matrix or small MLP). Per token, take a gradient step on a self-supervised inner loss ℓ (e.g. reconstruct value from key):

  Wₜ = Wₜ₋₁ − η ∇_W ℓ(Wₜ₋₁; xₜ),  ℓ = ‖Wₜ₋₁ kₜ − vₜ‖²

Output: yₜ = Wₜ qₜ. With ℓ a linear-reconstruction loss this gradient step is exactly the **delta rule** (Wₜ = Wₜ₋₁ + βₜ(vₜ − Wₜ₋₁kₜ)kₜᵀ) — linear attention with error-correcting writes.

**Titans (Behrouz et al., 2024)** generalize the optimizer of the inner loop: add **momentum** (surprise = past gradient) and a data-dependent **weight decay** as forgetting gate:

  Sₜ = θₜ Sₜ₋₁ − ηₜ ∇ℓ ; Wₜ = (1 − αₜ) Wₜ₋₁ + Sₜ

αₜ controls forgetting, ηₜ controls momentary surprise; both are functions of the input. The memory module Wₜ can be a 2+ layer MLP ("deep memory"), giving capacity beyond a rank-update matrix. Training is parallelized with a **mini-batch / chunkwise** inner-loop (TTT-Linear, TTT-MLP; Sun et al., 2024).

## Where it appears

- **TTT (Sun et al., 2024, "Learning to (Learn at Test Time)")** — TTT-Linear / TTT-MLP as drop-in linear-time sequence layers; hidden state literally trained by GD on the test sequence.
- **Titans (Google, 2024) / Atlas, Moneta** — deep neural long-term memory with momentum + forgetting; "Memory as Context/Gate/Layer" variants for long context.
- **DeltaNet / Gated DeltaNet (2024-25)** — delta-rule fast weights = the special-case linear-memory TTT, with hardware-aware chunked parallel scan.

## Common mistake

Thinking the inner-loop "learning at test time" updates the *network's* weights or persists across sequences. It doesn't: the outer (slow) weights are frozen at inference; only the per-sequence memory state Wₜ is "trained," and it resets each new sequence. The GD step IS the recurrence — not a separate fine-tuning phase.

## See also
- [[delta-rule-fast-weight-update]] — the linear-memory special case; error-correcting state write
- [[gated-linear-attention-data-dependent-decay]] — data-dependent forgetting gate, same family
- [[linear-attention]] — the matrix-valued state these methods make expressive and trainable
