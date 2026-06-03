# AdamW

**One-liner:** Adam with *decoupled* weight decay — the L2 penalty is applied directly to the weights as a separate shrinkage step instead of being added into the gradient, making regularization independent of the adaptive per-parameter learning rate; the default optimizer and the baseline every new optimizer is benchmarked against.

## The update rule (Loshchilov & Hutter, 2019)

Per step t, with gradient g_t, betas (β₁, β₂), lr η, decay λ:

m_t = β₁ m_{t-1} + (1−β₁) g_t        (first moment)
v_t = β₂ v_{t-1} + (1−β₂) g_t²        (second moment)
m̂_t = m_t / (1−β₁^t),  v̂_t = v_t / (1−β₂^t)   (bias correction)

θ_t = θ_{t-1} − η ( m̂_t / (√v̂_t + ε) + λ θ_{t-1} )

The decoupling: the **λ θ** term sits *outside* the adaptive `/√v̂` rescaling. In plain Adam-with-L2 you instead do g_t ← g_t + λθ, which then gets divided by √v̂ — so parameters with large v̂ (noisy/large gradients) get *less* effective decay. AdamW gives every weight the same multiplicative shrinkage (1 − ηλ).

## Where it appears

- **Every transformer pretrain** — GPT, Llama, etc. use AdamW with β=(0.9, 0.95), λ≈0.1, ε=1e-8, often decoupled-decay-excluded on norms/biases.
- **Muon / Shampoo benchmarks** — new optimizers (2024-2026) report wall-clock or token-to-loss speedup *over a tuned AdamW baseline*; AdamW is the bar to beat.
- **LoRA / QLoRA fine-tuning** — AdamW (or 8-bit/paged AdamW for memory) is the standard adapter optimizer.

## Common mistake

Thinking AdamW's "weight decay" equals L2 regularization. It does not — that equivalence holds for plain SGD but breaks under adaptive preconditioning. Adam-with-L2 ≠ AdamW. Also: in AdamW the effective shrinkage is η·λ, so **changing the LR rescales your regularization** unless you account for the coupling.

## See also
- [[decoupled-weight-decay]] — the precise mechanism that distinguishes AdamW from Adam+L2
- [[adam-update-rule]] — the moment estimates AdamW is built on
- [[muon-optimizer]] — the leading 2024+ challenger benchmarked against AdamW
