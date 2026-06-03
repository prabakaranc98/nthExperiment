# Decoupled Weight Decay

**One-liner:** Multiplicative per-step shrinkage of weights applied directly to the parameter, decoupled from the loss gradient; under Adam it behaves less like classic L2 regularization and more as an implicit control on weight norm / effective learning rate via the LR×wd product.

## The formula

Classic L2 (coupled): add λ‖θ‖² to the loss, so the gradient gains a +λθ term that then flows through the optimizer's preconditioner. Under Adam the adaptive denominator √v̂ rescales that term, so the effective decay is *not* uniform across coordinates — large-gradient weights get decayed less. This is the bug Loshchilov & Hutter (AdamW, 2017/2019) identified.

**AdamW (decoupled):** apply decay straight to the weight, outside the adaptive update:

θ ← θ − η · ( m̂ / (√v̂ + ε) ) − η · λ · θ

The decay step − η·λ·θ is a clean exponential pull toward zero, independent of g and of v̂. Note it scales with the LR η, so the relevant knob is the **product η·λ**, not λ alone — retune wd whenever you change the LR schedule.

## Where it appears

- **AdamW** — the default optimizer for essentially all transformer pretraining (GPT, Llama, etc.); fixed Adam's broken L2.
- **Effective-LR / norm-growth dynamics** — with normalization layers, weight scale is gauge-irrelevant for the forward pass, so wd mainly controls the *effective* learning rate by capping ‖θ‖ at an equilibrium where decay balances gradient growth (rotational equilibrium).
- **Muon / Lion / modern optimizers** — carry decoupled decay over; spectral-norm methods often pair it with explicit norm constraints.
- **muP / hyperparameter transfer** — wd is one of the HPs whose scaling must be reasoned about jointly with η.

## Common mistake

Treating AdamW weight decay as overfitting prevention you tune like L2. At scale with LayerNorm/RMSNorm it is mostly a stable-norm / effective-LR controller: the steady-state weight norm is set by √(η/λ)-type balance, so doubling η changes the regularization unless you also adjust λ. Also: people forget to *exclude* norm gains, biases, and embeddings from decay (decaying them is usually wrong/harmful).

## See also
- [[adamw]] — the canonical realization of decoupled weight decay
- [[effective-lr-norm-growth-dynamics]] — why wd acts as a norm/effective-LR knob under normalization
- [[implicit-bias]] — decoupled decay shapes the implicit regularization of the trained solution
