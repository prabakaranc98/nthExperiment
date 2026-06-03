# Expert Load Balancing

**One-liner:** Mechanisms (auxiliary load-balancing loss, or aux-loss-free per-expert bias correction) plus capacity-factor / token-dropping tuning that stop an MoE router from collapsing onto a few experts and wasting the rest of its parameters.

## The aux-loss formulation (Switch / GShard)

For E experts over a batch of T tokens, let f_i = fraction of tokens routed to expert i, and P_i = mean router probability (softmax gate) for expert i over the batch. The load-balancing loss is

  L_aux = α · E · Σ_i f_i · P_i

f_i is non-differentiable; P_i carries the gradient. Minimizing this pushes f_i and P_i toward the uniform 1/E. Typical α ≈ 0.01. Added to the LM loss. (Often paired with a [[z-loss-logit-stabilization]] on router logits.)

## Aux-loss-free balancing (DeepSeek-V2/V3)

Drop L_aux (it distorts the LM gradient and hurts quality). Instead keep a per-expert bias b_i added only to the routing score used for top-k selection — NOT to the gate value that weights the output:

  affinity s_i = sigmoid(token · centroid_i),  select top-k of (s_i + b_i)

After each step, nudge biases by the sign of the load error: b_i ← b_i + γ · sign(load_target − load_i). Overloaded experts get a lower bias, starved ones get raised, with zero interference to the loss landscape.

## Capacity factor and token dropping

Each expert holds at most C = capacity_factor · (T·k / E) tokens. Overflow tokens are dropped (skip the expert, pass through residual) or rerouted. CF ≈ 1.0–1.25 train, often higher (or drop-free) at inference. Low CF + bad balance = many dropped tokens.

## Where it appears

- **Switch Transformer / GShard** — original auxiliary loss L_aux = α·E·Σ f_i·P_i with hard capacity + token dropping.
- **DeepSeek-V3 / DeepSeekMoE** — aux-loss-free bias-update balancing as the headline trick; fine-grained + shared experts; tiny sequence-wise aux loss only to prevent intra-sequence collapse.
- **Mixtral, Qwen-MoE, OLMoE** — top-k routing with Switch-style aux loss; balance reported via per-expert token-distribution entropy.
- **Expert parallelism (EP)** — imbalance becomes a wall-clock problem: the slowest GPU's expert gates the all-to-all, so balance directly drives [[mfu-model-flops-utilization]].

## Common mistake

Conflating router collapse with the aux loss being "too weak." Cranking α up forces uniform routing but degrades quality (experts can't specialize) — the loss is a quality/balance tradeoff, not a free lunch. The aux-loss-free bias trick exists precisely to get balance without paying that tax. Also: adding the balancing bias to the output gate weight (it must only affect top-k selection, not the combine weights).

## See also
- [[moe-routing]] — the top-k gating mechanism this loss regularizes
- [[expert-parallelism]] — why imbalance hurts: stragglers in the all-to-all
- [[z-loss-logit-stabilization]] — co-deployed router-logit regularizer for stable MoE training
