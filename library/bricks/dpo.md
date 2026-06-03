# Direct Preference Optimization (DPO)

**One-liner:** Closed-form preference training — fit the policy directly on preference pairs with a binary-classification loss whose implicit reward is the policy's own log-ratio against a reference, eliminating the separate reward model and the RL loop.

## The derivation in one step

RLHF maximizes E[r(x,y)] − β·KL(π_θ ‖ π_ref). This KL-regularized objective has a known closed-form optimum:

π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)

Invert it to express reward in terms of the optimal policy:

r(x,y) = β·log(π_θ(y|x)/π_ref(y|x)) + β·log Z(x)

Plug this **implicit reward** into the Bradley–Terry preference model P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l)). The intractable partition function Z(x) cancels (it depends only on x), giving the DPO loss:

L = −E_{(x,y_w,y_l)}[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]

That's it: a logistic loss on the difference of two log-ratios. No reward model, no sampling, no PPO.

## Where it appears

- **DPO** (Rafailov et al., NeurIPS 2023) — the original; Zephyr, Tülu 2 popularized it as the cheap RLHF replacement
- **Llama 3 / 3.1 post-training** — iterative DPO on on-policy preference pairs is a core alignment stage
- **IPO / KTO / SimPO / ORPO** — the variant zoo: IPO fixes BT overfitting with a squared loss, KTO uses unpaired binary feedback, SimPO drops π_ref and uses length-normalized reward, ORPO folds preference into SFT
- **cDPO / robust DPO** — label-smoothed versions for noisy preferences

## Common mistake

Believing DPO is fully "RL-free" and reference-free. The β·log(π_θ/π_ref) term *is* a KL-constrained reward — drop π_ref or set β wrong and you get reward hacking by another name. The classic failure: DPO often *lowers the probability of the chosen response* (both logps fall, the *margin* just grows), so likelihood is the wrong thing to monitor. Off-policy preference data also drifts from the policy, which is why iterative/on-policy DPO and SimPO-style fixes exist.

## See also
- [[rlhf]] — the PPO+reward-model pipeline DPO collapses into one loss
- [[grpo]] — the on-policy RL alternative; DPO is its offline, sampling-free counterpart
- [[kl-divergence]] — the β-weighted KL-to-reference is the regularizer DPO inherits and bakes into its implicit reward
