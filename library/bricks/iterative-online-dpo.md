# Iterative / Online DPO

**One-liner:** Close DPO's distribution-mismatch gap by looping {sample on-policy from the current model → label fresh pairs with a judge/RM → run a round of DPO}, recovering much of online RL's gains while keeping DPO's offline simplicity (no value net, no rollout PPO machinery).

## The key insight

Vanilla DPO trains on a *fixed offline* preference set. As the policy π_θ drifts away from the data-collection policy, those preference pairs become off-policy and stale — the loss optimizes regions the model no longer visits. Iterative DPO fixes this by regenerating pairs from the current policy each round:

```
π_0 ← SFT model
for t = 0, 1, 2, ... (rounds):
    # 1. on-policy generation
    for each prompt x:
        sample y_a, y_b ~ π_t(· | x)        # often K samples, take best/worst
    # 2. labeling (the "online" signal)
    (y_w, y_l) ← judge(x, {y_i})            # RM score, LLM-as-judge, or rule/verifier
    # 3. DPO update on the FRESH pairs
    π_{t+1} ← argmin_θ  E[ -log σ( β log (π_θ(y_w|x)/π_ref(y_w|x))
                                      - β log (π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

The DPO loss is unchanged; what changes is that the (y_w, y_l) distribution tracks π_θ. "Online" = label as you generate (streaming feedback); "iterative" = batch the loop into discrete rounds. The reference π_ref is typically reset to π_t at the start of each round (or kept as the original SFT to bound drift).

## Where it appears

- **Self-Rewarding LMs (Yuan et al., 2024)** — the model judges its own generations to build each round's preference set; iterative DPO over rounds improves both instruction-following and the judging ability.
- **Llama-3 post-training** — rejection-sampling best-of-N from the current checkpoint, RM-labeled, then DPO; repeated across post-training rounds.
- **Online DPO / OAIF (Guo et al., 2024)** — replaces the static dataset with an online annotator (a strong LLM judge) giving fresh feedback on on-policy samples, narrowing the gap to RLHF.
- **SPIN / self-play** — uses the model's own samples as "losers" against ground-truth "winners," an iterative DPO-style loop without an external RM.

## Common mistake

Thinking iterative DPO is just "DPO run a few times on the same data" — the whole point is *regenerating samples from the updated policy each round*. Re-running on the original offline set buys you almost nothing. Second mistake: forgetting to refresh/manage π_ref — leaving it as round-0 SFT while the policy moves far can make the implicit reward miscalibrated, while resetting it every round can let the model collapse or reward-hack the judge.

## See also
- [[dpo]] — the offline base method this loops on
- [[rejection-sampling-best-of-n]] — the standard on-policy generation+filtering step inside each round
- [[reward-hacking-over-optimization]] — repeated rounds against a fixed judge/RM invite over-optimization
