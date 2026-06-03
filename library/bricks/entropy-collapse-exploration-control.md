# Entropy Collapse & Exploration Control

**One-liner:** During RLVR/RLHF the policy's token-level output entropy can collapse toward a deterministic mode, killing exploration and locking in pass@1 at the cost of pass@k; countered with entropy bonuses, decoupled clipping (DAPO), temperature, and KL tuning.

## The key insight

Entropy H(π) = −Σ_a π(a|s) log π(a|s) measures policy stochasticity. Under policy-gradient RL, the standard PPO/GRPO objective has no force keeping H up, and clipping is asymmetric: the lower clip (1−ε) bounds how much you suppress already-low-probability tokens, but the upper clip (1+ε) lets high-probability tokens be reinforced freely → the distribution sharpens monotonically and H → 0. Empirically (Cui et al. 2025, "The Entropy Mechanism"), validation reward tracks a near-deterministic fit R = a − b·exp(H): once entropy is spent, the policy stops improving. Collapse correlates with the high-covariance tokens (logit ∝ advantage), so clipping those is the lever.

Countermeasures:
- **Entropy bonus:** add +β·H(π) to the loss (β ~ 1e-3 to 1e-2); blunt, can destabilize.
- **DAPO decoupled clip (Clip-Higher):** split ε into ε_low, ε_high with ε_high > ε_low (e.g. 0.20 vs 0.28) so low-prob exploratory tokens can still be boosted — keeps entropy alive without an explicit bonus.
- **KL to reference π_ref:** anchors the policy; DAPO and many RLVR recipes *drop* the KL term entirely to let entropy/distribution drift for harder exploration.
- **Sampling temperature T:** rollout T>1 widens the behavior distribution feeding training (separate from the loss-side fixes).
- **Clip-Cov / KL-Cov:** directly clip or penalize the highest covariance(logp, advantage) tokens to cap the entropy drain.

## Where it appears

- DAPO (ByteDance/Tsinghua, 2025) — Clip-Higher + dynamic sampling + no KL; explicitly motivated by GRPO entropy collapse on AIME-style math RL.
- "The Entropy Mechanism of RL for LLMs" (Cui et al. 2025) — R = a − b·e^H law; Clip-Cov / KL-Cov interventions.
- Open-Reasoner-Zero, DeepSeek-R1 lineage, Skywork-OR1 — monitor entropy as a first-class training-health signal; collapse means the run has stopped learning.
- pass@k literature — entropy-preserving RL widens pass@k (diverse solutions) even when pass@1 gains saturate; collapsed policies ace pass@1 but lose pass@k coverage.

## Common mistake

Treating low entropy as "the model got confident/correct." Collapse usually means the policy memorized one rollout mode and can no longer explore alternative reasoning paths — reward plateaus and pass@k degrades. The fix is rarely just cranking the entropy bonus (which often causes loss spikes or gibberish); asymmetric clip-side / covariance-targeted control is more stable than a global +β·H.

## See also
- [[grpo]] — the group-relative objective where collapse is most acute (no learned value baseline)
- [[ppo-clipped-surrogate-objective]] — the asymmetric clip whose decoupling (DAPO) controls entropy
- [[pass-k-self-consistency-estimation]] — the pass@k-vs-pass@1 tradeoff entropy directly governs
