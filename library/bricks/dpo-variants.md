# DPO Variants (IPO / KTO / ORPO / SimPO)

**One-liner:** The post-DPO zoo, each patching one DPO weakness — IPO replaces the logistic loss with a squared margin to stop preference-overfitting, KTO trains on unpaired good/bad labels via a prospect-theory utility, ORPO folds an odds-ratio penalty into the SFT loss (no reference, no separate stage), and SimPO drops π_ref entirely for a length-normalized average-logprob reward.

## The four losses (all start from DPO's implicit reward r̂ = β·log(π_θ/π_ref))

**IPO** (Azar et al., 2023). DPO's σ saturates: as the margin grows, gradient → 0 only at infinity, so it overfits deterministic preferences and pushes logps to extremes. IPO instead regresses the margin to a fixed target 1/(2β):
L_IPO = E[ ( log(π_θ(y_w)/π_ref(y_w)) − log(π_θ(y_l)/π_ref(y_l)) − 1/(2β) )² ]
Bounded target ⇒ no runaway; β here directly caps the achievable margin.

**KTO** (Ethayarajh et al., 2024). No pairs needed — just per-example "desirable/undesirable" labels. Define implicit reward r̂(x,y)=β·log(π_θ/π_ref) and a reference point z₀=KL(π_θ‖π_ref) (estimated, gradient-stopped). Maximize a Kahneman–Tversky value:
v = σ(r̂ − z₀) for desirable, σ(z₀ − r̂) for undesirable, scaled by λ_D/λ_U.
Robust to label noise and to class imbalance via the λ weights.

**ORPO** (Hong et al., 2024). Reference-free and single-stage: monolithic SFT + odds-ratio penalty, no π_ref, no separate preference phase.
L = L_SFT(y_w) − λ·log σ( log[odds(y_w)/odds(y_l)] ),  odds(y)=p(y)/(1−p(y))
The SFT term keeps p(y_w) up; the OR term widens the gap to y_l.

**SimPO** (Meng et al., 2024). Reference-free, length-normalized reward to align the training objective with greedy decoding (which scores avg logprob) and to kill length bias:
r(x,y) = (β/|y|)·log π_θ(y|x),  L = −log σ( r(x,y_w) − r(x,y_l) − γ )
γ is a target reward margin; the 1/|y| normalization is the key difference from DPO.

## Where it appears

- **Llama 3 / Tülu 3 / Zephyr post-training** — DPO + length-normalization tricks and on-policy data, directly motivated by SimPO/IPO findings
- **KTO in production alignment** — used when only thumbs-up/down telemetry exists (no curated pairs), e.g. RLHF from sparse user signals
- **ORPO for cheap one-shot alignment** — small-model and on-device SFT pipelines that can't afford a reference model in memory
- **SimPO / CPO** — strong reference-free baselines on AlpacaEval 2 / Arena-Hard leaderboards (2024–2025)

## Common mistake

Treating these as strictly better than DPO. The reference-free ones (SimPO, ORPO) drop the explicit KL-to-reference anchor, so the only thing preventing distribution collapse / reward hacking is the SFT term, β, or the margin γ — they are far more sensitive to these knobs and to data quality. SimPO's length normalization helps *only* because the reward now matches greedy decoding; misapplied, it just trades length bias for other artifacts.

## See also
- [[dpo]] — the parent objective; every variant is a one-term edit of its logistic-on-log-ratio loss
- [[length-normalization-bias-control]] — SimPO's 1/|y| reward is exactly this fix applied to the implicit reward
- [[kl-regularization-to-reference-policy]] — the anchor IPO/KTO keep and ORPO/SimPO discard
