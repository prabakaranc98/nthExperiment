# Elo / Online Rating for Model Ranking

**One-liner:** An incremental SGD-style update on Bradley-Terry log-skills — maintain one rating per model and after each pairwise battle nudge winner up / loser down by an amount proportional to surprise; Chatbot Arena's lingua franca, but order-dependent and noisy.

## The update rule

Expected score (logistic, scale 400):

E_A = 1 / (1 + 10^((R_B − R_A)/400))

After a battle with outcome S_A ∈ {1, 0.5, 0} (win/tie/loss):

R_A ← R_A + K · (S_A − E_A),   R_B ← R_B + K · (S_B − E_B)

K is the step size (Arena uses K≈4–32 scaled by battle count). This is online SGD on the Bradley-Terry NLL: P(A beats B) = σ(R_A − R_B) in nats, or σ((R_A−R_B)·ln(10)/400) at Elo scale. Ratings ARE log-odds of winning, up to scale + offset.

## Where it appears

- **LMSYS Chatbot Arena** — crowd pairwise votes → model leaderboard. Now reports **Bradley-Terry MLE** (batch logistic regression, all data at once) instead of sequential Elo, since online Elo's final ranking depends on battle order. Bootstrap CIs over the BT fit give the error bars.
- **RLHF reward models** — the BT loss −log σ(r(x,y_w) − r(x,y_l)) is the same objective; DPO inherits it. Elo/BT scores and reward-model scores are the same latent quantity.
- **Arena-Hard / auto-eval** — LLM-as-judge pairwise battles fed into the same BT/Elo machinery to rank models cheaply.

## Common mistake

Treating online Elo as a fixed property of a model. Sequential Elo is **path-dependent** (final rating depends on match order and K), under-determined for models with few games, and unidentifiable in absolute terms (only differences matter). For a static pool, fit the **Bradley-Terry MLE with bootstrap CIs** — don't read a 5-point gap inside overlapping intervals as real, and beware non-transitivity (A>B>C>A) that a single scalar cannot represent.

## See also
- [[bradley-terry-model]] — the static MLE Elo is the online approximation of
- [[llm-as-a-judge]] — the dominant source of automated pairwise battles
- [[a-b-testing-statistics]] — bootstrap CIs and significance for rating gaps
