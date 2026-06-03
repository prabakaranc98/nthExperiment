# Bradley-Terry Model

**One-liner:** A pairwise-comparison model where P(i beats j) = σ(s_i − s_j), fitting latent quality scores by MLE on win/loss data — the statistical backbone of RLHF reward models and Arena leaderboards.

## The formula / definition

Each item i has a latent score s_i (a "merit"). The probability i is preferred over j depends only on the difference:

P(i ≻ j) = σ(s_i − s_j) = exp(s_i) / (exp(s_i) + exp(s_j))

The negative log-likelihood over a dataset of pairwise comparisons {(winner w, loser l)} is:

L = −Σ log σ(s_w − s_l)

Scores are identifiable only up to an additive constant (shift all s_i → s_i + c, probabilities unchanged), so you pin one score or add a regularizer/prior. Logistic regression with a difference-of-features design matrix recovers the MLE; equivalent to a 1-parameter logistic IRT model.

## Where it appears

- **RLHF reward modeling** (InstructGPT, Llama 2/3) — the reward model loss IS the BT NLL: −log σ(r_θ(x, y_w) − r_θ(x, y_l)) over chosen/rejected response pairs. The scalar reward head plays the role of s.
- **DPO** — derived by substituting the closed-form optimal RLHF policy into the BT likelihood, eliminating the explicit reward model; the implicit reward is β·log(π_θ/π_ref).
- **Chatbot Arena / LMSYS leaderboards** — fits BT scores to crowd pairwise battles (then often displayed as Elo-scaled ratings); Arena's "BT coefficients" are exactly s_i.

## Common mistake

Conflating Bradley-Terry with Elo. BT is a static MLE fit to all comparisons jointly; Elo is an online stochastic-approximation update (one gradient step per game with fixed learning rate K). They share the logistic form and agree at convergence under stationarity, but Elo is path-dependent and BT is not. Also: BT models transitive strength only — it cannot capture rock-paper-scissors / non-transitive matchups (use a low-rank or disc-based extension for that).

## See also
- [[reward-modeling]] — the reward model objective is the BT log-likelihood
- [[dpo]] — derived directly from the BT preference model
- [[elo-online-rating-for-model-ranking]] — the online cousin used to display Arena ratings
