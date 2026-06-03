# LLM-as-a-Judge

**One-liner:** Use a strong LLM with a rubric to score or pairwise-rank model outputs as a scalable proxy for human evaluation — the default open-ended-eval substrate and a cheap reward/verifier signal.

## The two protocols

**Pointwise (scoring):** prompt the judge with (instruction, response, rubric) → return a scalar score, usually 1–10 or a Likert scale, optionally with a CoT critique first.

**Pairwise (preference):** prompt with (instruction, response A, response B) → return a winner ∈ {A, B, tie}. More reliable than absolute scores because relative judgments are easier than calibrated ones.

Pairwise wins aggregate into rankings via Bradley-Terry / Elo:

P(A ≻ B) = σ(s_A − s_B),  fit scores s by MLE over all judged pairs.

## Where it appears

- **MT-Bench / Chatbot Arena** (Zheng et al., 2023) — GPT-4 judge for multi-turn quality; validated ~80% agreement with human preference, matching inter-human agreement.
- **AlpacaEval 2.0 / Arena-Hard** — length-controlled win-rate vs. a reference model (e.g. GPT-4) as a leaderboard metric.
- **RLAIF / Constitutional AI** (Bai et al., 2022) — judge produces the preference labels that train the reward model, replacing human annotators.
- **RLVR & agentic RL** — for non-verifiable tasks the judge acts as a soft verifier / reward model; for math/code a rule-based checker is preferred over a judge.
- **Self-rewarding LMs** (Yuan et al., 2024) — model judges its own generations to bootstrap preference data.

## Common mistake

Treating the judge as an unbiased oracle. Judges have systematic biases: **position bias** (favoring the first/second option), **length/verbosity bias** (longer = better), **self-preference** (favoring outputs from the same model family), and **sycophancy**. Mitigations: swap A/B order and average, length-control, and use a judge from a different family than the model under test. Critically, you cannot use a judge to reliably rank a model *stronger than the judge itself*.

## See also
- [[judge-bias-mitigation]] — position/length/self-preference debiasing techniques
- [[bradley-terry-model]] — turns pairwise judgments into a global ranking
- [[rlaif-constitutional-ai]] — judge as the source of preference labels for alignment
