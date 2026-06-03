# Judge Bias & Mitigation

**One-liner:** Systematic distortions in LLM-as-judge scoring — position, verbosity, self-preference, sycophancy, bandwagon — corrected by position-swapping, length controls, jury/panel aggregation, and reference-anchored rubrics; core eval hygiene for any model-graded benchmark.

## The key insight

LLM judges do not score on content alone; the judgment factorizes as `score = quality + Σ bias_i`. Known biases (Zheng et al., MT-Bench/Chatbot Arena 2023):

- **Position bias:** judge favors the first (or last) option. Mitigate by swapping order and keeping the verdict only if consistent: report `win` iff A beats B in *both* orderings, else `tie`. Measures: position consistency, preference fairness.
- **Verbosity/length bias:** longer answers score higher even when no better. Control by regressing out length or report length-controlled win rate (AlpacaEval 2.0 LC: fit a GLM, `logit(win) = θ·quality + γ·Δlen`, then predict at `Δlen=0`).
- **Self-preference / self-enhancement:** a judge rates its own family's outputs higher. Mitigate with a *different-family* judge or a jury.
- **Sycophancy / bandwagon:** caves to stated user opinion or majority label.

**Jury / panel (PoLL):** aggregate K diverse judges, `score = (1/K) Σ_k s_k` (or majority vote). Cheaper, more diverse small judges beat one large judge while reducing intra-model self-bias and cutting cost.

## Where it appears

- **MT-Bench / Chatbot Arena (Zheng 2023)** — formalized position, verbosity, self-enhancement bias; swap-and-require-consistency protocol.
- **AlpacaEval 2.0** — length-controlled win rate to neutralize verbosity gaming; closed the loop where models learned to win by being verbose.
- **Panel-of-LLM-evaluators / PoLL (Cohere 2024)** — juries of small heterogeneous judges to cut self-preference and variance.
- **G-Eval / Prometheus** — rubric + reference-answer anchoring to reduce free-floating bias; chain-of-thought before the score.

## Common mistake

Trusting a single pass of a single judge as ground truth. A score is only as good as its bias controls: without order-swapping you are measuring position bias, without length control you reward verbosity, and using the same model as both generator and judge bakes in self-preference. Report inter-judge agreement and human correlation, not a bare number.

## See also
- [[llm-as-a-judge]] — the evaluation paradigm these biases corrupt
- [[length-normalization-bias-control]] — the length-debiasing machinery (LC win rate)
- [[sycophancy]] — the bias channel by which judges echo stated/implied preferences
