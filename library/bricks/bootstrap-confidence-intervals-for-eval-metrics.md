# Bootstrap & Confidence Intervals for Eval Metrics

**One-liner:** Resample your eval set with replacement, recompute the metric each time, and read CIs off the resampled distribution — the cheapest way to know whether a benchmark gap is real or noise, and the single most-skipped step in LLM evaluation.

## The procedure

Given n examples with per-item scores (correct/incorrect, or a continuous metric):

```
for b in 1..B:                      # B ~ 1000-10000
    sample n items WITH replacement
    theta_b = metric(resample)      # accuracy, win-rate, BLEU, ...
CI_95 = [percentile(theta, 2.5), percentile(theta, 97.5)]   # percentile method
SE     = std(theta_*)
```

Resample the **unit of independence**: bootstrap items, not tokens; for win-rates bootstrap the prompts (cluster bootstrap if multiple judgments per prompt). For accuracy on n items the analytic SE is the same first-order object: SE = sqrt(p(1−p)/n) (Wald), so a 1000-item eval has SE ~1.5pp near 50% — a "2-point improvement" is inside the noise.

For paired model comparisons, bootstrap the **per-item difference** d_i = score_A,i − score_B,i; the CI on mean(d) is tighter than two independent CIs because it cancels item-difficulty variance.

## Where it appears

- **HELM / lm-evaluation-harness** — report bootstrap stderr on every metric; harness uses ~1000 resamples by default.
- **Chatbot Arena / LMSYS** — bootstrap over battles to put CIs on Elo/Bradley-Terry ratings; overlapping CIs => ranks not distinguishable.
- **AlpacaEval, MT-Bench, Arena-Hard** — bootstrap CIs on LLM-as-judge win-rates to separate real wins from judge noise.
- **GPT-4 / Llama / frontier model cards** — increasingly report CIs or note when SOTA claims are within error bars (cf. "Adding Error Bars to Evals", Miller 2024).

## Common mistake

Treating non-overlapping point estimates as significance, or overlapping CIs as non-significance. The correct test is a CI on the **paired difference** (or a permutation/paired bootstrap on d_i): two CIs can overlap while the paired difference CI excludes 0. Also: bootstrapping the wrong unit (tokens/sub-questions instead of independent items) understates variance and fabricates significance.

## See also
- [[a-b-testing-statistics]] — the hypothesis-testing frame for "is this gap real?"
- [[elo-online-rating-for-model-ranking]] — bootstrap CIs are how Arena ratings get error bars
- [[ppi]] — tighter valid CIs when mixing human labels with model-predicted labels
