# Decoding / Sampling Strategies (Greedy, Beam, Top-k, Top-p, Min-p)

**One-liner:** How a token is actually picked from the next-token distribution at inference — deterministic (greedy/beam) vs. truncated stochastic (top-k, top-p/nucleus, min-p) sampling, all gated by a temperature that rescales logits before softmax.

## The formula / definition

Logits z → temperature → softmax → truncate → renormalize → sample.

p_i = softmax(z_i / T):  T→0 = argmax (greedy), T=1 = unscaled, T>1 flattens, T<1 sharpens.

- **Greedy:** x_t = argmax_i p_i. Deterministic, cheap, repetitive.
- **Beam search (width B):** keep the B highest-log-prob *sequences*, expand all, prune to top B by Σ log p. Maximizes sequence likelihood; standard for MT/summarization, rarely used for open-ended chat (bland, degenerate).
- **Top-k:** keep the k highest-prob tokens, zero the rest, renormalize, sample. (Fan et al. 2018)
- **Top-p / nucleus:** keep the smallest set V_p with Σ_{i∈V_p} p_i ≥ p; renormalize, sample. Adaptive support size. (Holtzman et al. 2020, "Curious Case of Neural Text Degeneration")
- **Min-p:** keep tokens with p_i ≥ p_min · max_j p_j — threshold scales with the peak. Sharp distribution → tight set; flat → permissive. Robust at high T. (Nguyen et al. 2024)

Typical chat defaults: T ≈ 0.7–1.0 with top-p ≈ 0.9–0.95, or min-p ≈ 0.05–0.1.

## Where it appears

- **vLLM / TGI / SGLang `SamplingParams`** — temperature, top_k, top_p, min_p, repetition/presence penalty are the exposed inference knobs; combined filters apply in order top_k → top_p → min_p.
- **Reasoning models (o-series, R1, Qwen3)** — T≈0.6, top_p≈0.95 for long CoT; greedy collapses chains and hurts pass@k diversity needed for self-consistency / best-of-n.
- **GRPO / RLVR rollouts** — sampling temperature controls exploration; entropy of the sampled distribution is monitored to avoid collapse.
- **Speculative decoding** — the draft/target acceptance test must reproduce the *exact* target sampling distribution under the chosen temperature/filters.

## Common mistake

Treating temperature and truncation as the same dial. Temperature *reshapes* the whole distribution (and applies before truncation); top-p/top-k/min-p *clip the tail* without reshaping the survivors' relative odds. Also: applying truncation before temperature, or thinking top-p=1.0 + T=0 is "random" — it's still greedy.

## See also
- [[softmax]] — the distribution all these strategies sample from
- [[temperature-scaling]] — same T, but for calibration vs. generation diversity
- [[pass-k-self-consistency-estimation]] — why diverse sampling beats greedy for reasoning
