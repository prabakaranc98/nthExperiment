# Rejection Sampling / Best-of-N

**One-liner:** Sample N candidates from a policy, score each with a reward model or verifier, keep the top one(s); used at inference (BoN) or to filter SFT data for offline distillation (RFT / STaR / expert iteration).

## The procedure

Draw y₁,...,y_N ~ π(·|x), score with reward/verifier r(x,y), then:

- **Best-of-N (inference):** return y* = argmaxᵢ r(x, yᵢ). A pure test-time-compute knob — no weight updates.
- **Rejection sampling FT / RFT (training):** keep {(x, yᵢ) : yᵢ passes verifier or r above threshold}, SFT the base model on those filtered samples. Iterate (expert iteration / STaR): the FT'd model becomes the new sampler.

BoN is a soft-argmax over samples that **provably maximizes E[r] subject to a KL budget** — it implicitly targets the tilted distribution π(y|x)·exp(r/β)/Z. The induced KL is bounded: KL(BoN ‖ π) ≈ log N − (N−1)/N, so reward gains plateau and over-optimization eventually hurts (Goodhart on the proxy r).

## Where it appears

- **WebGPT / Llama 2** — BoN reranking with the RM at inference; Llama 2 also used rejection-sampling FT as a stage before PPO.
- **STaR / RAFT / ReST / expert iteration** — bootstrap reasoning: sample CoT, keep traces with correct final answer, SFT, repeat.
- **Test-time scaling (2024+)** — BoN + verifier is the simplest scaling axis; compute-optimal work (Snell et al.) shows when BoN beats sequential revision vs. just using a bigger model.
- **RLVR / o1-style pipelines** — rejection sampling on verifiable rewards (math/code unit tests) is the cheap, stable alternative to full RL.

## Common mistake

Treating BoN as free accuracy. It is bounded by **coverage** (pass@N — can the policy even sample a correct answer?) and by **verifier quality** — a noisy RM gets gamed as N grows (reward hacking), so reward improves while true quality drops. BoN-as-distillation also amplifies the policy's own biases since you only train on its own filtered samples.

## See also
- [[rlvr]] — verifiable rewards make rejection sampling exact and game-resistant
- [[self-improvement-star-bootstrapping]] — STaR is iterated rejection-sampling FT
- [[reward-hacking-over-optimization]] — the KL-bounded reason BoN gains plateau
