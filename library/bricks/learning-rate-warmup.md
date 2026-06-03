# Learning-Rate Warmup

**One-liner:** Linearly ramp the LR from ~0 to its peak over the first W steps to avoid early divergence from large, poorly-conditioned updates and to let Adam's second-moment estimate accumulate before full-size steps land.

## The formula

Linear warmup to peak η_max over W warmup steps, then hand off to the main schedule (cosine, WSD, etc.):

η(t) = η_max · t / W,                  for t ≤ W
η(t) = decay_schedule(t),              for t > W

Common defaults: W = a few hundred to a few thousand steps, or a fixed fraction (~1–5%) of total steps. The classic Transformer (Vaswani 2017) used `η ∝ d_model^{-0.5} · min(t^{-0.5}, t · W^{-1.5})` with W = 4000 — linear up, inverse-sqrt down.

## Why it works

Two distinct mechanisms, often conflated:
- **Adam variance warmup:** early on, the second-moment v_t is estimated from few samples and is noisy; the bias-corrected step `η · m̂/(√v̂+ε)` can be huge and erratic. Warmup gives v_t time to stabilize. RAdam (2020) showed this and proposed a rectification term as a *warmup-free* substitute.
- **Sharpness / curvature:** at init the loss landscape is sharp and ill-conditioned; large steps overshoot. Warmup keeps early steps small until the trajectory reaches a flatter region, lowering the operative sharpness (connects to edge-of-stability dynamics).

## Where it appears

- **Every LLM pretraining run** — GPT-3, Llama, Chinchilla, etc. all warm up before cosine/WSD decay; standard in nanoGPT and Megatron defaults.
- **Adam/AdamW + large batch** — near-mandatory; the larger the batch and peak LR, the longer the warmup needed (LARS/LAMB pair warmup with layer-wise scaling).
- **Warmup-Stable-Decay (WSD)** — warmup is the explicit first phase before the long constant-LR plateau.
- **Fine-tuning / RLHF** — short warmups (often a few % of steps) to avoid clobbering pretrained weights with a cold large step.

## Common mistake

Treating warmup length as a fixed constant rather than scaling it with batch size and peak LR. Larger batch / higher η_max needs more warmup; too-short warmup is a leading cause of early loss spikes. Conversely, warmup is not a cure-all — with QK-norm, z-loss, and good init you can often shorten it, and it never compensates for a peak LR that is simply too high.

## See also
- [[cosine-lr-decay]] — the schedule warmup typically hands off to
- [[warmup-stable-decay-schedule]] — modern schedule where warmup is phase one
- [[loss-spikes-training-instability]] — what insufficient warmup tends to cause
