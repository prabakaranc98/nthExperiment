# SFT / Instruction Tuning

**One-liner:** Supervised fine-tuning on (instruction, response) demonstrations via next-token cross-entropy to teach format and behavior; stage zero of every post-training pipeline and the reference policy the KL anchors to.

## The objective

Standard causal-LM cross-entropy, but the loss is **masked to response tokens only**:

L(θ) = − E_{(x,y)~D} [ Σ_{t : token_t ∈ response} log π_θ(y_t | x, y_{<t}) ]

- x = prompt/instruction tokens (loss-masked), y = target response tokens (loss-active)
- Same teacher-forcing setup as pretraining; only the data distribution and the mask change
- Multi-turn: mask every turn except assistant tokens (or weight by turn)
- Typically 1–3 epochs, LR ~1e-5 to 2e-5 (full FT) on 10³–10⁶ curated examples; quality ≫ quantity (LIMA: ~1k examples suffice for format/style)

## Where it appears

- **InstructGPT / RLHF pipeline** — SFT is step 1; produces the policy π_SFT that PPO/DPO initializes from and that the KL penalty regularizes toward (π_ref = π_SFT).
- **DPO** — the reference model in the DPO loss is the SFT checkpoint; DPO is run on preference pairs *starting from* SFT.
- **Distillation / STaR / rejection sampling** — collect responses from a stronger model or best-of-n, filter, then SFT on them (a.k.a. behavior cloning / "SFT on rollouts").
- **Reasoning models** — long-CoT SFT cold-start before RLVR (e.g. DeepSeek-R1's cold-start SFT stage); Tülu, Zephyr, OLMo recipes.

## Common mistake

Computing loss over the prompt tokens too. You must mask the instruction so gradients only flow through response tokens — otherwise the model learns to *generate* instructions, dilutes the behavior signal, and degrades instruction-following. (Related trap: forgetting the chat template's special tokens, so train-time and inference-time formatting mismatch.)

## See also
- [[dpo]] — initialized from and references the SFT policy
- [[rlhf]] — SFT is stage 1; π_SFT becomes the reference policy
- [[kl-regularization-to-reference-policy]] — the KL anchor that pulls RL back toward π_SFT
