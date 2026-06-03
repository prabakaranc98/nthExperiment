# Adapters & Prefix/Prompt Tuning

**One-liner:** The pre-LoRA PEFT family — freeze base weights and train either small bottleneck modules inserted between layers (adapters) or learned virtual tokens / KV vectors prepended to the sequence (prefix/prompt tuning), or per-channel rescalers (IA³).

## The variants (the math)

Freeze W₀ everywhere; train only the added parameters.

- **Adapters** (Houlsby 2019): per layer insert `h ← h + W_up · σ(W_down · h)`, with `W_down ∈ ℝ^{d×r}`, `W_up ∈ ℝ^{r×d}`, r << d, near-zero init. Adds a residual bottleneck after attention and/or FFN — **sequential, so it adds inference latency.** Pfeiffer/Parallel adapters reduce to one insertion or run alongside.
- **Prompt tuning** (Lester 2021): prepend P learned embedding vectors to the *input* only: `[soft_tokens; embed(x)]`. Just P·d params; everything else frozen.
- **Prefix tuning** (Li & Liang 2021): prepend P learned key/value vectors at *every* attention layer (`K=[K_pre; K]`, `V=[V_pre; V]`); reparameterized through an MLP during training for stability.
- **IA³** (Liu 2022): scale activations by learned vectors — `K ← l_k ⊙ K`, `V ← l_v ⊙ V`, FFN `← l_ff ⊙ act`. Tiny (3 vectors/layer); element-wise, can be merged into weights.

## Where it appears

- **AdapterHub / adapter-transformers** — modular per-task adapters, composed via stacking/fusion.
- **P-Tuning v2 / prefix tuning** — the soft-prompt route, now mostly the conceptual ancestor of **soft prompts** in continuous/latent reasoning and **prefix caching**.
- **IA³ / (IA)³ in T-Few** — strong few-shot PEFT; ~0.01% params, often beats in-context learning at lower cost.
- **LoRA superseded most of these** for LLMs (zero added latency once merged), but adapters/IA³ persist in multimodal connectors and multi-task serving.

## Common mistake

Conflating soft *prompt tuning* (P trainable input embeddings, attention layers untouched) with *prefix tuning* (trainable KV at every layer). Also: assuming adapters are latency-free like LoRA — sequential bottleneck adapters cannot be folded into W₀ and do add per-token compute, unlike IA³ scalers or merged LoRA.

## See also
- [[lora]] — the low-rank successor; merges into weights with no inference overhead
- [[latent-continuous-reasoning]] — soft prompts as continuous tokens generalize prompt tuning
- [[prefix-caching-radixattention]] — prefix tuning's learned KV vs. cached KV for shared prefixes
