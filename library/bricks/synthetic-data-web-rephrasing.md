# Synthetic Data & Web Rephrasing (WRAP, Phi)

**One-liner:** Use an LLM to rewrite raw web text into clean styles (WRAP) or generate textbook-quality/math/code corpora from scratch (Phi), trading inference compute for higher-quality, less-noisy pretraining tokens that beat raw web at fixed token or compute budgets.

## The two recipes

**WRAP (Web Rephrase Augmented Pre-training, Maini et al. 2024):** prompt an instruction-tuned model to paraphrase each web doc into a target style — {easy/Wikipedia, medium, Q&A, conversational}. Train on a mix of real + rephrased:

L_pretrain = λ · L(D_web) + (1−λ) · L(rephrase(D_web)),  λ ≈ 0.5

Same factual content, denoised surface form → ~3x fewer tokens or ~10x less compute to hit a target perplexity; large gains on zero-shot QA.

**Phi / "textbooks are all you need" (Gunasekaran et al. 2023, Phi-1/1.5/2/3/4):** generate net-new synthetic data conditioned on seed topics/personas for diversity, filtered by a model-based quality classifier. Phi-4 (2024) is built on a majority of synthetic tokens (math, code, reasoning chains) plus heavily filtered web.

## The diversity problem

Synthetic data is generated i.i.d.-ish from one model → low entropy, repeated phrasings, narrow coverage. Fixes: persona-conditioning (Persona Hub, 1B personas), topic seeding, temperature/sampling diversity, multi-model generation, and rejection-filtering. Without diversity controls, you hit model collapse.

## Where it appears

- **WRAP** — rephrases C4/web into clean styles; faster convergence at fixed tokens
- **Phi-1 → Phi-4 (Microsoft)** — small models matching larger ones via synthetic textbook/code/math data + filtering
- **Llama 3, Qwen2.5, Nemotron** — synthetic data for math/code/long-context mid-training and instruction data
- **Cosmopedia / FineWeb-Edu (HF)** — open synthetic-textbook corpus + model-scored educational-quality filtering of web

## Common mistake

Believing the synthetic model injects *new* knowledge or capability beyond the teacher/web. WRAP-style rephrasing mainly denoises and reformats existing content (verifier-generator gap matters); recursively training on a model's own outputs without fresh real data shrinks distribution tails and degrades — model collapse. Synthetic data stretches and cleans real data; it does not manufacture information.

## See also
- [[model-collapse-curse-of-recursion]] — failure mode when synthetic data feeds back into training
- [[model-based-quality-filtering]] — the classifier step that gates both rephrased and generated tokens
- [[reasoning-distillation]] — generating CoT/reasoning traces is the reasoning-corpus flavor of this
