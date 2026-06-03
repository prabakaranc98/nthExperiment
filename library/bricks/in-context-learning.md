# In-Context Learning

**One-liner:** LLMs solve new tasks from prompt examples (x₁,y₁,…,x_k,y_k, x_query) without any weight update — interpretable as implicit Bayesian inference over a latent task, or as the forward pass implementing an optimizer over the demonstrations.

## The setup

Given a prompt of k demonstrations and a query, the frozen model predicts:

  p(y | x_query, x₁,y₁,…,x_k,y_k ; θ)

θ is fixed. No gradients, no fine-tuning. Performance improves with k (more shots) and with model scale — ICL is largely an emergent property of large pretrained transformers.

## The two main theoretical accounts

**(1) Bayesian / latent-task view (Xie et al. 2022).** Pretraining on documents with latent structure makes the model an implicit Bayesian: each demonstration sharpens a posterior over a latent concept θ_task, and the model marginalizes:

  p(y|x_q, D) = ∫ p(y|x_q, θ_task) p(θ_task | D) dθ_task

Demonstrations select the task; the *format/distribution* of inputs matters more than label correctness (Min et al. 2022 — randomizing labels barely hurts).

**(2) Implicit meta-optimization view (von Oswald 2023, Akyürek 2023, Dai 2023).** A transformer's forward pass can *implement* gradient descent / ridge regression on the in-context examples. For linear regression, trained transformers provably match the least-squares / Bayes-optimal predictor; one attention layer ≈ one GD step. ICL ≈ "mesa-optimization" inside the forward pass.

## Where it appears

- GPT-3 (Brown et al. 2020) — the paper that named few-shot ICL as the headline capability
- Chain-of-thought — few-shot CoT exemplars are ICL that elicits multi-step reasoning
- Induction heads (Olsson et al. 2022) — the mechanistic circuit (prefix-match → copy) that correlates with ICL onset during training
- Long-context / many-shot ICL (Agarwal et al. 2024) — hundreds–thousands of examples can rival fine-tuning on some tasks
- RAG and tool use — retrieved context is consumed via the same in-context mechanism

## Common mistake

Believing ICL "learns" the input→output mapping from the demonstrations. Often it mostly *locates* a task the model already knows: label correctness can be largely irrelevant, while the label space, input distribution, and format do the work. So strong few-shot accuracy is not evidence the model induced the rule from your examples.

## See also
- [[chain-of-thought]] — CoT is ICL specialized to elicit intermediate reasoning steps
- [[ntk]] — links the forward-pass-as-optimizer view to feature-learning / kernel regimes
- [[scaling-laws]] — ICL ability emerges and sharpens with model and data scale
