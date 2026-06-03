# Jailbreaks & Adversarial Prompts

**One-liner:** Inputs crafted to bypass safety alignment and elicit refused behavior — spanning hand-written roleplay/persona attacks, optimization-found adversarial suffixes (GCG), and many-shot in-context attacks; the standard adversarial threat model for aligned LLMs.

## The formula / definition

A jailbreak finds an input x that maximizes the probability of a harmful target response y* (e.g. starting "Sure, here is how...") under the aligned model p_θ, while passing through (or circumventing) the model's refusal behavior:

  maximize_x  log p_θ(y* | x)   s.t. x contains/embeds the harmful request

**GCG (Greedy Coordinate Gradient, Zou et al. 2023)** — optimize an adversarial suffix appended to the prompt. At each step, use the gradient of the loss w.r.t. one-hot token indicators to rank candidate token swaps per position, then greedily evaluate a batch and keep the best:

  L(x) = − Σ_t log p_θ(y*_t | x_{1:t-1})        (target = affirmative response prefix)
  for each position i: pick top-k tokens by −∇_{e_i} L, sample candidates, take argmin L

Suffixes transfer across models (trained on open-weights, attack closed APIs). **Many-shot jailbreaking (Anthropic 2024)** — fill a long context with hundreds of faux dialogues where the assistant complies with harmful requests; attack success rises log-linearly with shot count, exploiting in-context learning and long context windows.

## Where it appears

- **GCG / AdvBench** (Zou et al. 2023) — gradient-based universal+transferable suffixes; the canonical white-box automated attack and a standard robustness benchmark.
- **Many-shot jailbreaking** (Anthropic 2024) — scales with context length; longer-context frontier models are *more* vulnerable absent mitigation.
- **Roleplay / persona attacks** ("DAN", crescendo, "grandma exploit") — hand-written social-engineering that reframes the request so refusal training doesn't trigger.
- **Frontier red-teaming & safety pipelines** — GPT-4/Claude/Gemini system cards report jailbreak robustness; PAIR/TAP use an attacker LLM to auto-generate jailbreaks; constitutional classifiers (Anthropic 2025) wrap I/O to block universal jailbreaks.

## Common mistake

Believing a model is "safe" because it refuses the direct request. Safety training shifts the *refusal trigger*, it does not remove the underlying capability — the harmful knowledge still lives in the weights. Jailbreaks just route around the trigger (suffix, persona, context), so refusal is a shallow behavior, not a guarantee.

## See also
- [[refusal-safety-training]] — the safety behavior that jailbreaks are designed to defeat
- [[prompt-injection]] — adversarial inputs targeting tool/agent control flow rather than safety policy
- [[automated-red-teaming]] — systematically generating jailbreaks at scale to find failures pre-deployment
