# Automated Red-Teaming

**One-liner:** Use models, RL, or evolutionary search to automatically generate diverse adversarial prompts that elicit failures (harm, jailbreaks, leakage) at scale, replacing slow manual red-teaming — now a baseline pre-deployment requirement for frontier releases.

## The setup

Find inputs x that maximize a harm/failure signal from a target model p_θ, scored by a judge/classifier R (an LLM or safety classifier):

x* = argmax_x  R(x, p_θ(x))

Naive maximization mode-collapses to one exploit. The real objective is **coverage**: a diverse *set* of successful attacks across many harm categories. So the operative metrics are attack success rate (ASR) and a diversity term:

maximize  E_x[ASR(x)]   s.t.   diversity(attack set) high

**Three families:**
- **Red-team LM (Perez et al. 2022):** an attacker LLM (zero-shot / few-shot / SFT / RL) samples test cases; a red-team classifier scores target outputs; harmful cases are mined.
- **Attacker-policy RL:** train the attacker with PPO/GRPO, reward = judge(harm) + λ·diversity (e.g. novelty vs. prior attacks) − penalty for gibberish. Reward = ASR alone collapses.
- **Rainbow Teaming (Samvelyan et al. 2024):** MAP-Elites quality-diversity search over a descriptor grid (risk category × attack style); each cell keeps its highest-ASR prompt → a structured, diverse attack archive.

## Where it appears

- **Perez et al. 2022 "Red Teaming LMs with LMs"** — the founding recipe; attacker LM surfaces offensive text, leaked PII, distributional bias.
- **Rainbow Teaming / Ruby Teaming** — QD search producing diverse, mutated adversarial prompts; outputs double as safety-tuning data.
- **GCG / AdvPrompter / PAIR / TAP** — optimization- and LLM-driven jailbreak generators (gradient suffixes, attacker-LLM iterative refinement, tree-of-attacks).
- **Frontier safety pipelines (Anthropic, OpenAI, DeepMind, AISIs, 2024–26)** — automated + human red-teaming feeds RSP/Preparedness capability and propensity (ASR) evals before deployment.

## Common mistake

Optimizing the attacker for raw ASR only. It mode-collapses onto one cheap exploit (a single suffix or template), inflating numbers while leaving the threat surface unmapped. Without an explicit diversity/coverage objective and a judge robust to reward hacking, you measure how well you broke your own classifier, not the model's true vulnerability.

## See also
- [[jailbreaks-adversarial-prompts]] — the attack class automated red-teaming generates and stress-tests
- [[safety-evals]] — consumes red-team prompts to estimate ASR for release decisions
- [[refusal-safety-training]] — successful attacks become training data to patch the failures found
