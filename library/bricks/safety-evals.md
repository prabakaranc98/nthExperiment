# Safety & Dangerous-Capability Evals

**One-liner:** Empirical measurement of risky *capabilities* (CBRN/bio uplift, cyber-offense, autonomous replication, deception) and *propensities* (jailbreak susceptibility, refusal robustness) under adversarial red-teaming — the evidence base that gates a frontier model's release via Responsible Scaling Policies.

## The key insight (capability ≠ propensity)

Two distinct questions, measured differently:

- **Capability:** *Can* the model do the dangerous thing if it tries? Measure with elicitation maximized — best prompts, fine-tuning, scaffolding/agents, tools. You want an **upper bound**.
- **Propensity:** *Will* it do the harmful thing by default, or comply with attacks? Measure attack success rate (ASR) over a red-team / jailbreak distribution. You want a **lower bound** on safety.

Safety cases rely on the **inability argument**: deploy iff demonstrated capability stays below a danger threshold even under strong elicitation. So under-elicitation is a *failure of the eval*, not a safety result.

## The reporting metric

ASR = (1/N) Σ_i 𝟙[harmful_judge(response_i) = 1], with a 95% CI; report **best-of-k** attack success (k attempts), since attackers retry. Capability scored against a human-expert / uplift baseline:

uplift = (perf_with_model − perf_baseline) / (perf_expert − perf_baseline)

## Where it appears

- **Anthropic RSP / OpenAI Preparedness / DeepMind Frontier Safety Framework (2023–25)** — capability thresholds (ASL-3, "High"/"Critical") trigger mandated mitigations before deploy.
- **Apollo / METR (2024–25)** — autonomy & in-context scheming evals: agentic task suites, sandbagging and deceptive-alignment probes.
- **WMDP, HarmBench, AdvBench, StrongREJECT** — standardized hazardous-knowledge and jailbreak benchmarks with automated grader LLMs.
- **US/UK AI Safety Institutes** — pre-deployment third-party red-teaming of frontier releases.

## Common mistake

Treating a low default-refusal rate as evidence of safety. A model that refuses naive prompts but is broken by a 10-line jailbreak, fine-tuning, or agent scaffolding is *not* safe — capability evals must assume an adversary who maximizes elicitation. Weak elicitation produces a false negative, which is the most dangerous error here.

## See also
- [[alignment]] — evals measure whether alignment training actually held under pressure
- [[rlhf]] — refusal/harmlessness behavior these evals stress-test comes from RLHF/DPO
- [[conformal]] — when you need *guaranteed* error rates rather than point estimates of risk
