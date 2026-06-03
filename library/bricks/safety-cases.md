# Safety Cases

**One-liner:** A structured, evidence-backed argument that deploying a specific model is acceptably safe — borrowed from aviation/nuclear/medical-device assurance — decomposing a top-level safety claim into sub-claims, each discharged by evidence (evals, red-teaming, monitoring) tied to pre-committed capability thresholds.

## The structure (claim-argument-evidence)

A safety case is a tree, not a checklist:

- **Top claim:** "Deploying model M in context C does not cause unacceptable harm of type H."
- **Argument:** a decomposition into sub-claims that, if all true, jointly establish the top claim (plus the reasons the decomposition is sound).
- **Evidence:** measurements discharging each leaf sub-claim — eval scores, red-team transcripts, training-process facts, monitoring guarantees.

Validity requires the argument be **deductively sound OR explicitly probabilistic**: if leaves are uncertain, the argument must state how residual risk aggregates (no hidden "and then a miracle occurs" step).

## The four argument types (Clymer et al. 2024)

1. **Inability** — the model is too weak to cause harm H (evidence: capability evals scoring below a danger threshold). Cheapest; expires as capabilities scale.
2. **Control** — even if capable + misaligned, deployment safeguards prevent harm (evidence: monitoring, sandboxing, red-team on the *control protocol*, not the model).
3. **Trustworthiness** — the model is capable but won't *try* to cause harm (evidence: alignment + interp; hardest to establish rigorously today).
4. **Deference** — defer to credible AI advisors that themselves have safety cases.

## Where it appears

- **Anthropic RSP / OpenAI Preparedness / DeepMind FSF** — capability thresholds (ASL-3, "High"/"Critical") that trigger the *requirement* to produce a safety case before deployment.
- **"Safety Cases" (Clymer, Gabriel, Krueger et al. 2024)** — the canonical inability/control/trustworthiness/deference taxonomy.
- **AI Control (Greenblatt et al. 2024)** — supplies the evidence backing a *control* safety case via red-teamed protocols robust to intentional subversion.
- **UK/US AISI frontier evaluations** — third-party evals feeding the evidence base.

## Common mistake

Treating a pile of passed evals as a safety case. A safety case is the *argument* connecting that evidence to the deployment claim — including why the evals are sufficient, what they assume, and what they fail to cover. Evals without an explicit argument (and explicit treatment of sandbagging / eval-gaming and residual risk) are evidence in search of a case, not a case.

## See also
- [[safety-evals]] — the dangerous-capability measurements that serve as leaf evidence
- [[automated-red-teaming]] — generates adversarial evidence for inability/control claims
- [[deceptive-alignment-scheming]] — the threat model trustworthiness cases must rule out (and that can corrupt the evidence)
