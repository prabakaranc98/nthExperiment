# AI Safety via Debate

**One-liner:** A scalable-oversight protocol where two AIs argue opposing answers and a weaker (human or model) judge picks the winner — betting that exposing a lie is easier than telling one, so honesty is the equilibrium strategy even when the judge can't verify directly.

## The key insight

Frame oversight as a zero-sum game between two agents arguing for answers a₀, a₁ to question q. They alternate statements; a judge J sees the transcript and declares a winner. Reward is ±1 for win/loss.

The hope: at the optimal-play equilibrium, the truth-telling agent wins, so

  argmax over the debate game ≈ the answer a human would endorse with unlimited time.

Mechanism: a debater who lies creates an attack surface — the opponent recursively zooms in on the weakest sub-claim until reaching a leaf the judge *can* check. Asymmetry assumption: **defending a true claim is easier than defending a false one** (it's hard to lie consistently against a strong critic).

**Prover–Verifier Games (Anthropic/OpenAI framing):** generalize to a prover P producing a solution + proof and verifier V (the judge) accepting/rejecting. Train for *legibility* — P is rewarded only when V (and a sneaky adversarial prover) can correctly verify, pushing P toward checkable reasoning. Doubly-efficient debate adds complexity-theoretic guarantees (verification cost ≪ honest solution cost).

## Where it appears

- Irving, Christiano & Amodei (2018), "AI Safety via Debate" — original two-player MNIST/sparse-judge formulation
- OpenAI "Prover-Verifier Games improve legibility of LLM outputs" (2024) — checkability/legibility training on GSM8K
- Google DeepMind debate experiments (2024-25) — LLM debaters raise weak-judge accuracy on QuALITY/long-context QA; "doubly-efficient debate" theory
- Frontier scalable-oversight stacks — debate sits alongside RLHF/critique-models as a candidate for supervising superhuman models on hard-to-evaluate tasks

## Common mistake

Assuming the asymmetry ("truth is easier to defend") holds universally. It can fail: **obfuscated arguments** — a dishonest debater splits a false claim into many sub-claims each too costly for the honest side to fully refute within the budget, and persuasive-but-wrong rhetoric can beat a weak judge. Debate is a *conjecture* about an equilibrium, not a proof of safety; empirical judge-accuracy gains are modest and task-dependent.

## See also
- [[scalable-oversight]] — debate is one concrete instantiation of the broader problem
- [[weak-to-strong-generalization]] — sibling approach: weak supervisor eliciting a stronger model's knowledge
- [[generator-verifier-gap]] — debate exploits exactly this verify-is-easier-than-generate asymmetry
