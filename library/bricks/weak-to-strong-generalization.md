# Weak-to-Strong Generalization

**One-liner:** Fine-tuning a strong pretrained model on labels from a *weaker* supervisor and having it generalize *beyond* the supervisor's accuracy — an empirical analogy for humans (weak) overseeing superhuman models (strong) under scalable oversight.

## The setup (Burns et al., OpenAI 2023)

1. Train a **weak supervisor** (e.g., GPT-2-size) on ground truth; freeze it.
2. Generate weak labels: f_weak(x) on a held-out set.
3. **Fine-tune the strong model** (e.g., GPT-4-size) on those weak labels — *not* ground truth.
4. Measure the strong student's accuracy on ground truth.

Key empirical finding: the strong student **outperforms its weak teacher**. It is *not* upper-bounded by weak-label accuracy.

## The metric: PGR

Performance Gap Recovered =

  PGR = (weak-to-strong − weak) / (strong-ceiling − weak)

where strong-ceiling = the strong model fine-tuned on ground truth. PGR=0 means the student just imitates the teacher; PGR=1 means it fully recovers the strong ceiling. OpenAI saw PGR ≈ 0.2–0.8 depending on task, often improvable with an **auxiliary confidence loss** that encourages the student to confidently disagree with weak labels:

  L = CE(f, weak_label) + α · CE(f, f_thresholded)  (student's own hardened predictions)

## Why it can work

The strong model already *knows* the task from pretraining; weak labels mainly **elicit** a latent capability rather than teach it. The student fits the weak teacher's *signal* but not its *errors* (which look like noise / are not robustly representable).

## Where it appears

- **OpenAI W2SG (Burns et al. 2023)** — the founding empirical paper; NLP, chess, reward modeling tasks.
- **Scalable oversight / superalignment** — proposed proxy for aligning superhuman models humans can't directly label.
- **Easy-to-hard generalization** — train on easy (cheaply labeled) data, evaluate on hard; a W2SG variant where "weak" = easy-domain supervision.
- **Reward modeling** — using a weaker RM to bootstrap a stronger policy/RM (relates to debate, RLHF).

## Common mistake

Believing the strong student exceeding the weak teacher proves alignment is solved. It does not: the student may also generalize the teacher's *biases*, and "elicitation" assumes the desired capability is already latent — it gives no guarantee for capabilities (like honesty under deception) that pretraining never instilled. PGR < 1 is the norm, not the exception.

## See also
- [[scalable-oversight]] — W2SG is the empirical testbed for the superhuman-oversight problem
- [[knowledge-distillation]] — structurally similar (teacher→student) but here student > teacher, reversing the usual gap
- [[reward-hacking-over-optimization]] — failure mode when the strong model exploits weak-label errors instead of generalizing past them
