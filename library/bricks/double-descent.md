# Double Descent

**One-liner:** Test error falls, then rises to a peak exactly at the interpolation threshold (params ≈ training points), then falls again as the model grows further over-parameterized — reconciling the classical U-shaped bias-variance curve with deep nets that generalize despite zero training error.

## The shape

Plot test error vs. model capacity (width, params, or epochs):

```
test
err |   classical U      modern descent
    |      /\
    |     /  \___/\          <- PEAK at interpolation threshold
    |  __/         \____________
    |________________________________ capacity
       underparam  | N≈P |  overparam
```

The **interpolation threshold** is where capacity first suffices to fit the training set exactly (training error → 0). Belkin et al. (2019) showed the classical regime is just the *left half*; pushing past the threshold enters a second descent where larger models generalize *better*, not worse.

## Why the peak

At the threshold there is exactly one (or few) interpolating solution(s), so the fit is brittle — tiny label noise forces huge-norm, high-variance interpolants → variance explodes. Past it, many interpolating solutions exist and the optimizer's implicit bias selects the min-norm / flattest one, shrinking effective complexity. The relevant axis is the norm of the solution, not raw parameter count.

## Variants

- **Model-wise** — vary width/params (the canonical curve).
- **Epoch-wise** — fix the model, vary training time; test error can rise then fall mid-training (Nakkiran et al., 2020).
- **Sample-wise** — counterintuitively, *more data* can hurt near the threshold by shifting it into your operating regime.

## Where it appears

- **Belkin et al. (2019), "Reconciling modern ML practice"** — coined the curve; demonstrated on RFF, trees, small nets.
- **Nakkiran et al. (2020), "Deep Double Descent"** — ResNets/Transformers; introduced *effective model complexity* and showed epoch-wise + label-noise dependence.
- **Scaling-law practice** — overparameterized frontier LLMs sit deep in the *second* descent; explains why "too big to overfit" works and why early-stopping intuitions can mislead.
- **Random feature / NTK theory** — gives closed-form double-descent risk curves, linking it to the lazy-training regime.

## Common mistake

Thinking double descent contradicts bias-variance. It doesn't — it *extends* it. The classical U is real in the underparameterized regime; the second descent appears only past interpolation and is driven by the optimizer's implicit norm-minimizing bias, which the textbook count-the-parameters complexity measure ignores. Also: the peak is most pronounced with label noise and weak/absent regularization — strong regularization can erase it entirely.

## See also
- [[bias-variance]] — the classical U-curve that double descent extends past interpolation
- [[implicit-bias]] — min-norm solution selection drives the second descent
- [[grokking]] — related late-time generalization phenomenon in over-parameterized nets
