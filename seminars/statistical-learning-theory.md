# Statistical Learning Theory (+ High-Dimensional Statistics)

*Why does minimizing training error lower true error — and why does it sometimes refuse to? The mathematics of **generalization**: what makes learning possible, how much data it takes, and where the classical theory breaks for deep nets.*

**FAIRE context:** the theory spine of [501 Foundations](../faire-program.md) and [506 Applied Math](../faire-program.md); the *why* under the [Algorithms for AI/ML/DS](algorithms-for-ai-ml-ds.md) *what*, and the deep companion to [Statistical & Probabilistic Foundations](statistical-probabilistic-foundations.md). Cold lookup → the theory [bricks](../library/bricks/README.md) (bias-variance, double-descent, ntk, pac-bayes, implicit-bias, grokking).

---

## The one idea

A model that fits the training set tells you nothing on its own — *anything* fits a finite sample. Learning theory asks the only question that matters: **when does low empirical risk certify low true risk, and at what sample cost?** The answer is always a tension between **fit** (how well you match the data) and **complexity** (how much your hypothesis class could have fit *anything*). Every bound below is that tension made precise — and the deep-learning era is the story of the classical version of it *breaking*, then being rebuilt.

| Cluster | The question it answers |
|---|---|
| [The learning problem](#1-the-learning-problem) | What are we even minimizing, and against what? |
| [Complexity & uniform convergence](#2-complexity--uniform-convergence) | How "rich" is a hypothesis class — VC, Rademacher, covering? |
| [Generalization bounds](#3-generalization-bounds) | How much data to certify the gap is small? |
| [Kernels & RKHS](#4-kernels--rkhs) | Learning in infinite-dimensional feature spaces |
| [Margins, boosting & surrogate losses](#5-margins-boosting--surrogate-losses) | Why max-margin and convex surrogates work |
| [High-dimensional statistics & matrices](#6-high-dimensional-statistics--matrix-methods) | Estimation when `d ≳ n`: sparsity, low rank, random matrices |
| [Online learning & regret](#7-online-learning--regret) | Learning without an i.i.d. assumption |
| [Theory of deep learning](#8-the-theory-of-deep-learning-the-frontier) | Why overparameterized nets generalize anyway |

---

## 1. The learning problem

The setup everything is stated in. Get this precise and the rest is machinery.

| Concept | Statement | Why it matters |
|---|---|---|
| Risk vs. empirical risk | `R(h)=E[ℓ(h(x),y)]` vs. `R̂(h)=1/n Σ ℓ` | the gap `R−R̂` is the whole subject |
| ERM | `ĥ = argmin_{h∈H} R̂(h)` | what training *is*; the question is whether it generalizes |
| Realizable vs. agnostic | is there a perfect `h∈H`, or just a best one? | sets the achievable rate |
| Bias–complexity tradeoff | bigger `H` → lower approximation error, higher estimation error | the [bias-variance](../library/bricks/bias-variance.md) decomposition, formalized |
| No-Free-Lunch | no learner wins on *all* distributions | why inductive bias is mandatory |

## 2. Complexity & uniform convergence

The core classical engine: bound the *worst-case* gap over all `h∈H` at once.

| Measure | What it captures | Key result |
|---|---|---|
| **VC dimension** | largest set `H` can shatter | finite VC ⇒ PAC-learnable; sample `~ VC/ε²` |
| Growth function / Sauer's lemma | # labelings `H` realizes | polynomial in `n` once `n>VC` |
| **Rademacher complexity** | can `H` fit random ±1 noise? | data-dependent, tighter, the modern tool |
| Covering numbers / metric entropy | `ε`-net size of `H` | **Dudley's entropy integral**; chaining |
| Fat-shattering / pseudo-dimension | scale-sensitive capacity | real-valued / regression analogues |

## 3. Generalization bounds

Turning a complexity measure into "with prob ≥ 1−δ, the gap ≤ …".

| Bound type | Complexity term | Note |
|---|---|---|
| PAC / agnostic PAC | VC / Rademacher | the canonical `O(√(C/n))` |
| Margin bounds | margin-normalized complexity | large margin ⇒ small effective capacity (SVM, boosting) |
| **PAC-Bayes** | `KL(Q‖P)` posterior↔prior | the only **non-vacuous** bounds for real nets ([pac-bayes](../library/bricks/pac-bayes.md)) |
| Stability | algorithmic sensitivity to one point | bounds *the algorithm*, not the class (SGD, ridge) |
| Compression / MDL | description length | "if you can compress it, it generalizes" |
| Information-theoretic | `I(weights; data)` | mutual-information bounds (Xu–Raginsky, Russo–Zou) |

## 4. Kernels & RKHS

Learning in (possibly infinite-dimensional) feature spaces — and the bridge to deep nets.

- **Kernel trick** — replace `⟨φ(x),φ(x')⟩` with `K(x,x')`; learn nonlinearly with linear machinery.
- **RKHS & the representer theorem** — the optimum is a finite combination of training kernels, so infinite features stay tractable.
- **SVMs** — max-margin classification; the canonical margin-theory success.
- **Gaussian processes** — the Bayesian dual of kernel regression; calibrated uncertainty.
- **The NTK connection** — an infinitely-wide net trained by GD *is* kernel regression with the [neural tangent kernel](../library/bricks/ntk.md); the formal bridge from kernels to deep learning (and where it stops).

## 5. Margins, boosting & surrogate losses

Why we optimize convex proxies for the 0–1 loss — and when that's safe.

- **Margin theory** — generalization scales with the *margin*, not the parameter count; explains why max-margin (SVM) and boosting resist overfitting.
- **AdaBoost** — coordinate descent on exponential loss; the margin-maximization story.
- **Surrogate-loss consistency** (Bartlett–Jordan–McAuliffe) — *which* convex surrogates (logistic, hinge, exponential) are **classification-calibrated**, i.e. minimizing them actually minimizes 0–1 risk. The theory under [cross-entropy](../library/bricks/cross-entropy.md) training.

## 6. High-dimensional statistics & matrix methods

Estimation when the dimension rivals or exceeds the sample size (`d ≳ n`) — the regime of modern data. *(Your "high-dimensional matrix" + the statistical core of "data/stats/probability".)*

| Topic | Key results | Where it shows up |
|---|---|---|
| Concentration in high-D | sub-Gaussian / sub-exponential tails, Bernstein | the toolkit for all of the below |
| **Matrix concentration** | Matrix Bernstein / Chernoff (Tropp) | error bounds for [RandNLA](algorithms-for-ai-ml-ds.md) & covariance estimation |
| Sparsity & the **LASSO** | ℓ₁ recovery, restricted eigenvalue, oracle inequalities | feature selection, interpretable high-D regression |
| **Compressed sensing** | RIP, ℓ₁ = ℓ₀ recovery | sample-efficient sensing; the sparsity miracle |
| Low-rank recovery | matrix completion, nuclear-norm minimization | recommender systems, robust PCA |
| **Random matrix theory** | Marchenko–Pastur, spectra of large matrices | understanding wide-net Hessians, covariance, [double descent](../library/bricks/double-descent.md) |
| Minimax rates | information-theoretic lower bounds (Fano) | the best *any* estimator can do |

## 7. Online learning & regret

Drop the i.i.d. assumption: learn against an adversarial sequence, measured by **regret** vs. the best fixed hypothesis in hindsight.

- **Online convex optimization** — online gradient descent; `O(√T)` regret.
- **Multiplicative weights / Hedge** — the expert-aggregation workhorse (also boosting, game theory).
- **Online-to-batch** — convert a regret bound into a generalization bound for free.
- **Bandits** — regret under partial feedback (explore/exploit; ties to RL and [active learning](algorithms-for-ai-ml-ds.md)).

## 8. The theory of deep learning (the frontier)

Where classical SLT *breaks* — and the live research rebuilding it. This is the part that matters most for frontier work.

| Phenomenon | What it breaks | Current understanding |
|---|---|---|
| **Interpolation generalizes** | "fitting noise ⇒ overfit" | [double descent](../library/bricks/double-descent.md), benign overfitting (Bartlett) |
| Uniform convergence may fail | the whole §2 program | Nagarajan–Kolter: UC can be provably unable to explain it |
| **NTK vs. feature learning** | fixed-kernel view | lazy ([ntk](../library/bricks/ntk.md)) vs. feature-learning (μP) regimes |
| **Implicit bias** | "many minima, which one?" | GD picks min-norm / max-margin ([implicit-bias](../library/bricks/implicit-bias.md)) |
| **Scaling laws** | finite-sample bounds | generalization as a measured power law ([scaling-laws](../library/bricks/scaling-laws.md)) |
| **Grokking** | monotone learning curves | delayed generalization / phase transitions ([grokking](../library/bricks/grokking.md)) |

The throughline: classical SLT bounds *the class*; modern deep learning is explained by the *implicit bias of (architecture, optimizer, data spectrum)* selecting a low-complexity solution — best captured by PAC-Bayes / flatness / scaling laws, not VC.

---

## How to study this

Theory you can't *use* is trivia — so for each bound, ask "what does it predict, and can I see it in an experiment?" (the [501](../faire-program.md) rigor applies to theory too).

1. **§1–3 first** — the ERM → complexity → bound chain is the spine; everything else specializes it.
2. **§4–5** — kernels and margins, the classical successes (and the NTK bridge forward).
3. **§6** — high-dimensional statistics; this is where theory meets real data (`d ≳ n`).
4. **§7** — online learning, for the non-i.i.d. world (and the RL bridge).
5. **§8 last** — the deep-learning frontier only makes sense *against* the classical theory it breaks. Re-read the theory bricks here.

**Highest-leverage texts:** Shalev-Shwartz & Ben-David ([*Understanding Machine Learning*](../library/reads-and-references/00-books-the-canon.md) 📖); Mohri, Rostamizadeh & Talwalkar (*Foundations of ML*); Wainwright (*High-Dimensional Statistics*); Vershynin (*High-Dimensional Probability* 📖); Bach (*Learning Theory from First Principles*); Telgarsky (deep-learning-theory notes). Full list → [The Canon](../library/reads-and-references/00-books-the-canon.md).

## See also

- [Algorithms for AI/ML/DS](algorithms-for-ai-ml-ds.md) — the algorithmic *what* to this theoretical *why* (concentration, RandNLA, sketching)
- [Statistical & Probabilistic Foundations](statistical-probabilistic-foundations.md) — inference, modeling, and the probability backing
- [APSL — Algorithmic Probabilistic Structure Learning](apsl-structure-learning.md) — PAC-Bayes / amortized-inference deep dive
- [Concept library (bricks)](../library/bricks/README.md) — bias-variance · double-descent · pac-bayes · ntk · implicit-bias · grokking · scaling-laws
