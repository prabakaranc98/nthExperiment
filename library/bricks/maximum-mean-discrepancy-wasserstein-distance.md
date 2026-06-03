# Maximum Mean Discrepancy / Wasserstein Distance

**One-liner:** Two distribution distances that, unlike KL, stay finite and informative for disjoint-support distributions — MMD is a kernel two-sample statistic, Wasserstein is optimal-transport "earth-mover" cost; both power WGAN, drift detection, and generative eval.

## The definitions

**MMD** (witness via RKHS feature mean): for kernel k with feature map φ,
MMD²(P,Q) = ‖E_P[φ(x)] − E_Q[φ(y)]‖²_H = E[k(x,x′)] − 2E[k(x,y)] + E[k(y,y′)].
Unbiased U-statistic estimator over samples; MMD=0 iff P=Q for a *characteristic* kernel (e.g. Gaussian RBF). Cost O(n²) per pair.

**Wasserstein-p** (optimal transport):
W_p(P,Q) = ( inf_{γ∈Π(P,Q)} E_{(x,y)~γ}[‖x−y‖^p] )^{1/p},
where Π is the set of couplings with marginals P,Q. **Kantorovich–Rubinstein dual** for p=1:
W₁(P,Q) = sup_{‖f‖_L≤1} E_P[f] − E_Q[f]  (supremum over 1-Lipschitz f).

Key property both share: a true *metric* on distributions; finite and smooth even when supports do not overlap (where KL = ∞ and JS = const).

## Where it appears

- **WGAN / WGAN-GP** (Arjovsky 2017, Gulrajani 2017) — critic approximates the W₁ dual f; Lipschitz enforced by gradient penalty, fixing vanishing-gradient pathology of JS-based GANs.
- **FID / KID** — FID is a Fréchet (W₂ Gaussian) distance on Inception features; KID is squared-MMD on the same features (unbiased, no Gaussian assumption).
- **Drift / two-sample detection** — MMD as a kernel test for distribution shift; sliced/entropic-OT distances monitor embedding drift in production.
- **Sinkhorn / entropic OT** — adds entropy regularization → fast GPU-friendly W approximation; flow matching and Schrödinger-bridge training lean on OT couplings.
- **Domain adaptation** — MMD-minimization aligns source/target feature distributions (DAN, deep CORAL kin).

## Common mistake

Treating them as interchangeable with KL or each other. KL is asymmetric, mode-covering, and infinite under disjoint support; W and MMD are symmetric metrics that *do* see geometry between non-overlapping supports. Also: WGAN works only if the Lipschitz constraint is actually enforced — weight clipping/GP/spectral norm — otherwise the critic is not estimating W₁ at all. And FID being a Wasserstein distance does not make it assumption-free: it fits a single Gaussian to features.

## See also
- [[kl-divergence]] — the asymmetric, support-sensitive divergence MMD/Wasserstein are chosen to replace
- [[gan-objective-adversarial-losses]] — WGAN's critic is the W₁ dual; OT reframes the adversarial game
- [[flow-matching]] — optimal-transport couplings define the target velocity field
