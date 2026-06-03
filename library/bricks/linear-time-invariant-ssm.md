# Linear-Time-Invariant SSM (S4 / HiPPO / DPLR)

**One-liner:** A continuous linear ODE x'(t)=Ax+Bu, y=Cx discretized into a fixed-kernel sequence model whose A is initialized by HiPPO (optimal polynomial memory) and structured as diagonal (S4D) or diagonal-plus-low-rank (DPLR) so the recurrence can be unrolled as a long convolution in O(N log N) for stable long-range memory.

## The formula / definition

Continuous: x'(t) = A x(t) + B u(t),  y(t) = C x(t)  (+ Du skip).

Discretize with step Δ (bilinear/ZOH):
  Ā = (I − Δ/2·A)^{−1}(I + Δ/2·A),  B̄ = (I − Δ/2·A)^{−1} Δ B.

Recurrent form: x_k = Ā x_{k−1} + B̄ u_k,  y_k = C x_k.
Equivalent convolution: y = K * u with kernel K = (C B̄, C Ā B̄, C Ā² B̄, …, C Ā^{L−1} B̄).

**LTI** = A,B,C,Δ are *fixed across time* → kernel K is precomputable, enabling FFT convolution at train time and a constant-memory recurrence at inference. (Mamba breaks LTI by making B,C,Δ input-dependent — selective SSM.)

**HiPPO-LegS** sets A_{nk} = −√((2n+1)(2k+1)) if n>k, −(n+1) if n=k, else 0 — the operator that projects history onto Legendre polynomials, giving provably optimal compression of the past.

**DPLR / NPLR (S4):** HiPPO matrices are normal-plus-low-rank; conjugate to A = Λ − P Q* (diagonal Λ minus rank-1). The kernel reduces to a Cauchy-kernel evaluation, computed via the generating function over roots of unity in O(N+L). **S4D** drops the low-rank term, keeping just diagonal Λ initialized from HiPPO eigenvalues — nearly matches S4 and is far simpler.

## Where it appears

- **S4 (Gu et al. 2022)** — DPLR + HiPPO; first SSM to crack Long Range Arena (Path-X) where Transformers failed.
- **S4D / DSS** — diagonal-only variant; the practical default initialization for diagonal SSMs.
- **H3, Hyena, S5** — LTI SSM layers as the long-range mixing primitive in language/genomics stacks.
- **Mamba / Mamba-2 & hybrids (Jamba, Zamba)** — start from this LTI backbone, then add input-dependent selection; understanding LTI is the prerequisite.

## Common mistake

Conflating the SSM with Mamba's *selectivity*. Plain S4/S4D is strictly LTI — its kernel is content-independent, so it cannot do data-dependent gating or content-based copying (it fails associative recall). The convolution view only exists *because* it's LTI; the moment A/B/C/Δ depend on the input you lose the global FFT and must use a parallel scan instead.

## See also
- [[selective-state-space-models-mamba]] — drops LTI by making parameters input-dependent
- [[recurrence-convolution-scan-duality]] — why the same SSM has recurrent, convolutional, and scan forms
- [[hardware-aware-parallel-scan]] — the kernel that replaces FFT convolution once parameters vary in time
