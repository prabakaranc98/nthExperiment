# Causal Foundation Model (cFM / Do-FM) — Report

*Public-ready write-up. Fill in as the build progresses; the spec lives in [capstone.md](capstone.md).*

---

## Abstract

*(1 paragraph: a 300–500M PFN over synthetic SCMs that answers interventional / counterfactual / effect-estimation queries in-context, with calibrated, identifiability-aware uncertainty. Headline result here.)*

## 1. Motivation

*(Why amortized in-context causal inference; the gap above Do-PFN / CausalPFN.)*

## 2. Method

### 2.1 Architecture (Do-FM)
### 2.2 The synthetic SCM prior
### 2.3 Training objective & calibration

## 3. Experiments

### 3.1 Held-out synthetic SCMs (ATE error, √PEHE, interventional NLL, coverage)
### 3.2 Real benchmarks (IHDP, ACIC, Lalonde/Jobs, Twins)
### 3.3 Ablations (prior richness · confounding · context size · do-mask)

## 4. Results

*(Tables with bootstrap CIs. Calibration / reliability diagrams.)*

## 5. Limitations

*(Prior misspecification & sim-to-real; counterfactual fragility; identifiability bounds.)*

## 6. Conclusion & next steps

*(Path to the virtual-cell stretch — see [AIDO / digital-organism note](../../seminars/foundation-models/12-virtual-cells/aido-digital-organism.md).)*

## References

*(Do-PFN, CausalPFN, CausalFM, TabPFN v2, PFNs — see capstone.md.)*
