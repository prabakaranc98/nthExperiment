# cFM / Do-FM — source

Reproducible code for the Causal Foundation Model capstone. See [../capstone.md](../capstone.md) for the design.

## Run this to reproduce

```bash
# (planned)
python -m cfm.generate_scm   --config configs/phase0_linear.yaml   # synthetic SCM prior → sharded corpus
python -m cfm.train          --config configs/phase0.yaml          # pretrain the PFN
python -m cfm.eval           --bank data/heldout_scms --real ihdp,acic,lalonde,twins
```

## Planned layout

```
src/
├── cfm/
│   ├── scm/            # DAG sampler, mechanism/noise families, intervention & counterfactual oracle
│   ├── data/           # cell tokenization, dataset→grid encoder, sharded producer pool
│   ├── model/          # two-axis (sample/feature) attention, do-mask, Riemann/bar head
│   ├── train.py        # PFN posterior-predictive loss (WSD, bf16, FlashAttn, ckpt)
│   └── eval.py         # ATE/√PEHE, interventional NLL/CRPS, coverage + conformal wrapper
├── configs/            # phase0_linear → phase2_full
└── tests/
```

**Phase 0 first:** linear-Gaussian SCMs + a ~10–50M model; gate on beating a backdoor-regression baseline *and* 90% interval coverage in ~[0.86, 0.94] before enriching the prior.
