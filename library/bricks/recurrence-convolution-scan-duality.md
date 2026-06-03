# Recurrence-Convolution-Scan Duality

**One-liner:** A linear recurrence h_t = A h_{t-1} + B x_t can be unrolled into a convolution (a fixed kernel over the whole sequence) or computed by an associative scan — so the same layer trains in parallel (O(L log L) or chunked O(L)) and runs at inference as an O(1)-memory, O(L) sequential recurrence.

## The three equivalent forms

Take a linear (input-affine) recurrence with state h_t ∈ R^N:

  h_t = A h_{t-1} + B x_t,   y_t = C h_t

1. **Recurrence (inference):** step one token at a time. O(L) time, O(N) memory — no growing KV cache. This is the serving mode.

2. **Global convolution (training, LTI case):** if A,B,C are constant, unroll the recurrence: y_t = Σ_{k=0}^{t} (C A^k B) x_{t-k} = (K * x)_t, with kernel K_k = C A^k B. Compute y for the whole sequence via FFT in O(L log L). This is the S4 path.

3. **Associative scan (training, data-dependent case):** the map (A_t, b_t) with composition (A_2,b_2)∘(A_1,b_1) = (A_2 A_1, A_2 b_1 + b_2) is associative, so a parallel prefix scan (Blelloch) computes all h_t in O(L) work / O(log L) depth. Needed when A_t,B_t depend on x_t (no fixed kernel exists), e.g. Mamba.

The duality: form 1 ⟺ form 2/3 produce bit-identical outputs (modulo float error); only the FLOP/memory access pattern differs.

## Where it appears

- **S4 / S4D / H3** — diagonal-plus-low-rank A gives a closed-form convolution kernel; train via FFT, infer via recurrence.
- **Mamba (S6)** — selective (input-dependent) A_t,B_t kill the convolution form, so it uses a hardware-aware parallel scan in SRAM; inference is a constant-memory recurrence.
- **Linear attention / GLA / RetNet / chunked linear attention** — the same recurrence is materialized as a "chunkwise parallel" form: intra-chunk via matmul, inter-chunk via state passing — interpolates between full-parallel and pure-recurrent.
- **Transformer-to-recurrent distillation** — distill softmax attention into a layer with a recurrent inference mode for cheap serving.

## Common mistake

Assuming every fast SSM admits a convolution. The global-convolution (FFT) trick requires **LTI** dynamics — A,B,C constant in t. Mamba's selectivity makes the recurrence time-varying, so there is no fixed kernel and you must use the scan, not the FFT. Also: the duality holds only because the recurrence is **linear** in the state; a softmax/nonlinearity inside the loop breaks all three equivalences.

## See also
- [[linear-time-invariant-ssm]] — the LTI special case where the convolution kernel exists in closed form
- [[selective-state-space-models-mamba]] — input-dependent dynamics that force the scan over the convolution
- [[state-space-duality]] — the formal SSM↔(masked)attention equivalence underlying chunked forms
