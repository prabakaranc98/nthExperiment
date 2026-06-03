# Machine Unlearning

**One-liner:** Removing the influence of specific data or capabilities (hazardous knowledge, copyright, PII) from a trained model without full retraining — most methods only suppress outputs, leaving the knowledge shallow and relearnable.

## The definition

Goal: produce model θ_unlearn that approximates the model you'd have gotten by retraining-from-scratch on the retain set D_retain = D \ D_forget. The gold standard is *exact* unlearning (θ_unlearn ≈ θ trained without D_forget); everything else is *approximate*.

Typical objective splits into a forget term and a retain (utility) term:

  L = L_retain(θ; D_retain)  −  λ · L_forget(θ; D_forget)

The forget term is the hard part. Common instantiations:
- **Gradient ascent (GA):** maximize loss on D_forget — `θ ← θ + η ∇L(D_forget)`. Unstable, degrades the model fast.
- **RMU (Representation Misdirection for Unlearning, Li et al. 2024):** steer hidden activations on forget topics toward a random vector at a chosen layer ℓ, while keeping activations on retain data unchanged:
  L = ‖a_ℓ(x_forget) − c·u‖²  +  α‖a_ℓ(x_retain) − a_ℓ^frozen(x_retain)‖²
- **Preference/NPO (Negative Preference Optimization):** treat forget examples as dispreferred in a DPO-style loss to avoid GA's divergence.

## Where it appears

- **WMDP (Weapons of Mass Destruction Proxy, 2024)** — benchmark of hazardous bio/chem/cyber knowledge; RMU is the reference unlearning method evaluated on it
- **TOFU (2024)** — synthetic-author QA benchmark for forgetting specific entities while preserving general utility
- **Copyright / PII / GDPR "right to be forgotten"** — the legal motivation; influence functions and SISA (sharded retraining) target *exact* deletion
- **Refusal vs. unlearning** — labs increasingly prefer real capability removal over [[refusal-safety-training]] because refusals are easily jailbroken

## Common mistake

Believing unlearning removes the knowledge. Most methods only suppress it at the output layer — the information remains in the weights and is recoverable. Relearning attacks (a few fine-tuning steps on benign related data), quantization, or even adding the forget direction back via a steering vector frequently restores the "unlearned" capability. Unlearning evals must probe robustness, not just measure forget-set accuracy.

## See also
- [[refusal-safety-training]] — the shallow alternative unlearning tries to improve on; both are relearnable
- [[steering-vectors-activation-steering]] — RMU is essentially destructive activation steering on forget topics
- [[membership-inference-training-data-extraction]] — the attack model unlearning must defend against (proving data is gone)
