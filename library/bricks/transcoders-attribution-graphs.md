# Transcoders & Attribution Graphs

**One-liner:** A transcoder is a sparse, interpretable module that learns an MLP's input→output map (replacing it for analysis), and stacking these into a replacement model lets you build attribution graphs — directed feature-to-feature computation traces for a single prompt ("biology of an LLM").

## The formula / definition

An SAE *reconstructs* one activation: x → x̂ ≈ x. A **transcoder** instead *predicts the output of a layer from its input*, approximating the MLP as a wide sparse dictionary:

a = TopK(W_enc · LN(h_in) + b_enc)      # sparse feature acts, m ≫ d
MLP_out(h_in) ≈ W_dec · a + b_dec        # predicts the MLP's output, not its input

So the nonlinear MLP is replaced by a sparse linear read of interpretable features. **Cross-layer transcoders (CLTs)** let each layer's features write to *all later* layers' residual stream, capturing multi-layer circuits.

**Attribution graph:** swap every MLP for its transcoder → a *replacement model*. On one prompt, freeze attention patterns and LayerNorm denominators so the rest is *linear*; the direct effect of active feature i on feature j (or logit) is then the exact linear weight along that frozen path. Nodes = active features + embeddings + error terms; edges = these linear attributions. Prune to the subgraph driving the output.

## Where it appears

- **Anthropic — "Circuit Tracing" / "On the Biology of a Large Language Model" (2025)** — CLTs on Claude 3.5 Haiku; traced multi-step reasoning (Dallas→Texas→Austin), planned rhymes, multilingual features, refusal and hallucination circuits.
- **Transcoders (Jacob Dunefsky et al., 2024)** — original per-MLP transcoders, shown more faithful and circuit-friendly than reconstruction SAEs on the MLP.
- **Neuronpedia / open CLT tooling (2025)** — interactive attribution-graph viewers; open replication of the method.

## Common mistake

Reading the graph as the *real* mechanism. It is the **replacement model's** computation, not the original's: error nodes absorb everything the transcoders miss, frozen attention hides a lot, and edges are correlational-until-verified. A claimed circuit is only validated by **intervening** (clamp/ablate the feature) in the *actual* model and seeing the predicted effect.

## See also
- [[sparse-autoencoders]] — transcoders are the input→output sibling of reconstruction SAEs
- [[circuits]] — attribution graphs are an automated, feature-level circuit-discovery method
- [[attribution-patching]] — the linear-approximation-to-causal-effect idea that makes graph edges cheap
