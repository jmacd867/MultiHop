# Attention instrumentation captures per-head, pre- and post-softmax

The instrumentation pass (a separate eval-mode-only forward call, gated by a `capture_attention` flag, writing into an `nnx.Intermediate` collection with zero cost when disabled) needs a decision on capture granularity, since re-instrumenting later if this turns out insufficient means re-running the eval-mode pass across all three variants.

**Per-head, not averaged across heads.** If a hybrid variant's chain-tracking is concentrated in one or two specialized heads (analogous to induction heads), averaging at capture time would erase exactly the signal this study is looking for, leaving only a diffuse, uninformative pattern. Per-head detail can always be averaged later during analysis for a summary view; the reverse — recovering per-head detail from an already-averaged capture — is not possible. Capture at maximum granularity now.

**Both pre-softmax scores and post-softmax weights, not post-softmax alone.** Post-softmax attention weights give the standard "where is attention mass going" picture. But softmax normalization discards information that distinguishes two mechanistically different failure modes:

- **Diffuse/unconfident attention** (small logit gaps pre-softmax) — the model isn't sure where to look along the chain.
- **Sharp attention pointed at the wrong token** (a distractor winning cleanly) — the model is confident but wrong.

Post-softmax weights alone can't distinguish these; pre-softmax logits can. Since the capture path is already gated behind a flag with zero training-time cost, capturing both is a small addition to the same `nnx.Intermediate` write versus a costly re-instrumentation pass later if the post-softmax-only view turns out ambiguous when the degradation grid is actually analyzed.
