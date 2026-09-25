# JevBench entry: Gavel-Decide 4B (candidate)

Status: preparation. Weights + serving code public; awaiting operator run.
This file is the pre-run disclosure (mapping, cost basis, license,
provenance) — committed BEFORE any official measurement, per house rules.

## System

Pairwise scorer head (MLP 2560-512-256-1) over frozen Qwen3-4B-Base (4-bit),
temperature-fitted. One batched forward per question over its options,
softmax/T, argmax decides. Score items: expected level from the option
distribution. No generation, no sampling, deterministic given weights.

- Trunk: `Qwen/Qwen3-4B-Base` (Apache-2.0, public, pinned revision TBD at run)
- Head: `syedsohailhussain/gavel-decide-4b`, `v1/combined_head.pt`
  (MIT, ours), fitted T=0.53 on 111 public hard decisions (ECE 0.265→0.070)
- Serving: `models/training_state/serve_systemone.py` (this repo,
  branch `generalist-v1`) — TypeSafe-compatible `/v1/systemone`, stdlib
  HTTP, batch-per-question inference
- Confidence: temperature softmax, then meta-calibration top-rescale —
  top-label confidence = L2-logistic P(correct) on [maxprob, margin,
  entropy, log#options] (C=0.05, OOF ECE 0.032–0.057 over 4 seeds);
  remaining mass rescaled proportionally, argmax provably preserved
  (0 flips in 231). ECE 0.268 → 0.036 served (0.070 temp-only).
- Eval harness: `models/training_state/run_jevbench.py` (their Runner +
  our `gavel_adapter.py`, Windows fcntl shim documented inline)

## Mapping (fixed before any official run)

- choice: option text = `label: criteria`; distribution = softmax over
  option logits; answer = argmax.
- noul: options yes/no with criteria text where provided.
- score: option per level (index order preserved); expected level =
  probability-weighted sum; distribution returned whole.
- Long states left-truncated to 512 total tokens (matches training
  distribution; same budget precedent as laya's 512).
- No retries, no regeneration, no answer post-processing beyond argmax.

## Cost basis

Open weights run on the operator's hardware: no billable account exists,
so per their rules cost is NOT zero — price at the hosted 4B-class
reference tariff times measured tokens (same treatment as kev/openJev
rows). Our measured token counts ride in each run's usage ledger.

## License / provenance

- Trunk: Apache-2.0 (Qwen). Head + code: MIT (ours).
- Trained on: public JevBench items (MIT halves) + MNLI supplement only.
  Sealed items untouched — enforced by eval-gate, happy to submit to the
  normalized-text overlap audit.
- Public-benchmark-directed training disclosed (smalljev precedent):
  temperature fitted on 111 public hard decisions; pair data includes
  public items. Sealed generalization is the open question, stated plainly.

## Measured so far (local, pre-entry)

- Public: 173/231 = 74.9% (easy 91.7%, original 73.6%, hard 68.5%),
  schema validity 1.0, operational success 1.0.
- ECE (hard, official scorer): 0.268 → 0.036 served (0.070 temp-only).
- Latency here (throttled GTX 1650): p50 9.1s — NOT representative;
  entry latency is measured on operator hardware by design.
- Robustness gates: option-order shuffle invariance holds (664 vs 665,
  fp noise); +2 adversarial distractors hold 62.0% → 58.2%.
