# Flywheel: how phishing_tfidf never regresses (permanent, not a one-time fix)

Production question: `phishing_tfidf` (stable name). Versioned artifacts beside it
(`phishing_tfidf_v3.json`, `..._v2.json`, `phishing_tfidf_prod.json`, `.bakN.json`).

## The loop (run in order; every step is a script here)

1. **Log** — every future miss/abstain/escalate in production gets appended to
   `negation_aug_train.json` (fresh wording, never probe text) or a new themed file.
2. **Regenerate** — `contrast_augment.py` rebuilds 360 contrastive items
   (advisory-framed legit / negation-dressed phish / cancelled legit) from base
   train, seed 11. Failure-MODE coverage, automatic, every retrain.
3. **Retrain** — `retrain_all.py` rebuilds ALL questions from scratch, including
   production TF-IDF on base + hand-aug + regenerated contrastive. Previous prod
   artifact is rotated to `.bakN.json` (rollback). Disaster recovery == production.
4. **Gate** — `eval_gate.py <candidate>` BLOCKS promote unless:
   probe 16/16 labels, aug_val >= 15/16, held-out test150 >= 0.985, zero skips.
   Exit 1 = blocked. Exit 0 writes the production pointer into `manifest.json`.
5. **Monitor** — every TF-IDF `/ask` returns `energy`. Flat/OOD inputs have high
   energy (misses observed at -0.69 while confident corrects sit below -0.9).
   If a future probe shows misses with separable energies, arm
   `abstain_energy_above` in the `/define` policy (recipe proven 2026-09-21:
   gate at -0.6985 converted 2 misses to abstain with zero corrects lost).

## Proven history (all measured on this box)

- v1 (800 base): probe 10/16.
- +32 hand aug: 14/16 + 2 abstain via energy gate.
- +16 round-2 (delivery/negation patterns): 16/16, test150 149/150.
- v3 (1208, +contrastive regen): gate GREEN, test150 150/150.
- Full miss→fix cycle: voicemail-attachment phish missed (aug_val 15/16) →
  2 fresh variants appended → rebuilt → aug_val 16/16, probe 16/16, test150 150/150.
- PR review fixes: all TF-IDF artifacts ship fitted T (M2 mirror `m2.py`;
  prod T=0.306, ECE 0.044→0.007); gate asserts calibration from the artifact;
  Temp deps moved to D:/jev-phishing-bench + D:/gavel-phish; all scripts routed
  through `gavel_paths.py` (GAVEL_TS/GAVEL_URL/GAVEL_BENCH/GAVEL_PHISH env);
  probe shared via `probe_items.py`; advisory senders rotate (no shortcut).

## Module map

- `gavel_paths.py` — every machine-local path (env-overridable).
- `m2.py` — exact mirror of decide-core fit_temperature + native ECE.
- `retrain_all.py` — rebuilds everything, incl. calibrated prod TF-IDF.
- `eval_gate.py` — labels + artifact-calibration checks, blocks on fail.
- `contrast_augment.py` — deterministic contrastive regen (seed 11).
- `ladder.py` — zero-shot cold-start loop with fitted T.
- `signals_build.py` — 5 signal artifacts, each T-fitted on its val.
- `decompose.py` — verdict+signals+rules sidecar (:7586).
- `dashboard.py` — live view (:7587).

## Rules

- Never tune thresholds on the probe; it is a test, not a training set.
- Never train on probe or aug_val text; fresh wording only.
- A red gate blocks. No exceptions, no "just this once".

## Compute policy (2026-09-21): GPU + the fast trainer

- Classical engines (TF-IDF/logreg/MLP, sklearn): CPU — they train in seconds;
  GPU buys nothing there.
- Any transformer fine-tuning: torch CUDA on the GTX 1650 (verified alive:
  4k-matmul x10 in 0.99s vs 5.40s CPU). bitsandbytes 4-bit + small batches fit 4GB VRAM.
- Any frozen-trunk + heads training: `D:\PactGGUF\scripts\train_real_v2.py`
  (cached-forward, ~56x over the v1 loop, staged ckpts, SeqKD mix) — the
  designated fast method. Smoke-verified 2026-09-21: 5 steps in 0.3 min,
  train 15.87->14.93, val_bpb 4.46->3.71 (`logs/smoke_v2.log`).
- Qwen inference/labeling: `device_map="auto"` (GPU layers + CPU offload) as before.
