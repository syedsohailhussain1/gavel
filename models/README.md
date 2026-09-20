# Tiny sentiment model (CPU demo)

A real neural network that runs on any laptop CPU — no GPU, no internet
needed after download. This is the kind of model gavel's M3 ONNX engine
will load: small, quantized, and fast enough that inference feels instant.

## What it is

- **Checkpoint:** `philschmid/tiny-bert-sst2-distilled` (Apache-2.0) — a
  bert-tiny model (2 layers, 128 hidden size, ~4.4M parameters) fine-tuned
  on SST-2 movie-review sentiment.
- **What we do in `download.py`:** export it to ONNX and apply dynamic
  int8 quantization. The final `model.onnx` is a few MB.
- **Task:** binary sentiment — every input is classified `positive` or
  `negative`, with a confidence score.
- **Reported quality:** the author reports **83.3% accuracy on the SST-2
  evaluation set**. That is a real, modest number: fine for clear-cut
  reviews, shaky on sarcasm and mixed opinions. We ship it because it is
  tiny and honest, not because it beats frontier models — better students
  come from distilling bigger teachers on more data (the actual M3 recipe).

We tried a bert-tiny SMS-spam ONNX model first (thematically closer to
phishing), but it missed textbook spam examples in testing, so we dropped
it rather than demo a model we couldn't stand behind.

## Setup (one time)

Python 3.10+ on any laptop, Mac/Windows/Linux:

```bash
pip install torch optimum[onnxruntime] onnxruntime tokenizers huggingface_hub
cd models
python download.py     # downloads the checkpoint, exports + quantizes (~2-5 min)
python test_model.py   # runs the demo + benchmark
```

`torch` is only needed for the one-time export in `download.py`.
`test_model.py` needs just `onnxruntime`, `tokenizers`, and `numpy` —
that is the entire runtime: a ~4MB model file and two small libraries.

## What the output means

For each of 8 short reviews, `test_model.py` prints:

```
[OK  ] positive (0.972)  expected=positive  :: A wonderful, heartwarming ...
```

- **label** — the model's verdict: `positive` or `negative`.
- **confidence** — the softmax probability for that label (0–1). Treat it
  as the model's self-reported certainty, not a calibrated guarantee:
  small models are often overconfident. Gavel's calibration step
  (temperature scaling) exists precisely to fix that before these numbers
  drive `act` / `review` / `escalate` decisions.
- The script exits non-zero if any sample is misclassified.

Then a latency benchmark (200 CPU inferences). Measured on the build
machine: **p50 0.39 ms, p99 0.90 ms** — roughly 2,500 decisions/second on
one CPU thread, at zero marginal cost. Your laptop will differ, but the
order of magnitude is the point: this is sub-millisecond, not
hundreds-of-milliseconds.

## Limitations

- English movie-review language. It will misread sarcasm, domain slang,
  and non-English text.
- Binary output only: no "mixed" or "neutral".
- Confidence is uncalibrated out of the box (see above).
- 83.3% SST-2 accuracy is the honest baseline for this checkpoint;
  beating it means a better teacher and more distillation data, not a
  bigger laptop.
