# Cookbook: 200 judgments, one call

Ask 200 independent questions over the same input in a single request, and get
200 typed answers back. This cookbook runs one `POST /ask` against the
`issue-triage` question (a TF-IDF + logistic regression model trained on
5,832 labeled Kubernetes issues) and times it.

## Setup

Build and start the server, then define the question:

```bash
cargo build -p decide-server --release
./target/release/gavel-server &
curl -X POST http://localhost:7575/define \
  -H 'Content-Type: application/json' \
  -d @/home/hatch/workspace/tasks/issue-triage/gavel_define_tfidf.json
```

The model file (`models/tfidf-issuetriage.json`) and the define payload are
build artifacts from the issue-triage training run — they live outside this
repo and aren't downloadable. Train your own with the sklearn pipeline
described in the TF-IDF engine docs, or point the paths at your own export.
Everything below runs on commodity CPU; the numbers here were measured on a
2-core VM.

## The questions

200 of them, mixed kinds, all independent views over one input:

- 50 `choice` questions: pick one of `bug`, `feature`, `docs`, `support`.
- 150 `noul` questions: "is this a bug?", "is this a feature?", and so on —
  one per class, repeated. Each returns P(true) directly.

The input is a single real held-out issue (a kubelet crash report). Every
question sees the same input; no question sees another's answer.

## How we ask

One POST. The `questions` array rides alongside the input:

```bash
curl -X POST http://localhost:7575/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"issue-triage",
       "input":"<issue title + body>",
       "questions":[{... 200 questions ...}]}'
```

## What came back

| | Gavel (this cookbook) | TypeSafe cookbook (2026-09-11) |
|---|---|---|
| Questions per call | 200 | 14 |
| Wall time per call | 54.4 ms | 111 ms mean round-trip |
| Per question | 0.27 ms | ~7.9 ms |
| Answers in order | 200/200 | n/a |
| Errors | 0 | n/a |

All 200 answers came back `act` — the model was confident on this input (a
held-out issue labeled `bug`; per-answer predictions weren't individually
audited in this run). Run it twice and you get byte-identical answers: this
engine is deterministic, so run-to-run variance is zero by construction, not
by luck.

## The pattern

If the questions are independent, ask them together. Each question currently
runs a full inference pass against the same input — 0.27 ms each end to end,
no batching tricks inside. Sharing one featurization across items is an
obvious next optimization; the measured number already includes the
unoptimized path. Policy stays in your code — the response carries `p_true` /
confidence / `action` per question, and you decide what `act` means.

This is the same decomposition TypeSafe's skill recommends ("ask independent
questions over the same state together"). The difference is economic: at
0.27 ms and zero marginal cost per question, there is no reason to ration
questions. Ask the speculative ones. Ask the cheap verification ones. The
budget conversation disappears.

## Caveats

- Localhost vs. a production API is not a fair latency fight. Their 111 ms
  includes the network; ours doesn't. The per-question gap is about 30x —
  and the work per question isn't equal either (next bullet), so read it as
  "cheap judgments are cheap here," not as a victory over a neural net.
- Our 200 questions are views over one linear model; their 14 are 14 distinct
  semantic judgments from a neural net. Different work per question. This
  cookbook measures "answer many cheap judgments fast," not "out-reason a
  frontier model."
- Narrow task, one dataset, random split. The model knows Kubernetes issues
  and little else — that's the deal with specialists, and the abstention
  layer exists for everything outside it.
