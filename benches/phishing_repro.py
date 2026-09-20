#!/usr/bin/env python3
"""Reproduce the jev-phishing-bench evaluation with gavel's LogisticEngine.

End-to-end reproduction of benches/results/phishing-comparison.md:

  1. Clone the benchmark repo (third-party code stays out of this repo):
       git clone https://github.com/anisselbd/jev-phishing-bench /tmp/jev-phishing-bench
  2. Install the two pure-Python deps the driver needs:
       pip install numpy tldextract
  3. Run this script:
       python3 benches/phishing_repro.py
     It downloads PhishNChips v5.2 (SHA-256 verified against the release
     manifest), rebuilds emails.jsonl exactly like the benchmark's
     prepare_data.py (seed 20260916), replicates the published A/B split
     (bench/protocol.py, SPLIT_SEED=20260917), sanity-checks the split by
     re-running the published heuristic rule (must hit 0.918 on half B),
     writes train/val/test JSONL to --work-dir (default /tmp/gavel-phish),
     then runs `cargo run --release -p decide-phishing` and saves the metrics
     JSON next to this script's output.

Outputs:
  <work-dir>/metrics_gavel.json   raw metrics from the Rust binary

No third-party data is written into this repo; --work-dir defaults to /tmp.
The committed report is benches/results/phishing-comparison.md.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

SEED = 20260916          # prepare_data.py shuffle seed
SPLIT_SEED = SEED + 1    # bench/protocol.py stratified-half seed
TRAINVAL_SEED = 0x9E3779B9  # train/val split of half A (our choice, fixed)
HF_BASE = "https://huggingface.co/datasets/AreLit/PhishNChips/resolve/main/"
FILES = {
    "core_emails.csv": "cebb407ff8630491a97400e37464b8db8dfc4299164fca51fcb4ac7eec8204ef",
    "prompt_strategies.json": "0a293bb8e722d0621845f3e0f9b5cd8bd100818c35222c1cd6964a8fa3b367b0",
    "reference_results.csv": "5e15d7498aeadd73a23aec0b01979cacf02d172b97c88b3db0078609cc92ef15",
    "SOURCE_LICENSES.md": "129e19acd6ae8fb243b7861fd7f7c3c63987f431b0f73626b7d21206e275f8b7",
}
EMAIL_FIELDS = ["sender", "from", "subject", "body", "link_display_text", "link_url"]


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(bench: Path, work: Path):
    import urllib.request
    d = work / "phishnchips"
    d.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        target = d / name
        if target.exists() and sha256_file(target) == digest:
            print(f"  {name}: cached, checksum ok")
            continue
        print(f"  {name}: downloading")
        urllib.request.urlretrieve(HF_BASE + name, target)
        got = sha256_file(target)
        if got != digest:
            sys.exit(f"Checksum mismatch for {name}: {got} (expected {digest})")
        print(f"  {name}: checksum ok")


def build_emails(work: Path):
    """Replicates prepare_data.build_emails()."""
    csv.field_size_limit(1 << 30)
    with open(work / "phishnchips" / "core_emails.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2000, f"expected 2000 rows, got {len(rows)}"
    records = []
    for row in rows:
        email = json.loads(row["email_content"])
        assert all(k in email for k in EMAIL_FIELDS), row["id"]
        label = int(row["phish_label"])
        records.append({
            "id": row["id"],
            "y": label,
            "label": "phishing" if label == 1 else "legitimate",
            "email": {k: email[k] for k in EMAIL_FIELDS},
        })
    rng = random.Random(SEED)
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["order"] = i
    out = work / "emails.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} emails -> {out}")
    return records


def stratified_halves(ids, y, seed=SPLIT_SEED):
    """Replicates bench/protocol.stratified_halves()."""
    rng = np.random.default_rng(seed)
    a, b = set(), set()
    for label in (0, 1):
        group = sorted(i for i in ids if y[i] == label)
        perm = rng.permutation(len(group))
        half = len(group) // 2
        a.update(group[k] for k in perm[:half])
        b.update(group[k] for k in perm[half:])
    return a, b


def auroc(yy, p):
    """Rank-based AUROC with tie-averaged ranks (analyze.py)."""
    yy = np.asarray(yy)
    pos, neg = p[yy == 1], p[yy == 0]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allp = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allp):
        j = i
        while j + 1 < len(allp) and allp[j + 1] == allp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = len(pos)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def sanity_check_split(records, split_a, split_b, bench: Path):
    """Re-run the published heuristic rule on half B; must give ~0.918."""
    sys.path.insert(0, str(bench))
    from bench.heuristics import HEURISTIC_FEATURES, features as heuristic_features
    by_id = {r["id"]: r for r in records}
    y = {r["id"]: r["y"] for r in records}
    feats = {i: heuristic_features(by_id[i]["email"]) for i in by_id}
    xa = np.array([[feats[i][n] for n in HEURISTIC_FEATURES] for i in sorted(split_a)])
    xb = np.array([[feats[i][n] for n in HEURISTIC_FEATURES] for i in sorted(split_b)])
    ya = np.array([y[i] for i in sorted(split_a)])
    yb = np.array([y[i] for i in sorted(split_b)])
    aucs = {n: auroc(ya, xa[:, k]) for k, n in enumerate(HEURISTIC_FEATURES)}
    best = max(aucs, key=aucs.get)
    k = HEURISTIC_FEATURES.index(best)
    vals = np.unique(xa[:, k])
    cands = sorted([0.5] + [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)])
    t = max(cands, key=lambda tt: float((((xa[:, k] >= tt).astype(int) == ya).mean())))
    acc_b = float((((xb[:, k] >= t).astype(int) == yb).mean()))
    print(f"  sanity: {best} >= {t} on B -> accuracy {acc_b:.4f} (published 0.918)")
    if not (best == "hosting_or_shortener" and abs(acc_b - 0.918) < 0.005):
        sys.exit("SPLIT REPLICATION FAILED - refusing to continue")
    print("  sanity check passed: A/B split replicates the published one")


def serialize(email):
    e = email
    return (f"from: {e['from']} sender: {e['sender']} subject: {e['subject']} "
            f"link_text: {e['link_display_text']} link_url: {e['link_url']} body: {e['body']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-repo", default="/tmp/jev-phishing-bench")
    ap.add_argument("--work-dir", default="/tmp/gavel-phish",
                    help="scratch dir for third-party data (kept out of this repo)")
    args = ap.parse_args()
    bench = Path(args.bench_repo)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if not (bench / "bench" / "protocol.py").exists():
        sys.exit(f"benchmark repo not found at {bench}; clone it first (see --help)")

    print("== download + verify PhishNChips v5.2 ==")
    download(bench, work)
    print("== build emails.jsonl (seed 20260916) ==")
    records = build_emails(work)
    by_id = {r["id"]: r for r in records}
    ids = [r["id"] for r in records]
    y = {r["id"]: r["y"] for r in records}

    print("== replicate published A/B split ==")
    split_a, split_b = stratified_halves(ids, y)
    assert len(split_a) == 1000 and len(split_b) == 1000
    sanity_check_split(records, split_a, split_b, bench)

    print("== split A -> train (800) / val (200); B -> test (1000) ==")
    rng = random.Random(TRAINVAL_SEED)
    train_ids, val_ids = [], []
    for label in (0, 1):
        group = sorted(i for i in split_a if y[i] == label)
        rng.shuffle(group)
        train_ids += group[:400]
        val_ids += group[400:]
    test_ids = sorted(split_b)
    assert not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids)
                or set(val_ids) & set(test_ids)), "leakage between splits!"
    for name, sel in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        path = work / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in sel:
                r = by_id[i]
                f.write(json.dumps({"input": serialize(r["email"]), "label": r["label"]},
                                   ensure_ascii=False) + "\n")
        print(f"  wrote {path} ({len(sel)})")
    print("  no leakage between splits")

    print("== run decide-phishing ==")
    metrics_path = work / "metrics_gavel.json"
    with open(metrics_path, "w") as out:
        subprocess.run(
            ["cargo", "run", "--release", "-p", "decide-phishing", "--",
             str(work / "train.jsonl"), str(work / "val.jsonl"), str(work / "test.jsonl")],
            cwd=REPO_ROOT, stdout=out, check=True)
    m = json.loads(metrics_path.read_text())
    print(f"  accuracy={m['accuracy']:.4f} recall={m['recall_phishing']:.4f} "
          f"fpr={m['false_positive_rate']:.4f} auroc={m['auroc']:.4f} "
          f"ece={m['ece_reference_binning']:.4f} "
          f"p50={m['ask_latency_us']['p50']:.1f}us")
    print(f"== done. metrics -> {metrics_path} ==")


if __name__ == "__main__":
    main()
