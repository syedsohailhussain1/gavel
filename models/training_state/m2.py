#!/usr/bin/env python3
"""m2.py — exact Python mirror of decide-core's M2 calibration math.
- fit_temperature: 80 log-spaced T in [0.05, 10], T = 0.05*200**(i/79),
  minimize mean NLL with prob clamp 1e-15, strict < (first wins ties).
- expected_calibration_error: 10 equal bins over [0,1],
  b = min(floor(clamp(c)*10), 9), occupancy-weighted |acc-conf|.
- calibrate_logits: (T, ece_before, ece_after) for val logit sets + targets.
Use at export time so TF-IDF artifacts ship calibrated (T baked in).
"""
import math


def softmax_scaled(logits, temperature):
    s = [l / temperature for l in logits]
    m = max(s)
    exps = [math.exp(l - m) for l in s]
    tot = sum(exps)
    return [e / tot for e in exps]


def fit_temperature(logit_sets, targets):
    def nll(t):
        tot = 0.0
        for logits, y in zip(logit_sets, targets):
            p = softmax_scaled(logits, t)[y]
            tot += -math.log(max(min(p, 1.0), 1e-15))
        return tot / len(logit_sets)
    best_t, best_nll = 1.0, float("inf")
    for i in range(80):
        t = 0.05 * (200.0 ** (i / 79.0))
        n = nll(t)
        if n < best_nll:
            best_nll, best_t = n, t
    return best_t


def expected_calibration_error(confs, correct, n_bins=10):
    n = len(confs)
    if n == 0:
        return 0.0
    acc = [0.0] * n_bins
    csum = [0.0] * n_bins
    cnt = [0] * n_bins
    for c, ok in zip(confs, correct):
        b = int(math.floor(max(0.0, min(1.0, c)) * n_bins))
        b = min(b, n_bins - 1)
        acc[b] += 1.0 if ok else 0.0
        csum[b] += c
        cnt[b] += 1
    ece = 0.0
    for b in range(n_bins):
        if cnt[b]:
            ece += (cnt[b] / n) * (abs(acc[b] - csum[b]) / cnt[b])
    return ece


def calibrate_logits(logit_sets, targets):
    """Returns (T, ece_before, ece_after). ece uses max-prob confidence."""
    def at(T):
        conf, corr = [], []
        for logits, y in zip(logit_sets, targets):
            p = softmax_scaled(logits, T)
            j = max(range(len(p)), key=lambda k: p[k])
            conf.append(p[j])
            corr.append(j == y)
        return conf, corr
    c0, k0 = at(1.0)
    T = fit_temperature(logit_sets, targets)
    c1, k1 = at(T)
    return T, expected_calibration_error(c0, k0), expected_calibration_error(c1, k1)
