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


def meta_rescale(probs, meta):
    """Meta-calibration: set top-label confidence to P(correct) from an
    L2-logistic correctness head on decision signals; rescale the rest
    proportionally (argmax + accuracy unchanged, distributions preserved).

    probs: {label: prob} (already temperature-scaled). meta: artifact dict
    with feats/mu/sd/w/b. Supported features: c (top prob), margin
    (top-second), ent (entropy nats), kopts (log #options).
    Returns a new dict. Unknown features raise (fail loud, not silent).
    """
    import math
    ps = sorted(probs.values(), reverse=True)
    top = ps[0]
    second = ps[1] if len(ps) > 1 else 0.0
    ent = -sum(v * math.log(max(v, 1e-15)) for v in probs.values())
    table = {"c": top, "margin": top - second, "ent": ent,
             "kopts": math.log(max(len(probs), 2))}
    xs = []
    for f in meta["feats"]:
        if f not in table:
            raise ValueError(f"meta feature not servable: {f}")
        xs.append(table[f])
    mu, sd, w, b = meta["mu"], meta["sd"], meta["w"], meta["b"]
    z = sum(wi * (x - m) / s for wi, x, m, s in zip(w, xs, mu, sd)) + b
    p_top = 1.0 / (1.0 + math.exp(-z))
    labels = list(probs.keys())
    top_lab = max(labels, key=lambda k: probs[k])
    rest = 1.0 - p_top
    rest_old = 1.0 - probs[top_lab]
    out = {}
    for lab in labels:
        if lab == top_lab:
            out[lab] = p_top
        else:
            out[lab] = probs[lab] / rest_old * rest if rest_old > 1e-12 else 0.0
    return out
