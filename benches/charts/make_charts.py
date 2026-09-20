#!/usr/bin/env python3
"""Generate benchmark charts for the gavel repo.

Reproducible: all numbers are hardcoded from
  - benches/results/phishing-comparison.json (gavel's measured run)
  - the published jev-phishing-bench report dated 2026-09-17 (Jev/Haiku, quoted not re-run)
  - crates/decide-bench output (synthetic data, labeled as such)

Run: python3 benches/charts/make_charts.py
Outputs: benches/charts/*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
GAVEL = "#1d4ed8"   # gavel: strong blue
OTHER = "#9ca3af"   # everyone else: gray
OTHER2 = "#6b7280"

plt.rcParams.update({
    "figure.dpi": 170,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

def bar_chart(path, title, subtitle, labels, values, colors, fmt,
              footnote=None, yerr=None, logy=False, ylabel=None):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = range(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="white",
                  yerr=yerr, capsize=6, ecolor="#374151",
                  error_kw={"elinewidth": 1.5})
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(title, loc="left", pad=34)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=10.5,
                color="#4b5563", va="bottom", ha="left")
    if ylabel:
        ax.set_ylabel(ylabel, color="#4b5563")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # value labels on bars
    labels_txt = [fmt(v) if callable(fmt) else t for v, t in zip(values, fmt)] \
        if not callable(fmt) else [fmt(v) for v in values]
    ymax = max(v for v in values if v > 0) if any(v > 0 for v in values) else 1
    for b, v, txt in zip(bars, values, labels_txt):
        y = b.get_height()
        if logy:
            ax.text(b.get_x() + b.get_width() / 2, y * 1.25, txt,
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        else:
            ax.text(b.get_x() + b.get_width() / 2,
                    y + (0.02 * ymax if y > 0 else 0.02), txt,
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    if footnote:
        fig.text(0.02, 0.01, footnote, fontsize=8.5, color="#6b7280", wrap=True,
                 ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(os.path.join(OUT, path), bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# --- data (see module docstring for provenance) ---
acc_head = (("gavel", 99.7), ("Jev", 62.6), ("Haiku 4.5", 81.3))
sup = (("gavel\nraw text", 99.7, (0.6, 0.2)),
       ("Jev 5-signals\nlogistic", 95.0, (1.5, 1.2)),
       ("Haiku 5-signals\nlogistic", 93.2, (1.7, 1.4)),
       ("regex\nlogistic", 91.8, None))
ece = (("gavel", 0.003), ("Jev", 0.154), ("Haiku 4.5", 0.097))
lat_ms = (("gavel", 0.0112, "11 \u00b5s"), ("Jev", 239.0, "239 ms"),
          ("Haiku 4.5", 687.0, "687 ms"))
cost = (("gavel", 0.0, "$0"), ("Jev", 0.038, "$0.038"),
        ("Haiku 4.5", 0.462, "$0.462"))

bar_chart(
    "accuracy_headline.png",
    "How often it gets the right answer",
    "Accuracy on phishing emails, % correct (higher is better)",
    [l for l, _ in acc_head], [v for _, v in acc_head],
    [GAVEL, OTHER, OTHER2], lambda v: f"{v:.1f}%",
    footnote=("Jev/Haiku ran zero-shot (no training data); gavel trained on 800 labeled emails. "
              "Headline is apples-to-oranges \u2014 see accuracy_supervised.png for the fair comparison. "
              "Jev/Haiku numbers quoted from the published 2026-09-17 benchmark report."))

bar_chart(
    "accuracy_supervised.png",
    "The fair fight: trained models only",
    "Accuracy on the same held-out split, % (higher is better) \u00b7 bars show 95% intervals",
    [l for l, _, _ in sup], [v for _, v, _ in sup],
    [GAVEL, OTHER, OTHER, OTHER2], lambda v: f"{v:.1f}%",
    yerr=[[e[0] if e else 0 for _, _, e in sup],
          [e[1] if e else 0 for _, _, e in sup]],
    footnote=("All four trained on the same 800 emails, tested on the same 1,000. "
              "Jev/Haiku-signal baselines from the published report; gavel re-ran 2026-09-20."))

bar_chart(
    "calibration_ece.png",
    "How honest the confidence scores are",
    "Expected calibration error, 10 bins (lower is better)",
    [l for l, _ in ece], [v for _, v in ece],
    [GAVEL, OTHER, OTHER2], lambda v: f"{v:.3f}",
    footnote=("ECE = average gap between stated confidence and actual accuracy. "
              "Gavel calibrated with temperature scaling on held-out data. "
              "Jev/Haiku numbers quoted from the published 2026-09-17 report."))

bar_chart(
    "latency.png",
    "How fast it answers",
    "Median time per decision (lower is better) \u00b7 log scale",
    [l for l, _, _ in lat_ms], [v for _, v, _ in lat_ms],
    [GAVEL, OTHER, OTHER2], [t for _, _, t in lat_ms],
    logy=True,
    footnote=("Gavel measured in-process on this machine; Jev/Haiku are France\u2192US wall-clock incl. network. "
              "The defensible claim is \"no network hop\", not a like-for-like inference speedup."))

bar_chart(
    "cost.png",
    "What 1,000 decisions cost",
    "US dollars per 1,000 emails (lower is better)",
    [l for l, _, _ in cost], [v for _, v, _ in cost],
    [GAVEL, OTHER, OTHER2], [t for _, _, t in cost],
    footnote="Gavel: $0 marginal (self-hosted \u2014 your hardware, not literally zero TCO). Jev/Haiku: published list prices.")

bar_chart(
    "ece_scaling_synthetic.png",
    "Calibration actually works",
    "Expected calibration error before/after temperature scaling (lower is better) \u00b7 synthetic data",
    ["before", "after"], [0.0040, 0.0020],
    [OTHER, GAVEL], lambda v: f"{v:.4f}",
    footnote=("Synthetic 6-class benchmark from `cargo run -p decide-bench` (2,000 train / 1,000 test, 15% label noise). "
              "Not real-world data \u2014 it shows the calibration machinery working, nothing more."))
