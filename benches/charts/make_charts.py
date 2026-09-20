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
              footnote=None, yerr=None, logy=False, ylabel=None,
              figsize=(8, 4.6), xtick_fontsize=11):
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="white",
                  yerr=yerr, capsize=6, ecolor="#374151",
                  error_kw={"elinewidth": 1.5})
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=xtick_fontsize)
    ax.set_title(title, loc="left", pad=44)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10.5,
                color="#4b5563", va="bottom", ha="left")
    if ylabel:
        ax.set_ylabel(ylabel, color="#4b5563")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # value labels: inside tall bars (white), above short bars (black) —
    # keeps them clear of the subtitle line at the top of the axes
    labels_txt = [fmt(v) for v in values] if callable(fmt) else list(fmt)
    ymax = max(v for v in values if v > 0) if any(v > 0 for v in values) else 1
    for b, v, txt in zip(bars, values, labels_txt):
        h = b.get_height()
        cx = b.get_x() + b.get_width() / 2
        if logy:
            # log scale: labels sit above the bar; give the axes headroom so
            # they never touch the subtitle
            ax.text(cx, h * 1.4, txt, ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        elif h > 0.18 * ymax:
            ax.text(cx, h * 0.94, txt, ha="center", va="top",
                    fontsize=11, fontweight="bold", color="white")
        else:
            ax.text(cx, h + 0.03 * ymax, txt, ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
    if logy:
        ax.set_ylim(top=ymax * 2.6)
    if footnote:
        fig.text(0.02, 0.025, footnote, fontsize=7.5, color="#6b7280",
                 ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
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

NAVY = "#1e3a8a"   # full-data baselines

# --- label-efficiency experiment (2026-09-20, issue-triage task) ---
# SetFit (all-MiniLM-L6-v2, num_iterations=5, 1 epoch, seed 42) vs TF-IDF
# trained on the exact same 8/16/32-per-class subsamples; full-data TF-IDF
# and frozen MiniLM + logreg trained on all 5,832 examples.
# Same 1,250-issue test set throughout; latencies are 2-thread CPU wall-clock.
few_labels = ["SetFit 8", "TF-IDF 8",
              "SetFit 16", "TF-IDF 16",
              "SetFit 32", "TF-IDF 32",
              "frozen MiniLM", "TF-IDF full"]
few_colors = [GAVEL, OTHER, GAVEL, OTHER, GAVEL, OTHER, OTHER2, NAVY]
few_acc = [59.28, 66.16, 63.92, 62.48, 72.88, 68.48, 78.16, 82.72]
few_train_s = [297, 0.8, 536, 0.8, 1064, 0.9, 788.6, 13.1]
few_train_txt = ["297 s", "0.8 s", "536 s", "0.8 s", "1,064 s", "0.9 s",
                 "789 s", "13.1 s"]
few_lat_ms = [179.6, 0.0018, 177.6, 0.0023, 175.9, 0.0023, 96.5, 1.1]
few_lat_txt = ["180 ms", "0.0018 ms", "178 ms", "0.0023 ms", "176 ms",
               "0.0023 ms", "96.5 ms", "1.1 ms"]

bar_chart(
    "fewshot_accuracy.png",
    "What tens of labels buy you",
    "Test accuracy on the same 1,250 issues, % correct (higher is better)\n"
    "blue = SetFit (fine-tuned MiniLM) \u00b7 gray = same-budget TF-IDF \u00b7 "
    "dark gray = frozen MiniLM \u00b7 navy = full-data TF-IDF baseline",
    few_labels, few_acc, few_colors, lambda v: f"{v:.1f}%",
    footnote=("SetFit: all-MiniLM-L6-v2, num_iterations=5 (not the default 20), "
              "1 epoch, single seed 42 \u2014 one noisy draw, not a mean. "
              "TF-IDF rows use the same subsamples; same 1,250-issue test set; "
              "random split, duplicate leakage unaudited."),
    figsize=(9.6, 4.8), xtick_fontsize=10)

bar_chart(
    "fewshot_cost.png",
    "What those extra points cost",
    "Training wall-clock, seconds (lower is better) \u00b7 log scale\n"
    "blue = SetFit \u00b7 gray = same-budget TF-IDF \u00b7 "
    "dark gray = frozen MiniLM \u00b7 navy = full-data TF-IDF baseline",
    few_labels, few_train_s, few_colors, few_train_txt,
    logy=True,
    footnote=("Wall-clock on 2 CPU threads; frozen MiniLM = 788 s encoding + 0.6 s fitting. "
              "Relative ordering is robust; exact seconds are not."),
    figsize=(9.6, 4.8), xtick_fontsize=10)

bar_chart(
    "fewshot_latency.png",
    "The price you pay at decision time",
    "Per-question latency, milliseconds (lower is better) \u00b7 log scale\n"
    "blue = SetFit \u00b7 gray = same-budget TF-IDF \u00b7 "
    "dark gray = frozen MiniLM \u00b7 navy = full-data TF-IDF baseline",
    few_labels, few_lat_ms, few_colors, few_lat_txt,
    logy=True,
    footnote=("Encode+predict over 1,250 test questions (200 for frozen MiniLM); "
              "TF-IDF full-data from the flagship harness; 2-thread CPU; "
              "TF-IDF rows are scikit-learn in-process, SetFit/frozen are encoder-bound."),
    figsize=(9.6, 4.8), xtick_fontsize=10)

bar_chart(
    "ece_scaling_synthetic.png",
    "Calibration actually works",
    "Expected calibration error before/after temperature scaling (lower is better) \u00b7 synthetic data",
    ["before", "after"], [0.0040, 0.0020],
    [OTHER, GAVEL], lambda v: f"{v:.4f}",
    footnote=("Synthetic 6-class benchmark from `cargo run -p decide-bench` (2,000 train / 1,000 test, 15% label noise). "
              "Not real-world data \u2014 it shows the calibration machinery working, nothing more."))
