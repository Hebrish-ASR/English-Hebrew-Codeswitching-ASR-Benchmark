"""Static PNG charts from results/*.json, for the README, blog and HF article.
The Space draws the same charts interactively with plotly (space/app.py).

    uv run --with matplotlib python scripts/make_figures.py [--out docs/figures]
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FT, STOCK = "#2a78d6", "#8d96a8"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#1f2937", "#6b7280"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150})
FOOT = "Benchmark v1, clean subset: 84 clips, 172 Hebrew term occurrences, one speaker."


def load():
    rev = re.search(r'^REVISION = "(\w+)"', (ROOT / "ehcs_bench/data.py").read_text(), re.M).group(1)
    rs = [json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))]
    rs = [r for r in rs if r["benchmark"]["revision"] == rev]
    return sorted(rs, key=lambda r: r["metrics"]["subsets"]["clean"]["terms"]["all"]["recall"])


label = lambda r: r["model"]["name"] + (" *" if r["model"].get("contamination") else "")
ft = lambda r: bool(r["model"].get("fine_tuned_for_code_switching"))
clean = lambda r: r["metrics"]["subsets"]["clean"]


def finish(fig, out, note=FOOT):
    fig.text(0.01, -0.01, note, fontsize=9, color=MUTED, va="top")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


def term_recall(rs, out):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    v = [100 * clean(r)["terms"]["all"]["recall"] for r in rs]
    ax.barh([label(r) for r in rs], v, color=[FT if ft(r) else STOCK for r in rs], height=0.72)
    for i, x in enumerate(v):
        ax.text(x + 0.8, i, f"{x:.1f}%", va="center")
    ax.set_xlim(0, 80)
    ax.set_xlabel("Hebrew words transcribed correctly (term recall, %)")
    ax.set_title("Benchmark v1, clean subset: 84 clips, 172 Hebrew words", loc="left")
    finish(fig, out / "term_recall.png", "Blue: fine-tuned for code-switching. Grey: stock. "
           "* Training-data overlap; overlapping clips excluded from this subset.")


def breakdown(rs, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    h = 0.26
    for k, ((key, field, name), c) in enumerate(zip([("all", "recall", "Term recall"),
                                                     ("unseen", "recall", "Unseen-term recall"),
                                                     ("all", "recall_lenient", "Lenient recall")], SERIES)):
        v = [100 * clean(r)["terms"][key][field] for r in rs]
        ax.barh([i + (1 - k) * h for i in range(len(rs))], v, height=h - 0.03, color=c, label=name)
    ax.set_yticks(range(len(rs)), [label(r) for r in rs])
    ax.set_xlim(0, 80)
    ax.set_xlabel("%")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Term recall three ways: canonical spelling, unseen terms only, spelling variants accepted", loc="left")
    finish(fig, out / "recall_breakdown.png", FOOT + " * Training-data overlap.")


def scatter(rs, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pts = [(100 * clean(r)["wer"], 100 * clean(r)["terms"]["all"]["recall"], r) for r in rs]
    for x, y, r in pts:
        ax.scatter(x, y, s=90, color=FT if ft(r) else STOCK, edgecolor="white", linewidth=2, zorder=3)
        # Two points this close would print their labels on top of each other: put the lower one's below.
        crowded = any(abs(x - x2) < 2 and 0 < y2 - y < 3 for x2, y2, _ in pts)
        ax.annotate(label(r), (x, y), xytext=(8, -16 if crowded else -4), textcoords="offset points", fontsize=10)
    ax.scatter([], [], color=FT, label="Code-switching fine-tune")
    ax.scatter([], [], color=STOCK, label="Stock")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(3, 14)
    ax.set_ylim(25, 72)
    ax.grid(color="#e5e7eb", zorder=0)
    ax.set_xlabel("WER % (lower is better)")
    ax.set_ylabel("Term recall % (higher is better)")
    ax.set_title("WER and Hebrew term recall move together; top left is best", loc="left")
    finish(fig, out / "recall_vs_wer.png", FOOT + " * Training-data overlap.")


def control(rs, out):
    rs = sorted(rs, key=lambda r: -r["metrics"]["subsets"]["english_control"]["wer"])
    v = [100 * r["metrics"]["subsets"]["english_control"]["wer"] for r in rs]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh([label(r) for r in rs], v, color=[FT if ft(r) else STOCK for r in rs], height=0.72)
    for i, x in enumerate(v):
        ax.text(x + 0.08, i, f"{x:.1f}%", va="center")
    ax.set_xlim(0, 6)
    ax.set_xlabel("WER % on English-only clips (lower is better)")
    ax.set_title("English control: did the fine-tune damage plain English?", loc="left")
    finish(fig, out / "english_control.png", "24 English-only clips (3.3 min), same speaker. "
           "Blue: fine-tuned for code-switching. Grey: stock.")


def heatmap(rs, out):
    rs = rs[::-1]
    per = rs[0]["metrics"]["per_term"]
    terms = [t for t, d in per.items() if d["n"] >= 2]
    hit = {t: [r["metrics"]["per_term"][t]["strict"] / r["metrics"]["per_term"][t]["n"] for r in rs] for t in terms}
    terms.sort(key=lambda t: (sum(hit[t]), t.lower()))
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(terms) + 2.4))
    im = ax.imshow([hit[t] for t in terms], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i, t in enumerate(terms):
        for j, r in enumerate(rs):
            d = r["metrics"]["per_term"][t]
            ax.text(j, i, f"{d['strict']}/{d['n']}", ha="center", va="center", fontsize=9,
                    color="white" if hit[t][j] > 0.6 else INK)
    ax.set_yticks(range(len(terms)), terms)
    ax.set_xticks(range(len(rs)), [label(r) for r in rs], rotation=30, ha="left")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=lambda x, _: f"{x:.0%}").outline.set_visible(False)
    ax.set_title("Hebrew terms that occur 2+ times: correct / occurrences, hardest first", loc="left", pad=12)
    finish(fig, out / "per_term_heatmap.png", FOOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/figures"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rs = load()
    for f in (term_recall, breakdown, scatter, control, heatmap):
        f(rs, out)


if __name__ == "__main__":
    main()
