"""Leaderboard for the English-Hebrew code-switching ASR benchmark.

Reads the result files in ./results (deployed alongside this app from the GitHub repo's
results/ folder by scripts/deploy_space.sh). No network calls, no model inference: every
number shown is in a result file that anyone can rescore with `ehcs-bench rescore`.
"""
import html
import json
import re
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent
BENCH = "danielrosehill/English-Hebrew-Codeswitching-ASR-Benchmark"
GITHUB = "https://github.com/danielrosehill/English-Hebrew-Codeswitching-ASR-Benchmark"
REVISION = json.loads((ROOT / "benchmark.json").read_text())["revision"]
BLOG = "https://blog.danielrosehill.com/posts/english-hebrew-code-switching-asr-benchmark"
LINKS = " · ".join([
    f"[Benchmark repo]({GITHUB})",
    f"[Inference code]({GITHUB}/blob/main/ehcs_bench/adapters.py) ([Modal GPU runner]({GITHUB}/blob/main/modal_app.py))",
    f"[Scorer]({GITHUB}/blob/main/ehcs_bench/score.py)",
    f"[Test set](https://huggingface.co/datasets/{BENCH})",
    f"[Result files]({GITHUB}/tree/main/results)",
    f"[Write-up]({BLOG})",
])


def load():
    out = []
    for p in sorted((ROOT / "results").glob("*.json")):
        r = json.loads(p.read_text())
        if r["benchmark"]["revision"] == REVISION:
            out.append(r)
    return out


RESULTS = load()
pct = lambda x: None if x is None else round(100 * x, 1)


def link(m):
    return f"[{m['name']}]({m['url']})" if m.get("url") else m["name"]


def table(subset):
    rows = []
    for r in RESULTS:
        m, s = r["model"], r["metrics"]["subsets"]
        c = s[subset]
        rows.append({
            "Model": link(m),
            "Type": "Code-switching fine-tune" if m.get("fine_tuned_for_code_switching") else "Stock",
            "Term recall %": pct(c["terms"]["all"]["recall"]),
            "Unseen-term recall %": pct(c["terms"]["unseen"]["recall"]),
            "Lenient recall %": pct(c["terms"]["all"]["recall_lenient"]),
            "Exact-case recall %": pct(c["terms"]["all"]["recall_exact_case"]),
            "WER %": pct(c["wer"]),
            "CER %": pct(c["cer"]),
            "English control WER %": pct(s["english_control"]["wer"]),
            "Params (M)": m.get("params_m"),
            "Caveat": "⚠ " + m["contamination"] if m.get("contamination") else "",
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["Term recall %", "WER %"], ascending=[False, True]).reset_index(drop=True)
        df.insert(0, "#", range(1, len(df) + 1))
    return df


# Charts. Colour carries one job per chart: model type (fine-tune vs stock), or metric
# (term recall / unseen / lenient), or magnitude (heatmap, one-hue blue ramp).
FT, STOCK = "#2a78d6", "#8d96a8"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#6b7280"
BLUES = [[0, "#eef4fc"], [0.5, "#6da7ec"], [1, "#104281"]]


def _layout(fig, title, x=None, y=None, height=420, **kw):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height, font=dict(color=INK),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"), **kw)
    fig.update_xaxes(title=x, gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_yaxes(title=y, gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


def _ranked():
    return sorted(RESULTS, key=lambda r: r["metrics"]["subsets"]["clean"]["terms"]["all"]["recall"])


def _label(r):
    return r["model"]["name"] + (" ⚠" if r["model"].get("contamination") else "")


def fig_recall():
    rs = _ranked()
    fig = go.Figure()
    for (key, field, name), colour in zip([("all", "recall", "Term recall"), ("unseen", "recall", "Unseen-term recall"),
                                           ("all", "recall_lenient", "Lenient recall")], SERIES):
        v = [pct(r["metrics"]["subsets"]["clean"]["terms"][key][field]) for r in rs]
        fig.add_bar(y=[_label(r) for r in rs], x=v, name=name, orientation="h", marker_color=colour,
                    hovertemplate="%{y}<br>" + name + ": %{x:.1f}%<extra></extra>")
    # Horizontal groups draw the first trace at the bottom; reverse the legend to match.
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, legend_traceorder="reversed")
    return _layout(fig, "Hebrew terms transcribed correctly, clean subset", x="%", height=520)


def fig_scatter():
    fig = go.Figure()
    for ft, colour, name in [(True, FT, "Code-switching fine-tune"), (False, STOCK, "Stock")]:
        rs = [r for r in RESULTS if bool(r["model"].get("fine_tuned_for_code_switching")) == ft]
        c = [r["metrics"]["subsets"]["clean"] for r in rs]
        ys = [pct(s["terms"]["all"]["recall"]) for s in c]
        # A point just below another prints its label below itself, so the two do not collide.
        pos = ["bottom right" if any(0 < y2 - y < 3 for y2 in ys) else "middle right" for y in ys]
        fig.add_scatter(x=[pct(s["wer"]) for s in c], y=ys,
                        mode="markers+text", name=name, text=[_label(r) for r in rs], textposition=pos,
                        marker=dict(size=12, color=colour, line=dict(width=2, color="white")),
                        hovertemplate="%{text}<br>WER %{x:.1f}%<br>Term recall %{y:.1f}%<extra></extra>")
    fig.update_xaxes(range=[3, 16])
    return _layout(fig, "Lower WER goes with better Hebrew terms (top left is best)", x="WER % (clean)",
                   y="Term recall %")


def fig_control():
    rs = sorted(RESULTS, key=lambda r: -r["metrics"]["subsets"]["english_control"]["wer"])
    v = [pct(r["metrics"]["subsets"]["english_control"]["wer"]) for r in rs]
    fig = go.Figure(go.Bar(y=[_label(r) for r in rs], x=v, orientation="h", text=[f"{x:.1f}%" for x in v],
                           textposition="outside", cliponaxis=False,
                           marker_color=[FT if r["model"].get("fine_tuned_for_code_switching") else STOCK for r in rs],
                           hovertemplate="%{y}<br>English control WER %{x:.1f}%<extra></extra>"))
    return _layout(fig, "English-only control (24 clips): WER, lower is better. Blue: fine-tune, grey: stock",
                   x="WER %", height=380)


def fig_heatmap(min_n):
    rs = _ranked()[::-1]
    per = rs[0]["metrics"]["per_term"] if rs else {}
    terms = [t for t, d in per.items() if d["n"] >= min_n]
    hits = {t: [r["metrics"]["per_term"][t]["strict"] / r["metrics"]["per_term"][t]["n"] for r in rs] for t in terms}
    terms.sort(key=lambda t: (sum(hits[t]), t.lower()))
    z = [hits[t] for t in terms]
    txt = [[f"{r['metrics']['per_term'][t]['strict']}/{r['metrics']['per_term'][t]['n']}" for r in rs] for t in terms]
    fig = go.Figure(go.Heatmap(z=z, x=[_label(r) for r in rs], y=terms, text=txt, zmin=0, zmax=1, colorscale=BLUES,
                               xgap=2, ygap=2, colorbar=dict(title="Hit rate", tickformat=".0%"),
                               hovertemplate="%{y} · %{x}<br>%{text} occurrences written correctly<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    fig.update_xaxes(tickangle=-30, side="top")
    fig = _layout(fig, f"Per term, hardest first ({len(terms)} terms)", height=max(400, 18 * len(terms) + 220))
    fig.update_layout(margin=dict(t=190))
    return fig


# Archive: one file per published round in ./archive (scripts/snapshot_run.py). Unlike the
# leaderboard, it is not filtered to the current test-set revision.
def load_archive():
    return sorted((json.loads(p.read_text()) for p in (ROOT / "archive").glob("*.json")),
                  key=lambda a: a["date"], reverse=True)


def archive_md():
    out = []
    for a in load_archive():
        rev = a["benchmark"]["revision"]
        out.append(f"### {a['date']}: {a['title']}\n\n{a['notes']}\n\nTest set **{a['benchmark']['version']}** "
                   f"(dataset commit [`{rev[:12]}`](https://huggingface.co/datasets/{BENCH}/tree/{rev})). Full result "
                   f"files with every transcript: [`{a['repo_commit'][:7]}`]({GITHUB}/tree/{a['repo_commit']}/results)."
                   + ("" if rev == REVISION else " *Scored on an older test set; not comparable with the current leaderboard.*")
                   + "\n\n| # | Model | Type | Term recall % | Unseen % | Lenient % | WER % | English control WER % |\n"
                   "|---|---|---|---|---|---|---|---|")
        for i, m in enumerate(a["models"], 1):
            c, e = m["subsets"]["clean"], m["subsets"]["english_control"]
            name = (f"[{m['name']}]({m['url']})" if m.get("url") else m["name"]) + (" ⚠" if m.get("contamination") else "")
            out.append(f"| {i} | {name} | {'CS fine-tune' if m.get('fine_tuned_for_code_switching') else 'Stock'} | "
                       f"**{pct(c['terms']['all']['recall'])}** | {pct(c['terms']['unseen']['recall'])} | "
                       f"{pct(c['terms']['all']['recall_lenient'])} | {pct(c['wer'])} | {pct(e['wer'])} |")
        out.append("")
    return "\n".join(out) or "No rounds archived yet."


def per_term():
    names = [r["model"]["name"] for r in RESULTS]
    terms = {}
    for r in RESULTS:
        for t, d in r["metrics"]["per_term"].items():
            row = terms.setdefault(t, {"Term": t, "Unseen": "yes" if d["unseen"] else "", "n": d["n"]})
            row[r["model"]["name"]] = f"{d['strict']}/{d['n']}"
    df = pd.DataFrame(list(terms.values()), columns=["Term", "Unseen", "n", *names])
    hits = lambda row: sum(int(str(v).split("/")[0]) for k, v in row.items() if k in names and isinstance(v, str) and "/" in v)
    if len(df):
        df["_h"] = df.apply(hits, axis=1)
        df = df.sort_values(["_h", "n", "Term"], ascending=[True, False, True]).drop(columns="_h")
    return df


def highlight(text, terms, hit_test):
    out = html.escape(text, quote=False)
    for t in sorted(terms, key=len, reverse=True):
        rx = r"(?<![\w])" + r"[\s-]+".join(re.escape(p) for p in re.split(r"[\s-]+", t)) + r"(?![\w])"
        out = re.sub(rx, lambda m: f"<mark class='{'hit' if hit_test(t) else 'miss'}'>{m.group(0)}</mark>", out, flags=re.I)
    return out


def utterances(model_names, subset, only_misses):
    meta = {}
    p = ROOT / "metadata.jsonl"
    for l in p.read_text().splitlines():
        j = json.loads(l)
        meta[j["id"]] = j
    chosen = [r for r in RESULTS if r["model"]["name"] in (model_names or [])]
    if not chosen:
        return "<p>Pick one or more models.</p>"
    hyp = {r["model"]["name"]: {u["id"]: u["hyp"] for u in r["utterances"]} for r in chosen}
    parts = []
    for uid, m in meta.items():
        if subset != "all" and m["subset"] != subset:
            continue
        terms = [t["term"] for t in m["hebrew_terms"]]
        rx = {t: re.compile(r"(?<![\w])" + r"[\s-]+".join(re.escape(p) for p in re.split(r"[\s-]+", t)) + r"(?![\w])", re.I) for t in terms}
        lines, any_miss = [], False
        for name in hyp:
            y = hyp[name].get(uid, "")
            ok = lambda t, y=y: bool(rx[t].search(y))
            any_miss |= not all(ok(t) for t in terms)
            lines.append(f"<div class='hyp'><b>{html.escape(name)}</b>: {highlight(y, terms, ok)}</div>")
        if only_misses and not any_miss:
            continue
        ref = highlight(m["text"], terms, lambda t: True)
        parts.append(f"<div class='utt'><div class='ref'><span class='id'>{uid} · {m['subset']}</span><br>{ref}</div>{''.join(lines)}</div>")
    return "".join(parts) or "<p>Nothing to show.</p>"


CSS = """
mark.hit{background:#c8f7c5;color:#000} mark.miss{background:#ffc9c9;color:#000}
.utt{border-bottom:1px solid #8884;padding:10px 0} .ref{margin-bottom:4px} .id{font-size:.8em;opacity:.6}
.hyp{margin-left:1em;font-size:.95em}
"""

ABOUT = (ROOT / "ABOUT.md").read_text()
DICTIONARY = (ROOT / "DATA_DICTIONARY.md").read_text()

with gr.Blocks(title="English-Hebrew Code-Switching ASR Leaderboard", css=CSS) as demo:
    gr.Markdown(f"""# English-Hebrew Code-Switching ASR Leaderboard

How well does speech recognition handle **English with Hebrew words in it**: *"pick up some bamba from the makolet"*, *"I have to go to Bituach Leumi"*? Models are scored on a frozen test set ([`{BENCH}`](https://huggingface.co/datasets/{BENCH}), commit `{REVISION[:12]}`) with the open runner at [GitHub]({GITHUB}).

**Term recall** is the share of Hebrew words written with the canonical romanized spelling (case-insensitive, whole word). It is the headline: a model can have a low WER and still mangle every Hebrew word. **Unseen-term** recall counts only Hebrew words absent from the public training split. Single speaker, read speech: see *About* before drawing conclusions.

> **Status: early-stage, work in progress (test set v1).** Known issue: 22 of the 139 labelled terms are English tech words or product names, not Hebrew (*JSON*, *grep*, *Proxmox*, ...). Ranks are unaffected; absolute term recall is 2 to 6 points high. v1.1 will relabel them and rescore every model from its stored transcripts. Contributions welcome: more speakers, better labels, more models.

{LINKS}""")
    with gr.Tab("Leaderboard (clean, 84 clips)"):
        gr.Dataframe(table("clean"), datatype="markdown", wrap=True, interactive=False)
        gr.Markdown("Ranked by term recall, then WER. *Lenient* also accepts lexicon spelling variants (*tsaharon* for *tzaharon*), which is fairer to models never taught the canonical spelling. *English control* is 24 clips of the same speaker with no Hebrew: a fine-tune that wrecks English shows up there.")
    with gr.Tab("All code-switching (121 clips)"):
        gr.Markdown("Adds 37 clips excluded from the headline: 33 overlap Whisper-Hebrish's training data (26 identical recordings, 7 re-recordings of training sentences) and 4 are older desktop-mic recordings. Comparable with the numbers on the Parakeet fine-tune's model card, which used all 121. **Not a fair comparison for Whisper-Hebrish.**")
        gr.Dataframe(table("all_code_switch"), datatype="markdown", wrap=True, interactive=False)
    with gr.Tab("Charts"):
        gr.Markdown("Clean subset (84 clips, 172 Hebrew term occurrences) unless stated. Hover for values. "
                    "With 172 occurrences, a gap of 2 to 3 points is about 4 to 5 words: treat it as noise.")
        gr.Plot(fig_recall())
        gr.Plot(fig_scatter())
        gr.Plot(fig_control())
        n = gr.Radio([("Terms that occur 2+ times", 2), ("All 139 terms", 1)], value=2, label="Heatmap")
        heat = gr.Plot(fig_heatmap(2))
        n.change(fig_heatmap, n, heat)
    with gr.Tab("Per term"):
        gr.Markdown("Hits / occurrences for every Hebrew term in the clean subset. Hardest terms first.")
        gr.Dataframe(per_term(), wrap=True, interactive=False)
    with gr.Tab("Transcripts"):
        names = [r["model"]["name"] for r in RESULTS]
        with gr.Row():
            pick = gr.Dropdown(names, value=names[:2], multiselect=True, label="Models")
            sub = gr.Dropdown(["clean", "all", "contaminated_text", "contaminated_audio", "desktop_mic", "english_control"], value="clean", label="Subset")
            miss = gr.Checkbox(True, label="Only clips where a selected model missed a term")
        out = gr.HTML()
        for c in (pick, sub, miss):
            c.change(utterances, [pick, sub, miss], out)
        demo.load(utterances, [pick, sub, miss], out)
    with gr.Tab("Archive"):
        gr.Markdown("Every published benchmark round, newest first. A round is frozen when it is published; "
                    "the leaderboard tabs always show the latest result per model.\n\n" + archive_md())
    with gr.Tab("Data dictionary"):
        gr.Markdown(DICTIONARY)
    with gr.Tab("About & submit"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()
