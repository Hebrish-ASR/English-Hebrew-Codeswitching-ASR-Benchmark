"""Rewrite the results snapshot in README.md (between the results markers) from results/*.json.
Run after adding a result: python3 scripts/results_table.py"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rev = re.search(r'^REVISION = "(\w+)"', (ROOT / "ehcs_bench/data.py").read_text(), re.M).group(1)
res = [json.loads(p.read_text()) for p in sorted((ROOT / "results").rglob("*.json"))]
res = [r for r in res if r["benchmark"]["revision"] == rev]
p = lambda x: f"{100 * x:.1f}"
key = lambda r: (-r["metrics"]["subsets"]["clean"]["terms"]["all"]["recall"], r["metrics"]["subsets"]["clean"]["wer"])
lines = ["| Model | Type | Term recall % | Unseen-term recall % | Lenient % | WER % | English control WER % |",
         "|---|---|---|---|---|---|---|"]
for r in sorted(res, key=key):
    m, s = r["model"], r["metrics"]["subsets"]
    c = s["clean"]
    name = f"[{m['name']}]({m['url']})" if m.get("url") else m["name"]
    if m.get("contamination"):
        name += " ⚠"
    lines.append(f"| {name} | {'CS fine-tune' if m.get('fine_tuned_for_code_switching') else 'Stock'} | "
                 f"**{p(c['terms']['all']['recall'])}** | {p(c['terms']['unseen']['recall'])} | "
                 f"{p(c['terms']['all']['recall_lenient'])} | {p(c['wer'])} | {p(s['english_control']['wer'])} |")
readme = ROOT / "README.md"
t = readme.read_text()
t = re.sub(r"(<!-- results:start -->\n).*?(<!-- results:end -->)", lambda m: m.group(1) + "\n".join(lines) + "\n" + m.group(2),
           t, flags=re.S)
readme.write_text(t)
print("\n".join(lines))
