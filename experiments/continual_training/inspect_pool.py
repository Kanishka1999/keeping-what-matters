"""Inspect a W2 replay pool: gate, truncation, correctness of the model's own answers, samples."""
import json, re, sys, textwrap
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import skills as S
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
D = json.load(open(sys.argv[1])); P = D["families"]
CAP = {"chat": 256, "instruction": 384, "maths": 320, "tools": 160, "charts": 16}
print("built", D.get("built"), "| device", D.get("device"))
print("gate:", {f: f"{g['check']} {g['value']:.2f} (need {g['need']})" for f, g in D.get("gate", {}).items()})
for f, v in P.items():
    xs = v["train"] + v["probe"]
    dec = [tok.decode(x["own_ids"], skip_special_tokens=False) for x in xs]
    trunc = sum(len(x["own_ids"]) >= CAP[f] for x in xs) / len(xs)
    lens = sorted(len(x["own_ids"]) for x in xs)
    line = f"\n=== {f}: {len(xs)} answers | length median {lens[len(lens) // 2]}, max {lens[-1]} | hit the {CAP[f]}-token cap: {trunc:.0%}"
    if f == "maths":
        gold = [float(re.findall(r"Answer:\s*(-?[\d,\.]+)", x["ref_text"])[-1].replace(",", "").rstrip(".")) for x in xs]
        own = [S._last_number(d) for d in dec]
        acc = sum(o is not None and abs(o - g) < 1e-6 for o, g in zip(own, gold)) / len(xs)
        line += f" | 'Answer:' line {sum('Answer:' in d for d in dec) / len(xs):.0%} | own answer correct {acc:.0%}"
    if f == "charts":
        ok = sum(S.check_chart([x["ref_text"].replace("<|im_end|>", "").strip()], d.replace("<|im_end|>", "")) for x, d in zip(xs, dec)) / len(xs)
        line += f" | own answer matches label {ok:.0%}"
    if f == "tools":
        calls = [S.parse_tool_call(d) for d in dec]
        refs = [S.parse_tool_call(x["ref_text"]) for x in xs]
        valid = sum(c[0] is not None for c in calls) / len(xs)
        right = sum(c[0] == r[0] and c[1] is not None and all(str(c[1].get(k, "")).strip().lower() == str(vv).strip().lower() for k, vv in r[1].items())
                    for c, r in zip(calls, refs)) / len(xs)
        line += f" | parseable call {valid:.0%} | exactly the intended call {right:.0%} | has <tool_call> tag {sum('<tool_call>' in d for d in dec) / len(xs):.0%}"
    if f in ("chat", "instruction"):
        line += f" | contains '<think>' {sum('<think>' in d for d in dec) / len(xs):.0%}"
    print(line)
    for x, d in list(zip(xs, dec))[:2]:
        print("  PROMPT:", textwrap.shorten(x["prompt"], 110))
        print("  QWEN:  ", textwrap.shorten(d.replace("\n", " / "), 220))
