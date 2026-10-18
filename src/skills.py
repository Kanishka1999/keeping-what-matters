"""Post-trained SKILLS, measured from generated output -- the abilities replay and merging exist
to protect. Earlier channels (ARC, ScienceQA) were letter-choice tasks that test PRETRAINED
knowledge; post-trained skills only show in what the model writes, so these generate.

  instruction  IFEval prompts restricted to instruction types we can check exactly (a faithful
               re-implementation of a subset -- for tracking CHANGE, not leaderboard scores)
  tools        8 tools, templated requests with exactly one correct call; exact name+arg match
  maths        GSM8K, final number
  charts       ChartQA (image input), relaxed accuracy (5% numeric tolerance)

Greedy decoding, fixed item order and batch size: deterministic, so change across checkpoints is
change in the model. Report binomial SE with every score.
"""
from __future__ import annotations
import json, math, random, re
import torch

MAX_NEW = {"instruction": 384, "maths": 320, "tools": 160, "charts": 16}
SUPPORTED_IF = {"punctuation:no_comma", "change_case:english_lowercase", "change_case:english_capital",
    "keywords:forbidden_words", "keywords:existence", "keywords:frequency", "keywords:letter_frequency",
    "detectable_format:title", "detectable_format:number_bullet_lists", "detectable_format:number_highlighted_sections",
    "detectable_format:json_format", "startend:quotation", "startend:end_checker", "length_constraints:number_words",
    "combination:repeat_prompt"}


def _rel(n, target, relation):
    return n < target if relation == "less than" else n >= target


def check_instruction(iid, kw, r):
    s = r.strip()
    if iid == "punctuation:no_comma": return "," not in r
    if iid == "change_case:english_lowercase": return any(c.isalpha() for c in r) and r == r.lower()
    if iid == "change_case:english_capital": return any(c.isalpha() for c in r) and r == r.upper()
    if iid == "keywords:forbidden_words": return not any(re.search(r"\b" + re.escape(w) + r"\b", r, re.I) for w in kw["forbidden_words"])
    if iid == "keywords:existence": return all(re.search(re.escape(k), r, re.I) for k in kw["keywords"])
    if iid == "keywords:frequency": return _rel(len(re.findall(re.escape(kw["keyword"]), r, re.I)), kw["frequency"], kw["relation"])
    if iid == "keywords:letter_frequency": return _rel(r.lower().count(kw["letter"].lower()), kw["let_frequency"], kw["let_relation"])
    if iid == "detectable_format:title": return bool(re.search(r"<<[^\n]+>>", r))
    if iid == "detectable_format:number_bullet_lists":
        return len(re.findall(r"^\s*\*[^\*].*$", r, re.M)) + len(re.findall(r"^\s*-.*$", r, re.M)) == kw["num_bullets"]
    if iid == "detectable_format:number_highlighted_sections":
        hits = [h for h in re.findall(r"\*[^\n\*]*\*", r) if h.strip("*").strip()] + \
               [h for h in re.findall(r"\*\*[^\n\*]*\*\*", r) if h.strip("*").strip()]
        return len(hits) >= kw["num_highlights"]
    if iid == "detectable_format:json_format":
        t = re.sub(r"^```(?:json|Json|JSON)?\s*|\s*```$", "", s)
        try: json.loads(t); return True
        except Exception: return False
    if iid == "startend:quotation": return len(s) > 1 and s[0] == '"' and s[-1] == '"'
    if iid == "startend:end_checker": return s.lower().endswith(kw["end_phrase"].strip().lower())
    if iid == "length_constraints:number_words": return _rel(len(re.findall(r"\w+", r)), kw["num_words"], kw["relation"])
    if iid == "combination:repeat_prompt": return s.lower().startswith(kw["prompt_to_repeat"].strip().lower())
    raise KeyError(iid)


TOOLS = [
    ("get_weather", "Current weather for a city", {"city": "string"}),
    ("convert_currency", "Convert an amount between currencies (ISO codes)", {"amount": "number", "from_currency": "string", "to_currency": "string"}),
    ("search_case_law", "Search case law by topic and jurisdiction", {"query": "string", "jurisdiction": "string"}),
    ("schedule_meeting", "Schedule a meeting (date YYYY-MM-DD, time HH:MM)", {"attendee": "string", "date": "string", "time": "string"}),
    ("translate_text", "Translate text into a target language", {"text": "string", "target_language": "string"}),
    ("get_stock_price", "Latest price for a stock ticker", {"ticker": "string"}),
    ("send_email", "Send an email", {"to": "string", "subject": "string"}),
    ("get_exchange_filing", "Fetch a company's latest regulatory filing by ticker and form type", {"ticker": "string", "form": "string"}),
]
TOOLS_HARD = [   # near-duplicate pairs + light reasoning (unit/currency/form mapping, date +N, 12h -> 24h)
    ("get_weather", "Current weather for a city, unit celsius or fahrenheit", {"city": "string", "unit": "string"}),
    ("get_forecast", "Multi-day weather forecast for a city", {"city": "string", "days": "integer"}),
    ("convert_currency", "Convert an amount between currencies (ISO 4217 codes)", {"amount": "number", "from_currency": "string", "to_currency": "string"}),
    ("get_exchange_rate", "Current exchange rate between two currencies (ISO 4217 codes)", {"from_currency": "string", "to_currency": "string"}),
    ("search_case_law", "Search court decisions by topic, jurisdiction and earliest year", {"query": "string", "jurisdiction": "string", "year_from": "integer"}),
    ("search_statutes", "Search legislation by topic and jurisdiction", {"query": "string", "jurisdiction": "string"}),
    ("schedule_meeting", "Schedule a meeting; date YYYY-MM-DD, time HH:MM in 24-hour clock", {"attendee": "string", "date": "string", "time": "string", "duration_minutes": "integer"}),
    ("cancel_meeting", "Cancel a meeting by its ID", {"meeting_id": "string"}),
    ("get_stock_price", "Latest price for a stock ticker", {"ticker": "string"}),
    ("get_exchange_filing", "Fetch a company filing by ticker, form (10-K annual, 10-Q quarterly) and fiscal year", {"ticker": "string", "form": "string", "year": "integer"}),
    ("send_email", "Send an email", {"to": "string", "subject": "string"}),
    ("translate_text", "Translate text into a target language", {"text": "string", "target_language": "string"}),
]
_schemas = lambda tools: [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object",
                 "properties": {k: {"type": t} for k, t in p.items()}, "required": list(p)}}} for n, d, p in tools]
TOOL_SCHEMAS, TOOL_SCHEMAS_HARD = _schemas(TOOLS), _schemas(TOOLS_HARD)
ALL_TOOL_NAMES = {t[0] for t in TOOLS + TOOLS_HARD}


def _tool_items(n, rng):
    cities = ["Paris", "Nairobi", "Osaka", "Lima", "Toronto", "Cairo", "Oslo", "Perth"]
    people = ["Priya Shah", "Tom Okafor", "Lena Vogel", "Ana Ruiz"]
    gen = [
        lambda: ((c := rng.choice(cities)), f"What's the weather like in {c} right now?", "get_weather", {"city": c})[1:],
        lambda: ((a := rng.choice([50, 120, 999, 2500])), (f := rng.choice(["USD", "EUR", "GBP"])), (t := rng.choice(["JPY", "INR", "CAD"])),
                 f"Convert {a} {f} to {t}.", "convert_currency", {"amount": a, "from_currency": f, "to_currency": t})[3:],
        lambda: ((q := rng.choice(["data privacy", "unfair dismissal", "patent infringement"])), (j := rng.choice(["UK", "US", "EU"])),
                 f"Find {j} case law about {q}.", "search_case_law", {"query": q, "jurisdiction": j})[2:],
        lambda: ((p := rng.choice(people)), (d := rng.choice(["2026-10-03", "2026-11-14"])), (t := rng.choice(["09:30", "14:00"])),
                 f"Book a meeting with {p} on {d} at {t}.", "schedule_meeting", {"attendee": p, "date": d, "time": t})[3:],
        lambda: ((x := rng.choice(["Good morning", "Thank you for your help"])), (l := rng.choice(["French", "Japanese", "Spanish"])),
                 f"Translate '{x}' into {l}.", "translate_text", {"text": x, "target_language": l})[2:],
        lambda: ((k := rng.choice(["AAPL", "TRI", "NVDA"])), f"What's the latest stock price for {k}?", "get_stock_price", {"ticker": k})[1:],
        lambda: ((e := rng.choice(["legal@example.com", "ops@example.com"])), (s := rng.choice(["Contract review", "Q3 filing"])),
                 f"Send an email to {e} with the subject '{s}'.", "send_email", {"to": e, "subject": s})[2:],
        lambda: ((k := rng.choice(["TRI", "MSFT"])), (f := rng.choice(["10-K", "10-Q"])),
                 f"Get the latest {f} filing for {k}.", "get_exchange_filing", {"ticker": k, "form": f})[2:],
    ]
    return [dict(zip(("prompt", "name", "args"), gen[i % len(gen)]())) for i in range(n)]


ALIASES = {"united states": "us", "usa": "us", "united kingdom": "uk", "european union": "eu"}


def _tool_items_hard(n, rng):
    cities = ["Oslo", "Lima", "Osaka", "Nairobi", "Perth", "Toronto"]
    cur_from = [("euros", "EUR"), ("pounds", "GBP"), ("US dollars", "USD")]
    cur_to = [("yen", "JPY"), ("rupees", "INR"), ("Canadian dollars", "CAD")]
    topics = ["unfair dismissal", "data protection", "patent infringement"]
    times = [("2pm", "14:00"), ("9:30am", "09:30"), ("4:15pm", "16:15")]
    offsets = [("the day after", 4), ("two days after", 5)]          # base date 2026-10-03
    def one(i):
        k = i % 12
        if k == 0: c, d = rng.choice(cities), rng.choice([3, 5, 7]); return f"Will it be cold in {c} over the next {d} days?", "get_forecast", {"city": c, "days": d}
        if k == 1: c, u = rng.choice(cities), rng.choice(["Fahrenheit", "Celsius"]); return f"What's the temperature in {c} right now, in {u}?", "get_weather", {"city": c, "unit": u.lower()}
        if k == 2:
            a, (fn, fc), (tn, tc) = rng.choice([80, 999, 1250, 3400]), rng.choice(cur_from), rng.choice(cur_to)
            return f"How much is {a:,} {fn} in {tn}?", "convert_currency", {"amount": a, "from_currency": fc, "to_currency": tc}
        if k == 3: (_, fc), (_, tc) = rng.choice(cur_from), rng.choice(cur_to); return f"What's the current exchange rate from {fc} to {tc}?", "get_exchange_rate", {"from_currency": fc, "to_currency": tc}
        if k == 4:
            q, j, y = rng.choice(topics), rng.choice(["UK", "US", "EU"]), rng.choice([2012, 2015, 2019])
            return f"Find {j} court decisions on {q} from {y} onwards.", "search_case_law", {"query": q, "jurisdiction": j, "year_from": y}
        if k == 5: q, j = rng.choice(topics), rng.choice(["UK", "US", "EU"]); return f"Look up {j} legislation on {q}.", "search_statutes", {"query": q, "jurisdiction": j}
        if k == 6:
            who, (ph, day), (t12, t24), dur = rng.choice(["Ana Ruiz", "Tom Okafor", "Lena Vogel"]), rng.choice(offsets), rng.choice(times), rng.choice([30, 45, 60])
            return f"Set up a {dur}-minute call with {who} {ph} 2026-10-03 at {t12}.", "schedule_meeting", {"attendee": who, "date": f"2026-10-0{day}", "time": t24, "duration_minutes": dur}
        if k == 7: mid = f"MTG-{rng.randint(1000, 9999)}"; return f"Please cancel meeting {mid}.", "cancel_meeting", {"meeting_id": mid}
        if k == 8:
            t, (ph, form), y = rng.choice(["TRI", "MSFT"]), rng.choice([("annual report", "10-K"), ("quarterly report", "10-Q")]), rng.choice([2023, 2024])
            return f"Pull {t}'s {ph} for fiscal year {y}.", "get_exchange_filing", {"ticker": t, "form": form, "year": y}
        if k == 9: t = rng.choice(["NVDA", "AAPL", "TRI"]); return f"How is {t} trading right now?", "get_stock_price", {"ticker": t}
        if k == 10: e, sj = rng.choice(["legal@example.com", "ops@example.com"]), rng.choice(["Contract review", "Q3 filing"]); return f"Send an email to {e} with the subject '{sj}'.", "send_email", {"to": e, "subject": sj}
        x, l = rng.choice(["good evening", "thank you very much"]), rng.choice(["Japanese", "French", "Spanish"])
        return f"How do you say '{x}' in {l}?", "translate_text", {"text": x, "target_language": l}
    return [dict(zip(("prompt", "name", "args"), one(i))) for i in range(n)]


def _norm(v):
    if not isinstance(v, str): return v
    v = v.strip().strip("'\"").lower().rstrip(".")
    return ALIASES.get(v, v)


def parse_tool_call(r):
    """Qwen3.5 emits XML-style calls: <tool_call><function=NAME><parameter=K>V</parameter>...;
    older templates emit JSON. Accept both; return (name, args) or (None, None)."""
    m = re.search(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", r, re.S)
    if m:
        args = {k.strip(): v.strip() for k, v in re.findall(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", m.group(2), re.S)}
        return m.group(1).strip(), args
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", r, re.S) or re.search(r"(\{.*\})", r, re.S)
    if m:
        try:
            call = json.loads(m.group(1)); args = call.get("arguments", {})
            return call.get("name"), (json.loads(args) if isinstance(args, str) else args)
        except Exception:
            pass
    return None, None


def check_tool(item, r):
    name, args = parse_tool_call(r)
    if name is None: return False, False
    valid = name in ALL_TOOL_NAMES
    def same(k, v):
        a = args.get(k)
        if a is None: return False
        if isinstance(v, (int, float)):
            try: return float(str(a).replace(",", "")) == float(v)
            except ValueError: return False
        return _norm(str(a)) == _norm(v)
    return valid and name == item["name"] and all(same(k, v) for k, v in item["args"].items()), valid


def _last_number(r):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", r)
    return float(nums[-1].replace(",", "").rstrip(".")) if nums else None


def check_chart(labels, r):
    p = r.strip().rstrip(".").strip()
    for g in labels:
        try:
            gf, pf = float(g), float(p.replace(",", "").replace("%", ""))
            if (gf == 0 and pf == 0) or (gf != 0 and abs(pf - gf) / abs(gf) <= 0.05): return True
        except ValueError:
            if p.lower() == str(g).strip().lower(): return True
    return False


def build_items(n_if=120, n_math=120, n_tools=64, n_chart=100, seed=0, tool_mode="basic"):
    from datasets import load_dataset
    rng = random.Random(seed); items = {}
    ife = []
    for r in load_dataset("google/IFEval", split="train"):
        ids, kws = r["instruction_id_list"], [{k: v for k, v in (kw or {}).items() if v is not None} for kw in r["kwargs"]]
        if not set(ids) <= SUPPORTED_IF: continue
        if any(i == "length_constraints:number_words" and k.get("relation") == "at least" and k.get("num_words", 0) > 180
               for i, k in zip(ids, kws)): continue          # cannot be satisfied within the generation cap
        ife.append({"prompt": r["prompt"], "ids": ids, "kwargs": kws})
    rng.shuffle(ife); items["instruction"] = ife[:n_if]
    g = list(load_dataset("openai/gsm8k", "main", split="test")); rng.shuffle(g)
    items["maths"] = [{"prompt": x["question"] + "\n\nSolve step by step, then give the final answer on the last line as 'Answer: <number>'.",
                       "gold": float(x["answer"].split("####")[-1].strip().replace(",", ""))} for x in g[:n_math]]
    items["tools"] = (_tool_items_hard if tool_mode == "hard" else _tool_items)(n_tools, rng)
    items["tool_schemas"] = TOOL_SCHEMAS_HARD if tool_mode == "hard" else TOOL_SCHEMAS
    ch = list(load_dataset("HuggingFaceM4/ChartQA", split="test")); rng.shuffle(ch)
    items["charts"] = [{"prompt": x["query"] + "\nAnswer with a single word or number.", "image": x["image"].convert("RGB"),
                        "labels": x["label"]} for x in ch[:n_chart]]
    return items


@torch.no_grad()
def generate(model, proc, prompts, images, max_new, bs, tools=None):
    tok = proc.tokenizer; tok.padding_side = "left"; dev = next(model.parameters()).device; out = []
    for i in range(0, len(prompts), bs):
        ch, im = prompts[i:i + bs], (images[i:i + bs] if images else None)
        msgs = [[{"role": "user", "content": ([{"type": "image"}] if im else []) + [{"type": "text", "text": p}]}] for p in ch]
        texts = [proc.apply_chat_template(m, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        kw = dict(text=texts, return_tensors="pt", padding=True)
        if im: kw["images"] = im
        enc = proc(**kw).to(dev)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, use_cache=True,   # training sets config.use_cache=False
                           pad_token_id=tok.pad_token_id)
        out += [re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
                for t in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out


def read_skills(model, proc, items, bs=16):
    se = lambda p, n: math.sqrt(max(p * (1 - p), 1e-9) / n)
    res, it = {}, items
    r = generate(model, proc, [x["prompt"] for x in it["instruction"]], None, MAX_NEW["instruction"], bs)
    per = [[check_instruction(i, k, t) for i, k in zip(x["ids"], x["kwargs"])] for x, t in zip(it["instruction"], r)]
    p = sum(all(v) for v in per) / len(per); flat = [v for vs in per for v in vs]
    res["instruction_prompt"], res["instruction_level"] = p, sum(flat) / len(flat)
    res["instruction_se"] = se(p, len(per))
    r = generate(model, proc, [x["prompt"] for x in it["maths"]], None, MAX_NEW["maths"], bs)
    m_ok = [(v := _last_number(t)) is not None and abs(v - x["gold"]) < 1e-6 for x, t in zip(it["maths"], r)]
    res["maths"] = sum(m_ok) / len(r)
    res["maths_se"] = se(res["maths"], len(r))
    res["maths_format"] = sum(bool(re.search(r"Answer:\s*\$?-?\d", t)) for t in r) / len(r)   # gave the requested final line
    r = generate(model, proc, [x["prompt"] for x in it["tools"]], None, MAX_NEW["tools"], bs, tools=it.get("tool_schemas", TOOL_SCHEMAS))
    ok = [check_tool(x, t) for x, t in zip(it["tools"], r)]
    res["tools_correct"], res["tools_valid"] = sum(a for a, _ in ok) / len(ok), sum(b for _, b in ok) / len(ok)
    res["tools_se"] = se(res["tools_correct"], len(ok))
    r = generate(model, proc, [x["prompt"] for x in it["charts"]], [x["image"] for x in it["charts"]], MAX_NEW["charts"], max(1, bs // 4))
    c_ok = [check_chart(x["labels"], t) for x, t in zip(it["charts"], r)]
    res["charts"] = sum(c_ok) / len(r)
    res["charts_se"] = se(res["charts"], len(r))
    res["charts_format"] = sum(0 < len(t.strip().split()) <= 3 for t in r) / len(r)          # a single word or number, as asked
    # right/wrong per test item (same items in every run), so runs can be compared item by item (paired tests)
    res["items"] = {"instruction": [bool(all(v)) for v in per], "maths": [bool(x) for x in m_ok],
                    "tools": [bool(a) for a, _ in ok], "charts": [bool(x) for x in c_ok]}
    return res
