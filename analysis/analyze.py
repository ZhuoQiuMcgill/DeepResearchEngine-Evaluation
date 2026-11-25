"""Score the four judges' grades and build the comparison matrices. Standard library only.

Usage (from the repository root): python analysis/analyze.py

Reads judgments/<judge>/*.jsonl, rubrics/*.json and analysis/runs.csv (the unblinding key
with run metrics). Writes analysis/out/*.csv and analysis/results.md.

Score of a report (ResearchRubrics): sum(weight * score) / sum(positive weights), with
score 1 / 0.5 / 0 for Satisfied / Partially Satisfied / Not Satisfied. A negative-weight
criterion that is satisfied is a deduction, so a score can fall below 0.
Axis score: the same formula over the criteria of one axis (positive weights of that axis
as the denominator); undefined when the rubric has no positive criterion on that axis.
"""
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "out"
JUDGES = ["claude-code", "claude-code-fable", "codex", "antigravity"]
# One judge per model family. Opus 5.5 shares Fable 5.2's family (kappa 0.85 between them),
# so it is left out of the primary mean and reported as a robustness check.
PRIMARY = ["claude-code-fable", "codex", "antigravity"]
JUDGE_LABEL = {"claude-code": "Opus 5.5", "claude-code-fable": "Fable 5.2", "codex": "Astra", "antigravity": "Neon"}
MODELS = ["haiku", "luna", "deepseek"]
MODEL_LABEL = {"haiku": "Haiku 5.5", "luna": "GPT-6 Luna", "deepseek": "DeepSeek V4.1 Flash"}
AXES = ["Explicit Criteria", "Implicit Criteria", "Synthesis of Information", "Instruction Following",
        "Communication Quality", "References & Citation Quality"]
GROUPS = {"main (Haiku + Luna)": ["haiku", "luna"], "DeepSeek (separate)": ["deepseek"]}
SEED = 691


def load():
    runs = {r["report_id"]: r for r in csv.DictReader(open(ROOT / "analysis" / "runs.csv", encoding="utf-8"))}
    rubrics = {p.stem: {c["index"]: c for c in json.load(open(p, encoding="utf-8"))["criteria"]}
               for p in sorted((ROOT / "rubrics").glob("*.json"))}
    grades = {}  # (judge, report_id, criterion_index) -> score
    for j in JUDGES:
        for f in sorted((ROOT / "judgments" / j).glob("P*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    grades[(j, r["report_id"], r["criterion_index"])] = float(r["score"])
    return runs, rubrics, grades


def weighted(criteria, scores):
    pos = sum(c["weight"] for c in criteria if c["weight"] > 0)
    if pos == 0:
        return None
    return sum(c["weight"] * s for c, s in zip(criteria, scores)) / pos


def report_scores(runs, rubrics, grades):
    out = {}  # (judge, report_id) -> {"total": x, axis: y}
    for j in JUDGES:
        for rid, run in runs.items():
            crit = rubrics[run["rubric_id"]]
            idx = sorted(crit)
            row = {"total": weighted([crit[i] for i in idx], [grades[(j, rid, i)] for i in idx])}
            for a in AXES:
                sub = [i for i in idx if crit[i]["axis"] == a]
                row[a] = weighted([crit[i] for i in sub], [grades[(j, rid, i)] for i in sub]) if sub else None
            out[(j, rid)] = row
    return out


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_ci(values, reps=10000, seed=SEED):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def kappa(a, b, cats=(0.0, 0.5, 1.0), weights="none"):
    n = len(a)
    w = lambda x, y: (0.0 if x == y else 1.0) if weights == "none" else abs(x - y) / (cats[-1] - cats[0])
    obs = sum(w(x, y) for x, y in zip(a, b)) / n
    pa = {c: a.count(c) / n for c in cats}
    pb = {c: b.count(c) / n for c in cats}
    exp = sum(pa[x] * pb[y] * w(x, y) for x in cats for y in cats)
    return 1 - obs / exp if exp else float("nan")


def krippendorff_interval(units):
    """units: list of lists of values (one list per item, one value per coder)."""
    pairable = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in pairable)
    do = sum(sum((x - y) ** 2 for i, x in enumerate(u) for j, y in enumerate(u) if i != j) / (len(u) - 1) for u in pairable) / n
    allv = [v for u in pairable for v in u]
    counts = defaultdict(int)
    for v in allv:
        counts[v] += 1
    de = sum(counts[x] * counts[y] * (x - y) ** 2 for x in counts for y in counts) / (n * (n - 1))
    return 1 - do / de


def pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def fmt(x, nd=3):
    return "–" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) and x < 0 else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


MODEL_SHORT = {"haiku": "Haiku", "luna": "Luna", "deepseek": "DeepSeek"}
PIPES = ["baseline", "ebd"]
PIPE_LABEL = {"baseline": "Baseline", "ebd": "EBD"}


def load_derived():
    """(rubric, criterion) -> (rater A, rater B), classified from the criteria text only (see results.md, section 7)."""
    rows = csv.DictReader(open(ROOT / "analysis" / "derived_criteria.csv", encoding="utf-8"))
    return {(r["rubric_id"], int(r["criterion_index"])): (r["rater_a"] == "DERIVED", r["rater_b"] == "DERIVED") for r in rows}


def within_corr(rows, key, value):
    """Pearson r between key and value after removing each (question, pipeline) group's mean."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["q"], r["pipeline"])].append(r)
    xs, ys = [], []
    for g in groups.values():
        mk, mv = statistics.mean(r[key] for r in g), statistics.mean(r[value] for r in g)
        xs += [r[key] - mk for r in g]
        ys += [r[value] - mv for r in g]
    return pearson(xs, ys)


def model_gap_section(md, runs, rubrics, grades, scores):
    """Section 7: how far apart the three models score under the same pipeline, and what goes with it."""
    cell = {(r["rubric_id"], r["pipeline"], r["model"]): rid for rid, r in runs.items()}
    score = lambda rid: statistics.mean(scores[(j, rid)]["total"] for j in PRIMARY)
    verdict = lambda rid, i: statistics.mean(grades[(j, rid, i)] for j in PRIMARY)
    qs = sorted(rubrics)
    best = lambda vals: max(vals.values())
    bold = lambda x, top, nd=3, flat=False: f"**{x:.{nd}f}**" if x == top and not flat else f"{x:.{nd}f}"

    md += ["## 7. Same pipeline, different model", "",
           "> Corrected on 2026-11-25. The first version of this section (commit 14dd487) showed seven criteria picked after "
           "seeing the scores, ranked the causes without measuring them, and mislabelled several run measures (page reads, "
           "researcher tokens, unreported links) and left out that DeepSeek ran at a higher reasoning effort. Those parts are replaced below. The scores and gaps are unchanged.", "",
           "Each model's score on each question with each pipeline (primary mean of three judges). **Bold**: the best "
           "model on that question and pipeline. Gap: best minus worst. The models did not run under equal conditions: DeepSeek "
           "ran at high reasoning effort (its API maps medium to high) and the other two at medium; its search had no tool cap; "
           "its researchers could write 64k tokens, against 21k for Haiku and 32k for Luna. Each model used its own provider's "
           "search. A gap is between these setups, not between models alone.", "",
           "| Question | " + " | ".join(f"{PIPE_LABEL[p]} {MODEL_SHORT.get(m, m)}" for p in PIPES for m in MODELS + ["gap"]) + " |",
           "|---|" + "---|" * (len(PIPES) * (len(MODELS) + 1))]
    gap_rows, gaps = [], {p: [] for p in PIPES}
    for q in qs:
        cells = []
        for p in PIPES:
            vals = {m: score(cell[(q, p, m)]) for m in MODELS}
            g = best(vals) - min(vals.values())
            gaps[p].append(g)
            gap_rows.append({"rubric_id": q, "pipeline": p, **vals, "gap": g, "best": max(vals, key=vals.get)})
            cells += [bold(vals[m], best(vals)) for m in MODELS] + [f"{g:.3f}"]
        md.append(f"| {q} | " + " | ".join(cells) + " |")
    cells = []
    for p in PIPES:
        vals = {m: statistics.mean(score(cell[(q, p, m)]) for q in qs) for m in MODELS}
        cells += [bold(vals[m], best(vals)) for m in MODELS] + [f"{statistics.mean(gaps[p]):.3f}"]
    md += ["| Mean | " + " | ".join(cells) + " |", "",
           "Mean row: each model's mean over the six questions; its gap column is the mean of the six gaps.", ""]

    stable = []
    for q in qs:
        leaders = {max(MODELS, key=lambda m: scores[(j, cell[(q, p, m)])]["total"]) for p in PIPES for j in PRIMARY}
        if len(leaders) == 1:
            stable.append((q, leaders.pop()))
    rest = [q for q in qs if q not in [s[0] for s in stable]]
    md += [f"The same model leads in both pipelines under every primary judge on {', '.join(f'{q} ({MODEL_SHORT[m]})' for q, m in stable)}. "
           f"On {', '.join(rest)} the leader depends on the pipeline or the judge. With one run per cell, neither a gap nor a "
           "change in a gap can be told apart from run-to-run variation.", ""]

    rows = []
    for rid, r in runs.items():
        rows.append({"q": r["rubric_id"], "pipeline": r["pipeline"], "model": r["model"], "score": score(rid),
                     "words": int(r["report_words"]), "notes": int(r["notes_words"]), "links": int(r["report_links"]),
                     "offtool": int(r["report_links_page_not_reported"]), "home": int(r["report_links_not_reported_home"]),
                     "innotes": int(r["report_links_not_reported_in_notes"]),
                     "cost": float(r["cost_usd"]), "time": int(r["duration_s"])})
    md += ["### Run profile by model (12 runs each, both pipelines)", "",
           "| Model | Mean score | Report words | Words of notes given to the writer | Links in reports | To a page no search tool reported | Of those, bare home pages | Cost (USD) | Time (min) |",
           "|---|---|---|---|---|---|---|---|---|"]
    prof = {}
    for m in MODELS:
        rs = [r for r in rows if r["model"] == m]
        a = {k: statistics.mean(r[k] for r in rs) for k in ("score", "words", "notes", "cost", "time")}
        a.update({k: sum(r[k] for r in rs) for k in ("links", "offtool", "home", "innotes")})
        prof[m] = a
        md.append(f"| {MODEL_LABEL[m]} | {a['score']:.3f} | {a['words']:,.0f} | {a['notes']:,.0f} | {a['links']:,} | "
                  f"{a['offtool']} ({a['offtool'] / a['links']:.0%}) | {a['home']} | {a['cost']:.3f} | {a['time'] / 60:.1f} |")
    pd, ph, pl = prof["deepseek"], prof["haiku"], prof["luna"]
    hl = [r for r in rows if r["model"] != "deepseek"]
    md += ["", "Links are counted at every occurrence. Pages are compared as the engine does it, without query string or "
           "fragment (`build_run_columns.py`). A link outside the tool-reported pages is not proven wrong: failed or redirected "
           f"page opens are not recorded for DeepSeek. {pd['innotes']} of its {pd['offtool']} such links already stand in the researchers' "
           "notes, where the engine marks them \"(unverified link)\" for the writer. The judges did not open links or check "
           "facts. Cost uses list prices with assumed search fees for Haiku and Luna and none for DeepSeek, so it does not "
           "compare the models' token cost.", "",
           "Correlation with the score within the same question and pipeline (group means removed, 36 reports): report words "
           f"{within_corr(rows, 'words', 'score'):+.2f}, words of notes {within_corr(rows, 'notes', 'score'):+.2f}. Among the 24 Haiku and Luna "
           f"reports alone: {within_corr(hl, 'words', 'score'):+.2f} and {within_corr(hl, 'notes', 'score'):+.2f}. The first pair therefore "
           "mostly reflects DeepSeek against the others.", ""]

    # Derived criteria: classified blind, then compared with all the others.
    derived = load_derived()
    both = {k for k, (a, b) in derived.items() if a and b}
    agree = sum(a == b for a, b in derived.values())
    kap = kappa([float(a) for a, _ in derived.values()], [float(b) for _, b in derived.values()], cats=(0.0, 1.0))

    def parts(chosen, qs=qs):
        """Mean score of each model split into derived criteria, other criteria and penalties (they add up to the score)."""
        out = {}
        for m in MODELS:
            d = o = pen = dw = ow = 0.0
            for q in qs:
                pw = sum(c["weight"] for c in rubrics[q].values() if c["weight"] > 0)
                for p in PIPES:
                    for i, c in rubrics[q].items():
                        x = c["weight"] * verdict(cell[(q, p, m)], i) / pw / (len(qs) * len(PIPES))
                        full = c["weight"] / pw / (len(qs) * len(PIPES))
                        if c["weight"] < 0:
                            pen += x
                        elif (q, i) in chosen:
                            d, dw = d + x, dw + full
                        else:
                            o, ow = o + x, ow + full
            out[m] = {"derived": d, "other": o, "penalty": pen, "dshare": d / dw, "oshare": o / ow, "dw": dw}
        return out

    pt = parts(both)
    lead = lambda t, k: t["deepseek"][k] - (t["haiku"][k] + t["luna"][k]) / 2
    md += ["### Does committing to a derived answer explain the gap?", "",
           "The idea to test: some criteria need a result the report must work out itself (an estimate or calculation for this "
           "case, a forecast, a verdict or proof), and the models may differ in how readily they commit to one. Two agents "
           "classified all 139 criteria from the questions, criteria and weights, without scores, reports or model names, "
           f"using one written rule. They agreed on {agree} of {len(derived)} (Cohen's κ {kap:.2f}); they are two instances of one model, so this shows "
           "that the rule can be applied consistently, not that it is the right rule. The rule was written after the scores "
           f"were known. {len(both)} criteria count as derived, {pt['haiku']['dw']:.0%} of the positive weight (averaged over questions). Fault "
           "(negative) criteria are separate. The comparison was fixed before the classification was read. The "
           "classification is in `derived_criteria.csv`, the rule and the borderline cases in `derived_rule.md`.", "",
           "| | Haiku | Luna | DeepSeek | DeepSeek minus the mean of the other two |", "|---|---|---|---|---|"]
    for label, k in [("Share of derived-criteria weight earned", "dshare"), ("Share of other-criteria weight earned", "oshare")]:
        md.append(f"| {label} | " + " | ".join(f"{pt[m][k]:.3f}" for m in MODELS) + f" | {lead(pt, k):+.3f} |")
    for label, k in [("Points of mean score from derived criteria", "derived"), ("Points from other criteria", "other"), ("Points lost to penalties", "penalty")]:
        md.append(f"| {label} | " + " | ".join(f"{pt[m][k]:.3f}" for m in MODELS) + f" | {lead(pt, k):+.3f} |")
    tot = lambda t: lead(t, "derived") + lead(t, "other") + lead(t, "penalty")
    loo = [parts(both, [x for x in qs if x != q]) for q in qs]
    dr, or_ = [lead(t, "dshare") for t in loo], [lead(t, "oshare") for t in loo]
    wide, narrow = both | {("Q5", 20), ("Q4", 21)}, both - {("Q5", 6)}
    alt = lambda chosen: lead(parts(chosen), "derived") / tot(pt)
    per_q = [(q, parts(both, [q])) for q in qs if any(k[0] == q for k in both)]
    more = [q for q, t in per_q if lead(t, "dshare") > lead(t, "oshare")]
    less = [q for q, _ in per_q if q not in more]
    md += ["| Mean score | " + " | ".join(f"{sum(pt[m][k] for k in ('derived', 'other', 'penalty')):.3f}" for m in MODELS) + f" | {tot(pt):+.3f} |", "",
           f"Derived criteria carry {lead(pt, 'derived') / tot(pt):.0%} of DeepSeek's lead on {pt['haiku']['dw']:.0%} of the weight; "
           f"the other criteria add {lead(pt, 'other'):+.3f} of the {tot(pt):+.3f}, and DeepSeek loses {-lead(pt, 'penalty'):.3f} more to "
           f"penalties. Leaving out one question at a time, the lead on derived criteria stays between {min(dr):+.3f} and "
           f"{max(dr):+.3f}, on other criteria between {min(or_):+.3f} and {max(or_):+.3f}. It is larger on derived than on other "
           f"criteria in {', '.join(more)}, not in {', '.join(less) or 'none'} ({len(per_q)} questions have derived criteria).", "",
           f"The share depends on the classification. With Q5 #20 (equations or proofs) and Q4 #21 (quantifiable metrics) counted "
           f"as derived, which both agents called borderline, it is {alt(wide):.0%}; without Q5 #6 (a definite yes or no), {alt(narrow):.0%}.", "",
           "All derived criteria (verdict averaged over the three primary judges; Haiku / Luna / DeepSeek):", "",
           "| Question | # | Criterion | Weight | Baseline | EBD |", "|---|---|---|---|---|---|"]
    for q, i in sorted(both):
        cells = []
        for p in PIPES:
            vals = {m: verdict(cell[(q, p, m)], i) for m in MODELS}
            cells.append(" / ".join(bold(vals[m], best(vals), 2, best(vals) == min(vals.values())) for m in MODELS))
        text = " ".join(rubrics[q][i]["criterion"].split()).replace("|", "\\|")
        md.append(f"| {q} | {i} | {text[:110] + ('…' if len(text) > 110 else '')} | {rubrics[q][i]['weight']:+g} | {cells[0]} | {cells[1]} |")
    md.append("")
    by_q = []
    for q in qs:
        idx = [i for (qq, i) in both if qq == q]
        if idx:
            w = sum(rubrics[q][i]["weight"] for i in idx)
            by_q.append(f"{q} " + " / ".join(f"{sum(rubrics[q][i]['weight'] * verdict(cell[(q, p, m)], i) for i in idx for p in PIPES) / (2 * w):.2f}" for m in MODELS) + f" (n={len(idx)})")
    md += ["Share of derived weight earned per question (Haiku / Luna / DeepSeek): " + "; ".join(by_q) + ".", ""]
    g = {(r["rubric_id"], r["pipeline"]): r["gap"] for r in gap_rows}
    ch = lambda q: f"{q} {g[(q, 'baseline')]:.3f} → {g[(q, 'ebd')]:.3f}"
    dq = {q: statistics.mean(score(cell[(q, p, "deepseek")]) - (score(cell[(q, p, "haiku")]) + score(cell[(q, p, "luna")])) / 2 for p in PIPES) for q in qs}
    dp = {p: statistics.mean(score(cell[(q, p, "deepseek")]) - (score(cell[(q, p, "haiku")]) + score(cell[(q, p, "luna")])) / 2 for q in qs) for p in PIPES}
    top = sum(r["best"] == "deepseek" for r in gap_rows)
    md += ["### What this shows, and what it does not", "",
           f"1. **DeepSeek scores higher overall, under favourable conditions.** Its mean is {tot(pt):+.3f} against the mean of the other "
           f"two ({dp['baseline']:+.3f} with the baseline, {dp['ebd']:+.3f} with EBD). It is best in {top} of 12 question and pipeline cells, and "
           "leads Q3, Q4 and Q5 in both pipelines under every primary judge. On Q1 and Q2 it is below the mean of the other two "
           f"({dq['Q1']:+.3f} and {dq['Q2']:+.3f}). Effort, search and output limit differ with the model, so this data cannot separate them.",
           f"2. **Derived answers are part of the lead; how much depends on the classification ({alt(narrow):.0%} to {alt(wide):.0%}).** DeepSeek's lead "
           f"is larger on derived criteria ({lead(pt, 'dshare'):+.3f} of their weight against {lead(pt, 'oshare'):+.3f}). It is not uniform: on the "
           "five-year revenue criterion (Q3 #1) no baseline report scores and with EBD Haiku does as well as DeepSeek; on applying "
           "the framework (Q4 #22) Luna is ahead with EBD; on Q1 #18 DeepSeek is behind with the baseline.",
           "3. **Why the models differ there was not tested.** The shared writer prompt pulls both ways. A number or finding without "
           "a source URL in the notes \"must not appear in the report\" and \"five well-sourced claims beat twenty half-sourced "
           "ones\", but the report should also \"take positions where the evidence supports them\". That some models read this "
           "more cautiously is a guess; testing it would need runs with a changed prompt.",
           f"4. **DeepSeek's reports are longer ({pd['words']:,.0f} words against {ph['words']:,.0f} and {pl['words']:,.0f}) and its writer "
           f"gets more notes ({pd['notes']:,.0f} words against {ph['notes']:,.0f} and {pl['notes']:,.0f}).** Across the models longer reports score "
           "higher, but hardly within Haiku and Luna, and Haiku writes more than Luna and scores lower.",
           f"5. **{pd['offtool'] / pd['links']:.0%} of DeepSeek's links point to a page that no search tool reported** ({pd['offtool']} of {pd['links']:,}; "
           f"Haiku {ph['offtool'] / ph['links']:.1%}, Luna {pl['offtool'] / pl['links']:.1%}). {pd['home']} of them are bare home pages, the other "
           f"{pd['offtool'] - pd['home']} are specific pages. The judges graded against the rubric without opening links, so a higher score "
           "does not show a more reliable report.",
           f"6. **The mean gap is {statistics.mean(gaps['baseline']):.3f} with the baseline and {statistics.mean(gaps['ebd']):.3f} with EBD**: "
           f"{', '.join(ch(q) for q in qs)}. With one run per cell this is a description, not evidence that EBD evens out the models.", ""]
    stats = {"profile": prof, "parts": pt, "agree": agree, "kappa": kap, "n_both": len(both), "more": more, "less": less,
             "loo_derived": [min(dr), max(dr)], "loo_other": [min(or_), max(or_)],
             "derived_both": sorted([list(k) for k in both]), "mean_gap": {p: statistics.mean(gaps[p]) for p in PIPES},
             "lead": {k: lead(pt, k) for k in ("dshare", "oshare", "derived", "other", "penalty")}, "lead_total": tot(pt),
             "share": lead(pt, "derived") / tot(pt), "share_narrow": alt(narrow), "share_wide": alt(wide), "lead_by_question": dq,
             "lead_by_pipeline": dp, "best_cells": top, "r_words": within_corr(rows, "words", "score"), "r_words_hl": within_corr(hl, "words", "score")}
    return gap_rows, stats


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    runs, rubrics, grades = load()
    scores = report_scores(runs, rubrics, grades)

    # Per report, per judge.
    with open(OUT / "report_scores.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["report_id", "pair_id", "rubric_id", "model", "pipeline", "judge", "total"] + AXES)
        for rid, run in sorted(runs.items(), key=lambda kv: (kv[1]["pair_id"], kv[1]["pipeline"])):
            for j in JUDGES:
                s = scores[(j, rid)]
                w.writerow([rid, run["pair_id"], run["rubric_id"], run["model"], run["pipeline"], j, round(s["total"], 4)]
                           + ["" if s[a] is None else round(s[a], 4) for a in AXES])

    # Pairs: EBD and baseline of the same question and model.
    pairs = defaultdict(dict)
    for rid, run in runs.items():
        pairs[(run["rubric_id"], run["model"])][run["pipeline"]] = rid
    pair_rows = []
    for (q, m), d in sorted(pairs.items(), key=lambda kv: (kv[0][0], MODELS.index(kv[0][1]))):
        e, b = d["ebd"], d["baseline"]
        row = {"rubric_id": q, "model": m, "pair_id": runs[e]["pair_id"]}
        for j in JUDGES:
            row[f"ebd_{j}"] = scores[(j, e)]["total"]
            row[f"base_{j}"] = scores[(j, b)]["total"]
            row[f"diff_{j}"] = row[f"ebd_{j}"] - row[f"base_{j}"]
        row["ebd_mean"] = statistics.mean(row[f"ebd_{j}"] for j in PRIMARY)
        row["base_mean"] = statistics.mean(row[f"base_{j}"] for j in PRIMARY)
        row["diff_mean"] = row["ebd_mean"] - row["base_mean"]
        row["judges_ebd_higher"] = sum(row[f"diff_{j}"] > 0 for j in PRIMARY)
        row["ebd_mean4"] = statistics.mean(row[f"ebd_{j}"] for j in JUDGES)
        row["base_mean4"] = statistics.mean(row[f"base_{j}"] for j in JUDGES)
        for a in AXES:
            ds = [scores[(j, e)][a] - scores[(j, b)][a] for j in PRIMARY if scores[(j, e)][a] is not None]
            row[f"axis_{a}"] = statistics.mean(ds) if ds else None
        pair_rows.append(row)
    with open(OUT / "pair_matrix.csv", "w", encoding="utf-8", newline="") as f:
        cols = list(pair_rows[0])
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in pair_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    md = ["# Results", "",
          "Generated by `analysis/analyze.py` from the four judges' grades. Scores use the ResearchRubrics formula "
          "(sum of weight × score over the positive weights; 1 / 0.5 / 0 per verdict). A difference is EBD minus baseline "
          "on the same question and model. The primary result averages three judges, one per model family: "
          "Fable 5.2 (Anthropic), Astra (OpenAI) and Neon (Google). Opus 5.5 also graded every report; it shares "
          "Fable's family and agrees with it most, so it is left out of the mean and shown as a robustness check.", "",
          "Figures (`python analysis/plots.py`): "
          "[matrix](figures/1_matrix_ebd_minus_baseline.png) · [per judge](figures/2_per_judge_mean_diff.png) · "
          "[rubric axes](figures/3_axis_breakdown.png) · [judge agreement](figures/4_judge_agreement.png)", "",
          "Read with these limits:", "",
          "- One run per question, model and pipeline: 12 pairs in the main comparison (Haiku 5.5, GPT-6 Luna) and 6 "
          "for DeepSeek, which ran under different conditions (reasoning effort high against medium, built-in deep search without a tool cap, "
          "a larger researcher output limit, round cap 3 instead of 8 that no run reached) "
          "and is analysed separately.",
          "- Both pipelines had the same ceiling of 6 researchers, not the same spend. EBD used more researchers, "
          "rounds, money and time (section 6), so a difference includes what that extra work bought.",
          "- The judges graded correctness only against the rubric and the report itself, not their own knowledge. "
          "Neon's subagents could not confirm their model.",
          "- The References axis has two criteria in one rubric (Q3); its large difference rests on one question.",
          "- Code versions: Luna runs 4624de8, Haiku f1cc9de, DeepSeek 92f037b (see `runs.csv`). The fixes between "
          "them only act on failures the Luna runs did not hit.", ""]

    # 1. Main matrix.
    md += ["## 1. Matrix: EBD − baseline by question and model (primary mean of three judges)", "",
           "| Question | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    by = {(r["rubric_id"], r["model"]): r for r in pair_rows}
    for q in sorted({r["rubric_id"] for r in pair_rows}):
        cells = []
        for m in MODELS:
            r = by[(q, m)]
            cells.append(f"{r['diff_mean']:+.3f} ({r['judges_ebd_higher']}/{len(PRIMARY)})")
        md.append(f"| {q} | " + " | ".join(cells) + " |")
    md += ["", "In parentheses: how many of the three primary judges scored EBD higher.", ""]

    # 2. Summary per group, judge mean and per judge.
    md += ["## 2. Summary per comparison", "",
           "| Comparison | Judge | Pairs | EBD mean | Baseline mean | Mean diff | 95% bootstrap CI | EBD higher / lower | Sign test p |",
           "|---|---|---|---|---|---|---|---|---|"]
    summary_rows = []
    for g, ms in GROUPS.items():
        rows = [r for r in pair_rows if r["model"] in ms]
        for j in ["mean"] + JUDGES + ["mean4"]:
            e = [r[{"mean": "ebd_mean", "mean4": "ebd_mean4"}.get(j, f"ebd_{j}")] for r in rows]
            b = [r[{"mean": "base_mean", "mean4": "base_mean4"}.get(j, f"base_{j}")] for r in rows]
            d = [x - y for x, y in zip(e, b)]
            hi, lo = sum(x > 0 for x in d), sum(x < 0 for x in d)
            ci = bootstrap_ci(d)
            p = sign_test(hi, lo)
            label = {"mean": "Primary mean (3 judges)", "mean4": "All 4 judges (robustness)"}.get(j) or (
                JUDGE_LABEL[j] + ("" if j in PRIMARY else " (robustness)"))
            summary_rows.append([g, label, len(rows), statistics.mean(e), statistics.mean(b), statistics.mean(d), ci, hi, lo, p])
            md.append(f"| {g} | {label} | {len(rows)} | {statistics.mean(e):.3f} | {statistics.mean(b):.3f} | "
                      f"{statistics.mean(d):+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {hi} / {lo} | {p:.3f} |")
    md.append("")
    for m in MODELS:
        rows = [r for r in pair_rows if r["model"] == m]
        d = [r["diff_mean"] for r in rows]
        md.append(f"- {MODEL_LABEL[m]}: mean diff {statistics.mean(d):+.3f}, EBD higher on {sum(x > 0 for x in d)} of {len(d)} questions.")
    md.append("")

    # 3. Axis breakdown.
    md += ["## 3. EBD − baseline by rubric axis (primary mean, averaged over pairs where the axis exists)", "",
           "| Axis | Main (Haiku + Luna) | Pairs | DeepSeek | Pairs |", "|---|---|---|---|---|"]
    axis_rows = []
    for a in AXES:
        cells = []
        for ms in GROUPS.values():
            vals = [r[f"axis_{a}"] for r in pair_rows if r["model"] in ms and r[f"axis_{a}"] is not None]
            cells += [f"{statistics.mean(vals):+.3f}" if vals else "–", str(len(vals))]
            axis_rows.append((a, list(GROUPS).index([k for k, v in GROUPS.items() if v == ms][0]), statistics.mean(vals) if vals else None, len(vals)))
        md.append(f"| {a} | " + " | ".join(cells) + " |")
    md.append("")

    # 4. Judge reliability.
    keys = sorted({(rid, i) for (_, rid, i) in grades})
    md += ["## 4. Agreement between judges", "",
           f"Over {len(keys)} gradings (36 reports × their rubric's criteria).", "",
           "| Judge pair | Exact agreement | Cohen's κ | Linear-weighted κ | Report-score correlation (r) |",
           "|---|---|---|---|---|"]
    agree_rows = []
    for a, b in combinations(JUDGES, 2):
        va = [grades[(a, rid, i)] for rid, i in keys]
        vb = [grades[(b, rid, i)] for rid, i in keys]
        exact = sum(x == y for x, y in zip(va, vb)) / len(keys)
        k0, k1 = kappa(va, vb), kappa(va, vb, weights="linear")
        r = pearson([scores[(a, rid)]["total"] for rid in runs], [scores[(b, rid)]["total"] for rid in runs])
        agree_rows.append((a, b, exact, k0, k1, r))
        md.append(f"| {JUDGE_LABEL[a]} – {JUDGE_LABEL[b]} | {exact:.0%} | {k0:.2f} | {k1:.2f} | {r:.2f} |")
    alpha = krippendorff_interval([[grades[(j, rid, i)] for j in PRIMARY] for rid, i in keys])
    alpha4 = krippendorff_interval([[grades[(j, rid, i)] for j in JUDGES] for rid, i in keys])
    md += ["", f"Krippendorff's α (interval): **{alpha:.2f}** across the three primary judges, {alpha4:.2f} across all four.", ""]
    with open(OUT / "judge_agreement.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["judge_a", "judge_b", "exact_agreement", "cohen_kappa", "linear_weighted_kappa", "report_score_r"])
        for row in agree_rows:
            w.writerow([row[0], row[1]] + [round(x, 4) for x in row[2:]])

    # 5. Checks for judge bias.
    md += ["## 5. Bias checks", "", "| Judge | Mean score, position 1 | Mean score, position 2 | Corr. score–words |", "|---|---|---|---|"]
    for j in JUDGES:
        p1 = [scores[(j, rid)]["total"] for rid, r in runs.items() if r["position"] == "1"]
        p2 = [scores[(j, rid)]["total"] for rid, r in runs.items() if r["position"] == "2"]
        rw = pearson([scores[(j, rid)]["total"] for rid in runs], [int(runs[rid]["report_words"]) for rid in runs])
        md.append(f"| {JUDGE_LABEL[j]} | {statistics.mean(p1):.3f} | {statistics.mean(p2):.3f} | {rw:+.2f} |")
    md += ["", "Position is the report's place in its pair (report_1 or report_2), assigned at random.", ""]
    ebd_first = sum(1 for r in runs.values() if r["pipeline"] == "ebd" and r["position"] == "1")
    md += [f"EBD was report_1 in {ebd_first} of {len(pair_rows)} pairs.", ""]
    md += ["Length within pairs: does the score difference follow the word difference?", "",
           "| Comparison | Pairs | Mean word diff (EBD − baseline) | r(score diff, word diff) | Mean diff, pairs where EBD is shorter |",
           "|---|---|---|---|---|"]
    for g, ms in GROUPS.items():
        rows = [r for r in pair_rows if r["model"] in ms]
        wd, sd = [], []
        for r in rows:
            e = next(x for x in runs.values() if x["pair_id"] == r["pair_id"] and x["pipeline"] == "ebd")
            b = next(x for x in runs.values() if x["pair_id"] == r["pair_id"] and x["pipeline"] == "baseline")
            wd.append(int(e["report_words"]) - int(b["report_words"]))
            sd.append(r["diff_mean"])
            r["word_diff"] = wd[-1]
        shorter = [s for s, w in zip(sd, wd) if w < 0]
        md.append(f"| {g} | {len(rows)} | {statistics.mean(wd):+.0f} | {pearson(sd, wd):+.2f} | "
                  + (f"{statistics.mean(shorter):+.3f} ({len(shorter)} pairs)" if shorter else "–") + " |")
    md.append("")

    # 6. Cost and effort from the run records.
    md += ["## 6. Cost and effort per run (from the run records)", "",
           "| Model | Pipeline | Researchers | Failed researchers | Rounds | Searches | Page reads | Report words | Cost (USD) | Time (s) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for p in ("baseline", "ebd"):
            rs = [r for r in runs.values() if r["model"] == m and r["pipeline"] == p]
            avg = lambda k: statistics.mean(float(r[k]) for r in rs)
            md.append(f"| {MODEL_LABEL[m]} | {p} | {avg('researchers_used'):.1f} | {sum(int(r['researchers_failed']) for r in rs)} | "
                      f"{avg('rounds'):.1f} | {avg('searches'):.1f} | {avg('page_reads'):.1f} | {avg('report_words'):.0f} | "
                      f"{avg('cost_usd'):.3f} | {avg('duration_s'):.0f} |")
    md += ["", "Searches and page reads compare the two pipelines on one model, not the models with each other: each provider "
           "counts its own tool actions. Anthropic counts fetch requests; OpenAI and DeepSeek also count find-in-page actions on "
           "an open page; one OpenAI or DeepSeek search can hold several queries. The caps differ too: DeepSeek's search had none.", ""]

    gap_rows, gap_stats = model_gap_section(md, runs, rubrics, grades, scores)

    (ROOT / "analysis" / "results.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    json.dump({"pair_rows": pair_rows, "axis_rows": axis_rows, "agree_rows": agree_rows, "alpha": alpha, "alpha4": alpha4,
               "gap_rows": gap_rows, "gap_stats": gap_stats, "summary_rows": [r[:6] + [list(r[6])] + r[7:] for r in summary_rows]},
              open(OUT / "summary.json", "w", encoding="utf-8"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
