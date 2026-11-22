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
# Criteria whose answer has to be derived (an estimate, a forecast, a verdict) rather than found in a source.
COMMIT = [("Q5", 6, "Gives a definitive yes or no"), ("Q5", 20, "Includes equations or proofs"),
          ("Q3", 3, "Shows at least 4 intermediate calculations"), ("Q3", 8, "Estimates the cost of opening the branch"),
          ("Q4", 21, "Framework has at least 2 quantifiable metrics"), ("Q4", 22, "Applies the framework to several current states"),
          ("Q6", 8, "Gives a budget or cost analysis")]


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
    score = lambda rid, judges=PRIMARY: statistics.mean(scores[(j, rid)]["total"] for j in judges)
    verdict = lambda rid, i: statistics.mean(grades[(j, rid, i)] for j in PRIMARY)
    qs = sorted(rubrics)
    best = lambda vals: max(vals.values())
    bold = lambda x, top, nd=3: f"**{x:.{nd}f}**" if x == top else f"{x:.{nd}f}"

    md += ["## 7. Same pipeline, different model", "",
           "Each model's score on each question with each pipeline (primary mean of three judges). **Bold**: the best "
           "model on that question and pipeline. Gap: best minus worst. DeepSeek ran under its own conditions (see the "
           "limits at the top), so here \"DeepSeek\" means the model together with its deep search and round cap.", "",
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
    md += [f"| Mean | " + " | ".join(cells) + " |", "",
           "Mean row: each model's mean over the six questions; its gap column is the mean of the six gaps.", ""]

    # Which gaps hold up: same leader in both pipelines (two separate runs) under every primary judge.
    stable, same_order = [], []
    for q in qs:
        leaders = set()
        for p in PIPES:
            for j in PRIMARY:
                vals = {m: scores[(j, cell[(q, p, m)])]["total"] for m in MODELS}
                leaders.add(max(vals, key=vals.get))
        if len(leaders) == 1:
            stable.append(f"{q} ({MODEL_SHORT[leaders.pop()]})")
        orders = {tuple(sorted(MODELS, key=lambda m: score(cell[(q, p, m)]))) for p in PIPES}
        if len(orders) == 1:
            same_order.append(q)
    unstable = [q for q in qs if not any(s.startswith(q) for s in stable)]
    md += [f"The same model leads in both pipelines under every primary judge on {', '.join(stable) or 'no question'}. "
           f"The full order of the three models is the same in both pipelines on {', '.join(same_order) or 'no question'}. "
           f"On {', '.join(unstable)} the leader changes between pipelines or judges, so most of those gaps are run-to-run "
           "variation; with one run per cell the data cannot separate them further.", ""]

    # Run profile by model.
    rows = []
    for rid, r in runs.items():
        rows.append({"q": r["rubric_id"], "pipeline": r["pipeline"], "model": r["model"], "score": score(rid),
                     "words": int(r["report_words"]), "pages": int(r["page_reads"]), "searches": int(r["searches"]),
                     "notes": int(r["researcher_output_tokens"]), "writer_in": int(r["writer_input_tokens"]),
                     "urls": int(r["report_urls"]), "unreported": int(r["report_urls_unreported"]),
                     "cost": float(r["cost_usd"]), "time": int(r["duration_s"])})
    md += ["### Run profile by model (mean of 12 runs, both pipelines)", "",
           "| Model | Mean score | Report words | Pages opened | Searches | Researcher notes (tokens) | Writer input (tokens) | Report URLs (unreported by tools) | Cost (USD) | Time (min) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    prof = {}
    for m in MODELS:
        rs = [r for r in rows if r["model"] == m]
        a = {k: statistics.mean(r[k] for r in rs) for k in ("score", "words", "pages", "searches", "notes", "writer_in", "cost", "time")}
        prof[m] = a
        md.append(f"| {MODEL_LABEL[m]} | {a['score']:.3f} | {a['words']:,.0f} | {a['pages']:.0f} | {a['searches']:.0f} | "
                  f"{a['notes'] / 1000:.1f}k | {a['writer_in'] / 1000:.1f}k | {sum(r['urls'] for r in rs)} "
                  f"({sum(r['unreported'] for r in rs)}) | {a['cost']:.3f} | {a['time'] / 60:.1f} |")
    md += ["", "Pages opened: pages the researchers' search tools read in full (DeepSeek's deep search opens pages by itself). "
           "Unreported URLs: links in the report that the search tools never returned.", "",
           "Correlation with the score within the same question and pipeline (each group's mean removed, 36 reports):", ""]
    md += ["| Measure | r |", "|---|---|"] + [f"| {label} | {within_corr(rows, k, 'score'):+.2f} |" for k, label in
                                              [("words", "Report words"), ("pages", "Pages opened"),
                                               ("notes", "Researcher notes (tokens)"), ("writer_in", "Writer input (tokens)")]]
    md += ["", "These measures move together with the model (DeepSeek is high on all of them), so a correlation here does "
           "not separate the measure from the model.", ""]

    # Criteria that need a derived answer.
    md += ["### Criteria that need a derived answer", "",
           "Verdicts averaged over the three primary judges (1 satisfied, 0.5 partially, 0 not), in the order Haiku / Luna / "
           "DeepSeek. **Bold**: the highest in that pipeline.", "",
           "| Question | # | Criterion | Weight | Baseline | EBD |", "|---|---|---|---|---|---|"]
    for q, i, label in COMMIT:
        cells = []
        for p in PIPES:
            vals = {m: verdict(cell[(q, p, m)], i) for m in MODELS}
            cells.append(" / ".join(bold(vals[m], best(vals), 2) for m in MODELS))
        md.append(f"| {q} | {i} | {label} | {rubrics[q][i]['weight']:+g} | {cells[0]} | {cells[1]} |")
    md.append("")

    # Reading.
    g = {(r["rubric_id"], r["pipeline"]): r["gap"] for r in gap_rows}
    w = {(r["q"], r["pipeline"], r["model"]): r for r in rows}
    q6d, q6l = w[("Q6", "baseline", "deepseek")], w[("Q6", "baseline", "luna")]
    change = lambda q: f"{q} {g[(q, 'baseline')]:.3f} → {g[(q, 'ebd')]:.3f}"
    md += ["### Why the gaps appear", "",
           "Interpretation of the tables above; the reports quoted are in `reports/`.", "",
           "1. **Committing to an answer no source states.** Q3 asks for revenue estimates, Q4 for a forecast of future "
           "revolutions, Q5 to \"prove definitely\". The answer has to be derived, and the criteria in the table above carry "
           "heavy weight for that step. The shared writer prompt (`src/writer/writer.ts` in the engine) says a number "
           "without a source URL \"must not appear in the report\", that \"five well-sourced claims beat twenty half-sourced "
           "ones\", and asks the writer to be \"epistemically honest\". Haiku and Luna mostly hold back under these rules "
           "(Luna on estimates and forecasts, Haiku on the Q5 verdict); DeepSeek derives the figure and shows how. "
           "Haiku's Q5 baseline report (R967ed13f) says no formal result establishes the guarantee and none proves the "
           "opposite \"in full generality\"; DeepSeek's "
           "(R4141c12b) opens with \"Verdict: no unconditional mathematical guarantee exists\". Luna's Q4 baseline report "
           "(R6b7b9320) offers \"conditional scenarios, not forecasts about a specific country or date\"; DeepSeek's "
           "(R0b63ad33) applies its framework to named countries with probabilities. Luna's Q6 baseline report (Rc4b44883) "
           "says no cost estimate \"follows from a daily-user target alone\". Exceptions: Haiku does show its Q3 "
           "calculations, and with EBD Luna applies the Q4 framework more fully than DeepSeek. The judges grade against "
           "the rubric and do not check facts, so a committed estimate earns its points whether or not it is right.",
           f"2. **Length and coverage.** Most criteria ask whether the report includes something. DeepSeek writes "
           f"{prof['deepseek']['words']:,.0f} words on average, Haiku {prof['haiku']['words']:,.0f}, Luna "
           f"{prof['luna']['words']:,.0f}, and within a question and pipeline longer reports score higher (table above). "
           f"Q6, whose rubric has {len(rubrics['Q6'])} criteria, is the clearest case: with the baseline DeepSeek's "
           f"{q6d['words']:,}-word report scores {q6d['score']:.3f} and Luna's {q6l['words']:,}-word report {q6l['score']:.3f}.",
           f"3. **Each model brings its own search tool.** The pipeline fixes the steps, not the search: every researcher "
           f"uses its provider's built-in web search. DeepSeek opens about {prof['deepseek']['pages']:.0f} pages per run, "
           f"Luna {prof['luna']['pages']:.0f} and Haiku {prof['haiku']['pages']:.0f}; Luna returns the fewest notes, so its "
           "writer gets about half the material of the other two and writes the shortest reports. Evidence is not the "
           "whole story: in the engine's run records, Haiku's Q5 notes cover inverse reinforcement learning in both runs and "
           "Rice's theorem in the EBD run (criteria 9 and 1), and neither Haiku report mentions either.",
           f"4. **EBD narrows coverage gaps, not commitment gaps.** The mean gap is {statistics.mean(gaps['baseline']):.3f} "
           f"with the baseline and {statistics.mean(gaps['ebd']):.3f} with EBD. It shrinks on {change('Q1')}, "
           f"{change('Q4')} and {change('Q6')}, where EBD's suggested outline of the question's parts likely makes every "
           f"model cover them, and barely moves on {change('Q3')} and {change('Q5')}: an outline lists the parts but does "
           "not make a model commit to a number or a verdict.", ""]
    return gap_rows


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
          "for DeepSeek, which ran under different conditions (built-in deep search without a tool cap, round cap 3) "
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
    md.append("")

    gap_rows = model_gap_section(md, runs, rubrics, grades, scores)

    (ROOT / "analysis" / "results.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    json.dump({"pair_rows": pair_rows, "axis_rows": axis_rows, "agree_rows": agree_rows, "alpha": alpha, "alpha4": alpha4,
               "gap_rows": gap_rows, "summary_rows": [r[:6] + [list(r[6])] + r[7:] for r in summary_rows]},
              open(OUT / "summary.json", "w", encoding="utf-8"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
