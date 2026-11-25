"""Build the data files of the results page (index.html) from this branch. Standard library only.

Usage (from the repository root), after python analysis/analyze.py:

    python analysis/explorer/build_data.py
    python -m http.server 8000 --bind 127.0.0.1 --directory analysis/explorer   # then open /preview.html

Writes meta.json and grades.json, which index.html loads, and preview.html, a copy of index.html with the
document head that the Artifact host adds when the page is published. All three are generated and not committed.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
JUDGES = [("claude-code", "Opus 5.5", "Claude Code"), ("claude-code-fable", "Fable 5.2", "Claude Code"),
          ("codex", "Astra", "Codex"), ("antigravity", "Neon", "Antigravity")]


def load(path):
    return json.load(open(path, encoding="utf-8"))


def main():
    questions = []
    for q in load(HERE / "questions.json"):
        crit = load(ROOT / "rubrics" / f"{q['id']}.json")["criteria"]
        questions.append({**q, "criteria": crit, "positive_weight": sum(c["weight"] for c in crit if c["weight"] > 0)})

    scores = {}
    for r in csv.DictReader(open(ROOT / "analysis" / "out" / "report_scores.csv", encoding="utf-8")):
        scores.setdefault(r["report_id"], {})[r["judge"]] = float(r["total"])
    reports = []
    for r in csv.DictReader(open(ROOT / "analysis" / "runs.csv", encoding="utf-8")):
        reports.append({"id": r["report_id"], "pair": r["pair_id"], "q": r["rubric_id"], "model": r["model"],
                        "pipeline": r["pipeline"], "words": int(r["report_words"]), "researchers": int(r["researchers_used"]),
                        "rounds": int(r["rounds"]), "cost": float(r["cost_usd"]), "seconds": int(r["duration_s"]),
                        "notes": int(r["notes_words"]), "links": int(r["report_links"]),
                        "offtool": int(r["report_links_page_not_reported"]), "home": int(r["report_links_not_reported_home"]),
                        "scores": scores[r["report_id"]]})

    meta = {"judges": [{"id": j, "label": l, "tool": t} for j, l, t in JUDGES], "questions": questions, "reports": reports,
            "gapStats": load(ROOT / "analysis" / "out" / "summary.json")["gap_stats"],
            "repo": "https://github.com/ZhuoQiuMcgill/DeepResearchEngine-Evaluation/blob/analysis/reports/"}
    dump = lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    (HERE / "meta.json").write_text(dump(meta), encoding="utf-8", newline="\n")

    grades = []
    for j, _, _ in JUDGES:
        for f in sorted((ROOT / "judgments" / j).glob("P*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    g = json.loads(line)
                    grades.append({"j": j, "r": g["report_id"], "c": g["criterion_index"], "s": g["score"], "v": g["verdict"],
                                   "why": g["reasoning"], "quotes": g["evidence_quotes"][:3], "missing": g["missing_elements"][:4]})
    (HERE / "grades.json").write_text("[\n" + ",\n".join(dump(g) for g in grades) + "\n]\n", encoding="utf-8", newline="\n")

    head = '<!doctype html>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    (HERE / "preview.html").write_text(head + (HERE / "index.html").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    print(f"{len(questions)} questions, {len(reports)} reports, {len(grades)} grades")


if __name__ == "__main__":
    main()
