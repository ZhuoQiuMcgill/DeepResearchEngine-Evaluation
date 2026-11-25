"""Fill the notes and link columns of analysis/runs.csv from the engine's run records.

Usage (from the repository root): python analysis/build_run_columns.py <engine>/local/experiments

The run records are not in this repository, so this needs the machine that ran the experiments.
notes_words                 words of researcher notes handed to the report writer
report_links                Markdown links in the report (every occurrence)
report_links_page_not_reported  links to a page that no search tool of the run reported. Pages are compared
                            as the engine does: scheme, "www.", query string and fragment do not count
report_links_not_reported_home  of those, links to a site's bare home page
report_links_not_reported_in_notes  of those, links that already stand in the researchers' notes
"""
import csv
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\((https?://[^\s)]+(?:\([^\s)]*\)[^\s)]*)*)\)")
NOTE_URL = re.compile(r"https?://[^\s)\]>\"]+")
NEW = ["notes_words", "report_links", "report_links_page_not_reported", "report_links_not_reported_home",
       "report_links_not_reported_in_notes"]
OLD = ["researcher_output_tokens", "writer_input_tokens", "report_urls", "report_urls_unreported",
       "report_links_not_in_tool_sources", "report_links_home_page"]


def page(url):
    s = urlsplit(unquote(url.strip()))
    host = s.netloc.lower()
    return (host[4:] if host.startswith("www.") else host) + s.path.rstrip("/")


def main():
    engine = Path(sys.argv[1])
    path = ROOT / "analysis" / "runs.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = [c for c in rows[0] if c not in NEW + OLD]
    for r in rows:
        d = engine / r["experiment"] / "runs" / r["run_id"]
        material = json.load(open(d / "synthesis.json", encoding="utf-8"))["synthesis"]["material"]
        results = [json.load(open(f, encoding="utf-8")) for f in sorted((d / "results").glob("*.json"))]
        reported = {page(s["url"]) for x in results for s in x.get("sources", []) if s.get("url")}
        noted = {page(u) for x in results for u in NOTE_URL.findall(x.get("notes") or "")}
        links = [page(u) for u in LINK.findall((ROOT / "reports" / f"{r['report_id']}.md").read_text(encoding="utf-8"))]
        off = [k for k in links if k not in reported]
        r["notes_words"] = sum(len(m["notes"].split()) for m in material)
        r["report_links"] = len(links)
        r["report_links_page_not_reported"] = len(off)
        r["report_links_not_reported_home"] = sum("/" not in k for k in off)
        r["report_links_not_reported_in_notes"] = sum(k in noted for k in off)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + NEW, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
