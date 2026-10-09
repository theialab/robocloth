#!/usr/bin/env python
"""Results section of the project page (index.html, <section id="main-viz">), built from the paper.

Every number in that section comes from the paper's tables: tables/ubo.tex (the UBO2014 transfer
results shown in the chart), tables/ours.tex (RoboCloth + Bonn test sets, parsed and verified but
not shown) and tables/related.tex (the comparison with existing material datasets). They are parsed
here and never retyped: the chart, its alt text, the related-datasets table and every
<span data-res="..."> in the section's prose are generated from the parsed values.

  export PYTHONNOUSERSITE=1; PY=/data/colin/envs/teaser/bin/python
  # re-parse the paper's tables -> results_data.json, then rebuild chart + HTML:
  $PY docs/tools/results/build_results.py --tex-dir <paper checkout>/tables
  # rebuild from the committed snapshot (no paper checkout needed):
  $PY docs/tools/results/build_results.py
  # verify only (page, chart and snapshot vs the data, and vs the .tex with --tex-dir):
  $PY docs/tools/results/build_results.py --check [--tex-dir <paper checkout>/tables]

Writes
  docs/tools/results/results_data.json            parsed tables + sha256 of each source .tex
  docs/images/results/psnr_averages.svg           UBO2014 chart, wide layout   + _2x.png
  docs/images/results/psnr_averages_narrow.svg    UBO2014 chart, phone layout  + _2x.png
  docs/index.html                                 the blocks between "<!-- BEGIN generated:results-* -->"
                                                  and "<!-- END generated:results-* -->", and the
                                                  text of every <span data-res="..."> in the section
CPU only. The PNGs are rendered by headless Chrome (playwright), exactly as the page shows the SVG.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, "..", ".."))
INDEX = os.path.join(DOCS, "index.html")
DATA_JSON = os.path.join(HERE, "results_data.json")
IMG_DIR = os.path.join(DOCS, "images", "results")
IMG_URL = "images/results"
ASSET_V = 2                       # ?v= cache-bust for the generated images (bump on change)
CHROME = "/usr/bin/google-chrome"

# --------------------------------------------------------------------------- display
# (names and wording only -- every number comes from the parsed tables)
SET_ORDER = ("robocloth", "ubo2014", "bonn")
SETS = {
    "robocloth": dict(title="RoboCloth test set", source="ours.tex"),
    "ubo2014": dict(title="UBO2014", source="ubo.tex"),
    "bonn": dict(title="Bonn test set", source="ours.tex"),
}
OURS = "RoboCloth"
OURS_YEAR = "2026"                 # release year; the paper's table leaves our own row blank
CHART_SET = "ubo2014"             # the only test set on the page
# second line under each method name in the chart
METHOD_SUB = {"RoboCloth": "ours", "Bonn": "UBOFAB19", "MERL": "", "PBR": "analytic fit"}
METHOD_LABEL = {"Bonn": "UBOFAB19"}   # shown name (the paper's column is "Bonn"; the page names the dataset)

# related-datasets table (tables/related.tex): header text in the .tex -> short column label
RELATED_COLS = {
    "Publication Year": "Year",
    "Dataset Scale": "Materials",
    "Cloth Specific": "Cloth-specific",
    "Real World": "Real-world",
    "Spatially Varying": "Spatially varying",
    "Multiple Views": "Multiple views",
    "Flexible Coverage": "Flexible coverage",
}
RELATED_NUMERIC = ("Publication Year", "Dataset Scale")   # the rest are check / cross columns
RELATED_BASELINES = ("MERL", "UBO2014", "UBOFAB19")       # highlighted + tagged "main baseline"
RELATED_TAGS = {OURS: "ours", "MERL": "main baseline", "UBO2014": "main baseline", "UBOFAB19": "main baseline"}

# page palette (css/style.css :root) + chart neutrals
INK = "#2b2b2b"          # --brand
INK2 = "#4a515b"
MUTED = "#6e7884"        # --muted
GRID = "#e3e6eb"
BASELINE = "#c5cad3"
ACCENT = "#3367d6"       # --accent: the RoboCloth bar (the author prefers the blue)
OTHER = "#b4bbc5"        # every other bar (every bar is also direct-labelled)
FONT = "Inter, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
AXIS = (25.0, 40.0, 5.0)  # dB: floor, top, tick step -- the floor is labelled in the chart and the caption
CHART_NARROW_MAX = 820    # px viewport: the phone layout up to here (keep in sync with css/results.css)


# =========================================================================== parsing
def _strip_comments(text: str) -> str:
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())


def _clean(s: str) -> str:
    s = s.replace(r"\sysname{}", "RoboCloth").replace(r"\sysname", "RoboCloth")
    s = re.sub(r"~?\\cite\{[^}]*\}", "", s)
    s = s.replace(r"\ ", " ").replace("~", " ")
    s = re.sub(r"\\(textbf|textit|emph|underline)\{(.*?)\}", r"\2", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _cell(cell: str):
    """'\\textbf{29.24}' -> ('29.24', 'best'); '\\underline{x}' -> 'second'; plain -> ''."""
    cell = cell.strip()
    for macro, mark in (("textbf", "best"), ("underline", "second")):
        m = re.fullmatch(r"\\%s\{(.*)\}" % macro, cell)
        if m:
            return m.group(1).strip(), mark
    return cell, ""


def _tabular_rows(path: str):
    """The rows (lists of raw cells) of the first tabular in a .tex file, rules stripped."""
    text = _strip_comments(open(path, encoding="utf-8").read())
    text = re.sub(r"\\makecell\{(.*?)\}", lambda m: m.group(1).replace(r"\\", " "), text, flags=re.S)
    body = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", text, re.S).group(1)
    rows = []
    for raw in body.split(r"\\"):
        r = re.sub(r"\\(toprule|midrule|bottomrule)", "", raw)
        r = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", r).strip()
        if r:
            rows.append(r)
    return rows


def parse_table(path: str):
    """ours.tex / ubo.tex -> (columns, [ {title, rows:[{material, cells, marks, delta}], average} ])."""
    columns, blocks, cur = None, [], None
    for r in _tabular_rows(path):
        m = re.fullmatch(r"\\multicolumn\{\d+\}\{l\}\{\\textit\{(.*)\}\}", r, re.S)
        if m:                                   # a block heading ("In-domain: ...")
            cur = {"title": _clean(m.group(1)), "rows": [], "average": None}
            blocks.append(cur)
            continue
        cells = [c.strip() for c in r.split("&")]
        if cells[0] == "Material":
            columns = [_clean(c) for c in cells[1:-1]]
            continue
        if columns is None:                     # the "Decoder training source" header row
            continue
        if len(cells) != len(columns) + 2:
            raise ValueError(f"{path}: unexpected row {r!r}")
        if cur is None:
            cur = {"title": None, "rows": [], "average": None}
            blocks.append(cur)
        name = _clean(cells[0])
        vals = [_cell(c) for c in cells[1:-1]]
        for v, _ in vals:
            float(v)                            # must be a number
        delta = re.sub(r"[{}$\s]", "", cells[-1])
        float(delta)
        row = {"material": name, "cells": [v for v, _ in vals], "marks": [k for _, k in vals],
               "delta": delta}
        if name == "Average":
            cur["average"] = row
        else:
            cur["rows"].append(row)
    if not columns or not blocks:
        raise ValueError(f"{path}: no tabular data found")
    return columns, blocks


def parse_related(path: str) -> dict:
    """related.tex -> {columns, rows:[{name, cite, year, scale, flags:[bool...]}]}.

    columns are the .tex header texts (keys of RELATED_COLS); flags follow the check/cross columns."""
    columns, rows = None, []
    for r in _tabular_rows(path):
        cells = [c.strip() for c in r.split("&")]
        if cells[0] == "Dataset":
            columns = [_clean(c) for c in cells[1:]]
            unknown = [c for c in columns if c not in RELATED_COLS]
            if unknown or columns[:2] != list(RELATED_NUMERIC):
                raise ValueError(f"{path}: unexpected header {columns}")
            continue
        if columns is None:
            continue
        if len(cells) != len(columns) + 1:
            raise ValueError(f"{path}: unexpected row {r!r}")
        m = re.search(r"\\cite\{([^}]*)\}", cells[0])
        flags = []
        for c in cells[3:]:
            if c == r"\cmark":
                flags.append(True)
            elif c == r"\xmark":
                flags.append(False)
            else:
                raise ValueError(f"{path}: not a check/cross cell {c!r} in {r!r}")
        year, scale = _clean(cells[1]), _clean(cells[2])
        if year != "--":
            int(year)
        int(scale.replace(",", ""))
        rows.append({"name": _clean(cells[0]), "cite": m.group(1) if m else None,
                     "year": (OURS_YEAR if (year == "--" and _clean(cells[0]).startswith("RoboCloth")) else year),
                     "scale": scale, "flags": flags})
    if not columns or not rows:
        raise ValueError(f"{path}: no tabular data found")
    if rows[-1]["name"] != OURS:
        raise ValueError(f"{path}: last row is {rows[-1]['name']!r}, expected {OURS}")
    return {"columns": columns, "rows": rows}


def parse_tex_dir(tex_dir: str) -> dict:
    sets, sources, columns = {}, {}, None
    for fname in ("ours.tex", "ubo.tex"):
        path = os.path.join(tex_dir, fname)
        sources[fname] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        cols, blocks = parse_table(path)
        if columns is None:
            columns = cols
        elif cols != columns:
            raise ValueError(f"column mismatch: {cols} vs {columns}")
        for b in blocks:
            if fname == "ubo.tex":
                key = "ubo2014"
            elif b["title"] and b["title"].lower().startswith("in-domain"):
                key = "robocloth"
            elif b["title"] and "bonn" in b["title"].lower():
                key = "bonn"
            else:
                raise ValueError(f"{fname}: unknown block {b['title']!r}")
            if key in sets:
                raise ValueError(f"duplicate block {key}")
            if b["average"] is None:
                raise ValueError(f"{fname}: block {key} has no Average row")
            sets[key] = dict(b, source=fname, tex_title=b["title"])
            sets[key].pop("title")
    missing = [k for k in SET_ORDER if k not in sets]
    if missing:
        raise ValueError(f"missing blocks: {missing}")
    if OURS not in columns:
        raise ValueError(f"no {OURS} column in {columns}")
    path = os.path.join(tex_dir, "related.tex")
    sources["related.tex"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    related = parse_related(path)
    return {"generated_by": "docs/tools/results/build_results.py",
            "note": "Parsed from the paper's tables/ours.tex, tables/ubo.tex and tables/related.tex; "
                    "do not edit by hand.",
            "sources_sha256": sources, "columns": columns,
            "sets": {k: sets[k] for k in SET_ORDER}, "related": related}


# =========================================================================== data helpers
def fnum(s: str) -> float:
    return float(s.replace("\u2212", "-"))


def fmt_delta(s: str) -> str:
    """'+3.61' -> '+3.61'; '-1.45' -> '−1.45' (true minus)."""
    s = s.strip()
    if s.startswith("-"):
        return "\u2212" + s[1:]
    return s if s.startswith("+") else "+" + s


def res_values(data: dict) -> dict:
    """Keys usable as <span data-res="KEY"> in the page."""
    out, total = {}, 0
    cols = data["columns"]
    for key, s in data["sets"].items():
        n = len(s["rows"])
        total += n
        out[f"{key}.n"] = str(n)
        out[f"{key}.delta"] = fmt_delta(s["average"]["delta"])
        out[f"{key}.delta_abs"] = fmt_delta(s["average"]["delta"]).lstrip("+\u2212")
        for c, v in zip(cols, s["average"]["cells"]):
            out[f"{key}.avg.{c}"] = v
        for r in s["rows"]:
            for c, v in zip(cols, r["cells"]):
                out[f"{key}.{r['material']}.{c}"] = v
            out[f"{key}.{r['material']}.delta"] = fmt_delta(r["delta"])
    out["total.n"] = str(total)
    out["sets.n"] = str(len(data["sets"]))
    out["axis.floor"] = f"{AXIS[0]:.0f}"
    rel = data["related"]
    out["related.n_other"] = str(len(rel["rows"]) - 1)
    for r in rel["rows"]:
        out[f"related.{r['name']}.scale"] = r["scale"]
        out[f"related.{r['name']}.year"] = r["year"]
    return out


def chart_rows(data: dict) -> list:
    """The chart's bars: (method, value text, mark, sub-label) of the shown set, best first."""
    s = data["sets"][CHART_SET]
    rows = [(c, v, m, METHOD_SUB.get(c, "")) for c, v, m in
            zip(data["columns"], s["average"]["cells"], s["average"]["marks"])]
    return sorted(rows, key=lambda r: -fnum(r[1]))


def consistency_report(data: dict) -> list:
    """Checks of the tables against themselves; returns a list of problems (empty = fine)."""
    probs = []
    cols = data["columns"]
    io = cols.index(OURS)
    for key, s in data["sets"].items():
        for r in s["rows"] + [s["average"]]:
            v = [fnum(x) for x in r["cells"]]
            order = sorted(range(len(v)), key=lambda i: -v[i])
            if r["marks"][order[0]] != "best" or r["marks"][order[1]] != "second":
                probs.append(f"{key}/{r['material']}: bold/underline do not match max/second ({r['cells']}, {r['marks']})")
            best_other = max(x for i, x in enumerate(v) if i != io)
            d = round(v[io] - best_other, 2)
            if abs(d - fnum(r["delta"])) > 0.0051:
                probs.append(f"{key}/{r['material']}: delta {r['delta']} != {d:+.2f}")
        for j, c in enumerate(cols):
            mean = sum(fnum(r["cells"][j]) for r in s["rows"]) / len(s["rows"])
            if abs(mean - fnum(s["average"]["cells"][j])) > 0.0051:
                probs.append(f"{key}/Average/{c}: {s['average']['cells'][j]} vs mean of rows {mean:.4f}")
    rel = data["related"]
    nflags = len(rel["columns"]) - len(RELATED_NUMERIC)
    for r in rel["rows"]:
        if len(r["flags"]) != nflags:
            probs.append(f"related/{r['name']}: {len(r['flags'])} flags, expected {nflags}")
    return probs


def prose_claims(data: dict) -> list:
    """Statements made in the page's prose and chart; each must hold for the data (else fix the prose)."""
    io = data["columns"].index(OURS)
    s = data["sets"][CHART_SET]
    rows = chart_rows(data)
    rel = data["related"]
    ours_rel = [r for r in rel["rows"] if r["name"] == OURS]
    claims = [
        ("RoboCloth decoder best on every UBO2014 material and on average",
         all(r["marks"][io] == "best" for r in s["rows"] + [s["average"]])),
        ("chart: the delta shown in the RoboCloth bar is vs the second bar (the next-best method)",
         rows[0][0] == OURS and rows[1][2] == "second"),
        ("related table: every baseline row is tagged and present",
         all(any(r["name"] == b for r in rel["rows"]) for b in RELATED_BASELINES)),
        ("related table: RoboCloth is the only dataset with every property (the footnote says so)",
         len(ours_rel) == 1 and all(ours_rel[0]["flags"]) and
         not any(all(r["flags"]) for r in rel["rows"] if r["name"] != OURS)),
        ("related table: no other dataset has flexible coverage",
         not any(r["flags"][-1] for r in rel["rows"] if r["name"] != OURS)),
    ]
    return claims


# =========================================================================== chart (SVG)
def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _bar_h(x, y, w, h, r=5.0):
    """Bar with a rounded data end (right), square at the baseline."""
    r = min(r, h / 2, w)
    return (f"M{x:.2f},{y:.2f}H{x + w - r:.2f}A{r},{r} 0 0 1 {x + w:.2f},{y + r:.2f}"
            f"V{y + h - r:.2f}A{r},{r} 0 0 1 {x + w - r:.2f},{y + h:.2f}H{x:.2f}Z")


def _svg_open(w, h, title, desc):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-labelledby="t d">',
        f"<title id=\"t\">{_esc(title)}</title>",
        f"<desc id=\"d\">{_esc(desc)}</desc>",
        "<style>"
        f"text{{font-family:{FONT};fill:{INK};font-variant-numeric:tabular-nums}}"
        f".tick,.sub,.axt{{fill:{MUTED};font-weight:400}}.name{{fill:{INK2};font-weight:600}}.name.o{{fill:{INK};font-weight:700}}"
        f".v{{fill:{INK2};font-weight:600}}.v.o{{fill:{INK};font-weight:700}}"
        ".in{fill:#ffffff;font-weight:600}"
        "</style>",
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
    ]


def chart_title(data: dict) -> str:
    n = len(data["sets"][CHART_SET]["rows"])
    return f"Mean PSNR on UBO2014 ({n} held-out materials), best configuration per method"


def chart_desc(data: dict) -> str:
    vals = ", ".join(f"{METHOD_LABEL.get(c, c)} {v}" for c, v, _m, _s in chart_rows(data))
    lo = AXIS[0]
    return f"PSNR in dB, higher is better; the axis starts at {lo:.0f} dB. {vals}."


def _delta_text(data: dict) -> str:
    return f"{fmt_delta(data['sets'][CHART_SET]['average']['delta'])} dB over the next best"


def _axis_break(out, x, y):
    """A small '//' mark on the baseline: the axis does not start at 0."""
    out.append(f'<rect x="{x - 4}" y="{y - 6}" width="8" height="12" fill="#ffffff"/>')
    for dy in (-3, 3):
        out.append(f'<line x1="{x - 3:.1f}" y1="{y + dy + 3:.1f}" x2="{x + 3:.1f}" y2="{y + dy - 3:.1f}" '
                   f'stroke="{MUTED}" stroke-width="1.4" stroke-linecap="round"/>')


def chart_wide(data: dict) -> str:
    rows = chart_rows(data)
    W = 880
    L, R = 172, 862          # plot area (bars start at L)
    T = 44                   # top of the first bar row (the PSNR title sits above)
    pitch, bh = 64, 36
    B = T + pitch * len(rows)
    H = B + 38
    lo, hi, step = AXIS
    kx = (R - L) / (hi - lo)
    X = lambda v: L + (v - lo) * kx
    out = _svg_open(W, H, chart_title(data), chart_desc(data))
    out.append(f'<text class="axt" x="{L}" y="20" font-size="15">PSNR (dB) on UBO2014 \u00b7 higher is better</text>')
    # grid
    v = lo + step
    while v <= hi + 1e-9:
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T - 6}" y2="{B}" stroke="{GRID}" stroke-width="1"/>')
        v += step
    # bars
    for i, (c, val, mark, sub) in enumerate(rows):
        y = T + i * pitch + (pitch - bh) / 2
        ours = c == OURS
        xe = X(fnum(val))
        # method name only (author: no sub-labels; the highlight says which one is ours)
        out.append(f'<text class="name{" o" if ours else ""}" x="{L - 16}" y="{y + bh / 2:.1f}" '
                   f'font-size="18" text-anchor="end" dominant-baseline="middle">{_esc(METHOD_LABEL.get(c, c))}</text>')
        out.append(f'<path d="{_bar_h(L, y, xe - L, bh)}" fill="{ACCENT if ours else OTHER}"/>')
        out.append(f'<text class="v{" o" if ours else ""}" data-set="{CHART_SET}" data-col="{_esc(c)}" '
                   f'x="{xe + 12:.1f}" y="{y + bh / 2:.1f}" font-size="18" dominant-baseline="middle">{_esc(val)}</text>')
    # baseline + ticks
    out.append(f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{BASELINE}" stroke-width="1"/>')
    out.append(f'<line x1="{L}" x2="{L}" y1="{T - 6}" y2="{B}" stroke="{BASELINE}" stroke-width="1"/>')
    v = lo
    while v <= hi + 1e-9:
        lab = f"{v:.0f} dB" if v == lo else f"{v:.0f}"
        anchor = "start" if v == lo else "middle"
        out.append(f'<text class="tick" x="{X(v) + (6 if v == lo else 0):.1f}" y="{B + 24}" font-size="15" '
                   f'text-anchor="{anchor}">{lab}</text>')
        v += step
    out.append("</svg>")
    return "\n".join(out) + "\n"


def chart_narrow(data: dict) -> str:
    rows = chart_rows(data)
    W = 330
    L, R = 0, 330 - 62       # value labels sit to the right of the bars
    head = 34                # the PSNR title
    pitch, bh, name_h = 68, 28, 24
    T = head
    B = T + pitch * len(rows) - (pitch - name_h - bh)
    H = B + 36
    lo, hi, step = AXIS
    kx = (R - L) / (hi - lo)
    X = lambda v: L + (v - lo) * kx
    out = _svg_open(W, H, chart_title(data), chart_desc(data))
    out.append(f'<text class="axt" x="0" y="17" font-size="15">PSNR (dB) on UBO2014 \u2191</text>')
    v = lo + step
    while v <= hi + 1e-9:
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{B}" stroke="{GRID}" stroke-width="1"/>')
        v += step
    for i, (c, val, mark, sub) in enumerate(rows):
        y0 = T + i * pitch
        ours = c == OURS
        xe = X(fnum(val))
        out.append(f'<text class="name{" o" if ours else ""}" x="2" y="{y0 + 15}" font-size="15">{_esc(METHOD_LABEL.get(c, c))}</text>')
        y = y0 + name_h
        out.append(f'<path d="{_bar_h(L, y, xe - L, bh)}" fill="{ACCENT if ours else OTHER}"/>')
        out.append(f'<text class="v{" o" if ours else ""}" data-set="{CHART_SET}" data-col="{_esc(c)}" '
                   f'x="{xe + 8:.1f}" y="{y + bh / 2:.1f}" font-size="15" dominant-baseline="middle">{_esc(val)}</text>')
    out.append(f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{BASELINE}" stroke-width="1"/>')
    out.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{B}" stroke="{BASELINE}" stroke-width="1"/>')
    v = lo
    while v <= hi + 1e-9:
        lab = f"{v:.0f} dB" if v == lo else f"{v:.0f}"
        anchor = "start" if v == lo else "middle"
        out.append(f'<text class="tick" x="{X(v) + (6 if v == lo else 0):.1f}" y="{B + 22}" font-size="15" '
                   f'text-anchor="{anchor}">{lab}</text>')
        v += step
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_pngs(pairs):
    """[(svg_path, png_path)] -> PNGs at 2x, rendered by headless Chrome."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        for svg, png in pairs:
            txt = open(svg, encoding="utf-8").read()
            w, h = (int(float(x)) for x in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', txt).groups())
            pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            pg.goto("file://" + svg)
            pg.wait_for_timeout(150)
            pg.screenshot(path=png, clip={"x": 0, "y": 0, "width": w, "height": h})
            pg.close()
        b.close()


# =========================================================================== HTML
def _ind(block: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln else ln for ln in block.splitlines())


def html_chart(data: dict) -> str:
    wide = f"{IMG_URL}/psnr_averages.svg?v={ASSET_V}"
    narrow = f"{IMG_URL}/psnr_averages_narrow.svg?v={ASSET_V}"
    ws = re.search(r'viewBox="0 0 (\d+) (\d+)"', chart_wide(data)).groups()
    ns = re.search(r'viewBox="0 0 (\d+) (\d+)"', chart_narrow(data)).groups()
    alt = "Horizontal bar chart. " + chart_title(data) + ". " + chart_desc(data)
    return "\n".join([
        "<picture>",
        f'  <source media="(max-width: {CHART_NARROW_MAX}px)" srcset="{narrow}" width="{ns[0]}" height="{ns[1]}">',
        f'  <img class="res-chart-img" src="{wide}" width="{ws[0]}" height="{ws[1]}" loading="lazy" decoding="async"',
        f'       alt="{_esc(alt)}">',
        "</picture>",
    ])


GLYPH_YES = ('<span class="res-glyph res-yes" role="img" aria-label="yes"><svg viewBox="0 0 16 16" aria-hidden="true">'
             '<path d="M3 8.6l3.2 3.2L13 5" fill="none" stroke="currentColor" stroke-width="2.2" '
             'stroke-linecap="round" stroke-linejoin="round"/></svg></span>')
GLYPH_NO = ('<span class="res-glyph res-no" role="img" aria-label="no"><svg viewBox="0 0 16 16" aria-hidden="true">'
            '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round"/></svg></span>')


def html_related(data: dict) -> str:
    rel = data["related"]
    cols = rel["columns"]
    lines = ['<table class="res-table res-related">',
             "  <colgroup>",
             '    <col class="res-c-stub">',
             *('    <col class="res-c-num">' for _ in RELATED_NUMERIC),
             *('    <col class="res-c-flag">' for _ in cols[len(RELATED_NUMERIC):]),
             "  </colgroup>",
             "  <thead><tr>",
             '    <th scope="col" class="res-stub">Dataset</th>']
    for c in cols:
        cls = ' class="res-num"' if c in RELATED_NUMERIC else ""
        lines.append(f'    <th scope="col"{cls}>{_esc(RELATED_COLS[c])}</th>')
    lines += ["  </tr></thead>", "  <tbody>"]
    for r in rel["rows"]:
        ours = r["name"] == OURS
        cls = ' class="res-ours"' if ours else (' class="res-base"' if r["name"] in RELATED_BASELINES else "")
        tag = RELATED_TAGS.get(r["name"])
        tag = f' <span class="res-tag{" res-tag-ours" if ours else ""}">{_esc(tag)}</span>' if tag else ""
        year = "\u2014" if r["year"] == "--" else r["year"]
        lines.append(f"    <tr{cls}>")
        lines.append(f'      <th scope="row" class="res-stub">{_esc(r["name"])}{tag}</th>')
        lines.append(f'      <td class="res-num">{_esc(year)}</td>')
        lines.append(f'      <td class="res-num">{_esc(r["scale"])}</td>')
        for f in r["flags"]:
            lines.append("      <td>" + (GLYPH_YES if f else GLYPH_NO) + "</td>")
        lines.append("    </tr>")
    lines += ["  </tbody>", "</table>"]
    return "\n".join(lines)


BLOCKS = {
    "results-chart": html_chart,
    "results-related": html_related,
}


def _section(page: str):
    m = re.search(r'<section id="main-viz"[^>]*>.*?</section>', page, re.S)
    if not m:
        raise ValueError('index.html: <section id="main-viz"> not found')
    return m.start(), m.end()


def fill_page(page: str, data: dict) -> str:
    a, b = _section(page)
    sec = page[a:b]
    for name, fn in BLOCKS.items():
        pat = re.compile(r"([ \t]*)(<!-- BEGIN generated:%s\b[^>]*-->)(.*?)(<!-- END generated:%s -->)"
                         % (re.escape(name), re.escape(name)), re.S)
        m = pat.search(sec)
        if not m:
            raise ValueError(f"index.html: markers for {name} not found in #main-viz")
        indent = len(m.group(1).expandtabs())
        new = m.group(1) + m.group(2) + "\n" + _ind(fn(data), indent) + "\n" + m.group(1) + m.group(4)
        sec = sec[:m.start()] + new + sec[m.end():]
    vals = res_values(data)

    def sub(m):
        key = m.group(2)
        if key not in vals:
            raise ValueError(f"index.html: unknown data-res key {key!r}")
        return f"<span{m.group(1)}>{vals[key]}</span>"
    sec = re.sub(r'<span((?:(?!>).)*?\bdata-res="([^"]+)"[^>]*)>(.*?)</span>', sub, sec)
    return page[:a] + sec + page[b:]


# =========================================================================== check
def check(data: dict, tex_dir: str | None) -> bool:
    ok = True

    def bad(msg):
        nonlocal ok
        ok = False
        print("  FAIL  " + msg)

    if tex_dir:
        fresh = parse_tex_dir(tex_dir)
        if any(fresh[k] != data.get(k) for k in ("sets", "columns", "related")):
            bad("results_data.json differs from the .tex (re-run with --tex-dir to refresh)")
        else:
            print("  ok    results_data.json == parsed .tex (" +
                  ", ".join(f"{k} {v[:12]}" for k, v in fresh["sources_sha256"].items()) + ")")
    for p in consistency_report(data):
        bad("table self-consistency: " + p)
    for text, holds in prose_claims(data):
        (print("  ok    claim: " + text) if holds else bad("claim no longer true: " + text))

    # chart: every value label in both SVGs vs the data
    print(f"\n  chart values (SVG text vs the {SETS[CHART_SET]['title']} Average row of the table):")
    print(f"  {'method':<12}{'tex':>8}{'wide':>8}{'narrow':>8}")
    labels = {}
    for tag, fname in (("wide", "psnr_averages.svg"), ("narrow", "psnr_averages_narrow.svg")):
        txt = open(os.path.join(IMG_DIR, fname), encoding="utf-8").read()
        for s_, c_, v_ in re.findall(r'<text class="v[^"]*" data-set="([^"]+)" data-col="([^"]+)"[^>]*>([^<]+)</text>', txt):
            labels[(tag, s_, html.unescape(c_))] = v_
    for c, v, _m, _s in chart_rows(data):
        w_, n_ = labels.get(("wide", CHART_SET, c), "-"), labels.get(("narrow", CHART_SET, c), "-")
        flag = "" if (w_ == v and n_ == v) else "   <-- MISMATCH"
        print(f"  {c:<12}{v:>8}{w_:>8}{n_:>8}{flag}")
        if flag:
            bad(f"chart label {CHART_SET}/{c}")
    if len(labels) != 2 * len(data["columns"]):
        bad(f"chart has {len(labels)} value labels, expected {2 * len(data['columns'])}")
    for fname in ("psnr_averages.svg", "psnr_averages_narrow.svg"):
        if open(os.path.join(IMG_DIR, fname), encoding="utf-8").read() != \
                (chart_wide if "narrow" not in fname else chart_narrow)(data):
            bad(f"{fname} is stale (re-run the build)")
    print()

    # page: generated blocks + data-res spans up to date, no stray numbers, no venue strings
    page = open(INDEX, encoding="utf-8").read()
    if fill_page(page, data) != page:
        bad("index.html generated blocks / data-res spans are stale (re-run the build)")
    else:
        print("  ok    index.html generated blocks and data-res spans match the data")
    a, b = _section(page)
    sec_text = re.sub(r"<!--.*?-->", "", page[a:b], flags=re.S)
    nums_in_section = re.findall(r"(?<![\w.#?=/-])[\u2212+]?\d+\.\d{2}(?![\w.])", sec_text)
    allowed = set()
    for k, v in res_values(data).items():
        allowed.add(v.lstrip("+\u2212"))
    stray = sorted({n.lstrip("+\u2212") for n in nums_in_section} - allowed)
    if stray:
        bad(f"numbers in #main-viz that are not in the tables: {stray}")
    else:
        print(f"  ok    all {len(nums_in_section)} two-decimal numbers in #main-viz come from the tables")
    rel_names = {r["name"] for r in data["related"]["rows"]}
    ints = set(re.findall(r"(?<![\w.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?![\w.,])", re.sub(r"<[^>]+>", " ", sec_text)))
    allowed_ints = {r["scale"] for r in data["related"]["rows"]} | {r["year"] for r in data["related"]["rows"]} \
        | {str(len(s["rows"])) for s in data["sets"].values()} | {f"{AXIS[0]:.0f}"}
    allowed_ints |= {"1", "2"}          # "Stage 1" / "Stage 2" are names in the prose, not data
    stray_ints = sorted(ints - allowed_ints)
    if stray_ints:
        bad(f"integers in #main-viz prose that are not in the tables: {stray_ints}")
    else:
        print(f"  ok    all integers in #main-viz prose come from the tables ({len(rel_names)} dataset rows)")
    venue = re.findall(r"(?i)in submission|under review|3DV|NeurIPS|CVPR|ICCV|ECCV|SIGGRAPH", sec_text)
    if venue:
        bad(f"venue / submission wording in #main-viz: {sorted(set(venue))}")
    else:
        print("  ok    no venue / submission wording in #main-viz")
    return ok


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tex-dir", help="the paper's tables/ directory (ours.tex, ubo.tex, related.tex)")
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    ap.add_argument("--no-png", action="store_true", help="skip the PNG exports")
    args = ap.parse_args()

    if args.check:
        data = json.load(open(DATA_JSON, encoding="utf-8"))
        sys.exit(0 if check(data, args.tex_dir) else 1)

    if args.tex_dir:
        data = parse_tex_dir(args.tex_dir)
        with open(DATA_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"  wrote {os.path.relpath(DATA_JSON, DOCS)}")
    else:
        data = json.load(open(DATA_JSON, encoding="utf-8"))
    probs = consistency_report(data)
    for p in probs:
        print("  WARN  " + p)

    os.makedirs(IMG_DIR, exist_ok=True)
    pairs = []
    for fname, fn in (("psnr_averages", chart_wide), ("psnr_averages_narrow", chart_narrow)):
        svg = os.path.join(IMG_DIR, fname + ".svg")
        with open(svg, "w", encoding="utf-8") as f:
            f.write(fn(data))
        pairs.append((svg, os.path.join(IMG_DIR, fname + "_2x.png")))
        print(f"  wrote images/results/{fname}.svg")
    if not args.no_png:
        render_pngs(pairs)
        for _s, p in pairs:
            print(f"  wrote images/results/{os.path.basename(p)}")

    page = open(INDEX, encoding="utf-8").read()
    new = fill_page(page, data)
    if new != page:
        with open(INDEX, "w", encoding="utf-8") as f:
            f.write(new)
        print("  updated index.html (#main-viz generated blocks)")
    else:
        print("  index.html already up to date")
    print("\ncheck:")
    sys.exit(0 if check(data, args.tex_dir) else 1)


if __name__ == "__main__":
    main()
