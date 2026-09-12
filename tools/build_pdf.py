"""Render the Course Module document to PDF via HTML + WeasyPrint."""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course_spec import COURSE, LECTURES, PARTS, folder_name, slug  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Deep_Learning_Course_Module.pdf"

PART_COLOURS = {0: "#4a5265", 1: "#1f6fb2", 2: "#1b866b", 3: "#c16a1c", 4: "#7a3ea8"}

CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 18mm 18mm;
  @bottom-center {
    content: counter(page);
    font-family: "Segoe UI", Calibri, sans-serif;
    font-size: 8.5pt; color: #7a8194;
  }
  @top-right {
    content: "DL-601 Deep Learning — Course Module";
    font-family: "Segoe UI", Calibri, sans-serif;
    font-size: 8pt; color: #9aa1b1;
  }
}
@page :first { @top-right { content: ""; } @bottom-center { content: ""; } }

body { font-family: "Segoe UI", Calibri, sans-serif; font-size: 10pt;
       line-height: 1.5; color: #1a1f2b; }
h1, h2, h3, h4 { line-height: 1.25; }

.cover { height: 245mm; display: flex; flex-direction: column; justify-content: center; }
.cover .rule { height: 5px; width: 70mm; background: #1f6fb2; margin: 8mm 0; }
.cover .code { font-size: 13pt; letter-spacing: 3px; color: #1f6fb2; font-weight: 700; }
.cover h1 { font-size: 38pt; margin: 2mm 0; }
.cover h2 { font-size: 17pt; font-weight: 400; color: #5a6375; margin: 0 0 6mm 0; }
.cover .meta { font-size: 10.5pt; color: #5a6375; }
.cover .meta strong { color: #1a1f2b; }
.cover .footer { margin-top: 14mm; font-size: 9pt; color: #7a8194; }

.page-break { page-break-before: always; }
.avoid-break { page-break-inside: avoid; }

h1.section { font-size: 21pt; margin: 0 0 2mm 0; padding-bottom: 2mm;
             border-bottom: 3px solid #1f6fb2; }
h2.part { font-size: 16pt; margin: 8mm 0 3mm 0; padding: 2mm 3mm;
          background: #f4f6fa; border-left: 5px solid #1f6fb2; }

table { width: 100%; border-collapse: collapse; margin: 3mm 0 5mm 0; font-size: 9pt; }
th { background: #eef1f7; text-align: left; padding: 2mm 2.5mm;
     border-bottom: 2px solid #c9d2e0; font-weight: 600; }
td { padding: 1.8mm 2.5mm; border-bottom: 1px solid #e4e8f0; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfd; }
td.num { text-align: right; white-space: nowrap; }

ul, ol { margin: 2mm 0 4mm 0; padding-left: 6mm; }
li { margin-bottom: 1.4mm; }

.lecture { page-break-inside: avoid; margin-bottom: 7mm; }
.lecture-head { padding: 2.5mm 3mm; border-radius: 2px; color: #fff; }
.lecture-head .num { font-size: 8.5pt; opacity: .85; letter-spacing: 1.5px; }
.lecture-head .title { font-size: 13pt; font-weight: 700; }
.lecture-head .tagline { font-size: 9.5pt; font-style: italic; opacity: .92; }
.lecture-body { border: 1px solid #e4e8f0; border-top: none; padding: 3mm; }
.lecture-body h4 { font-size: 9.5pt; margin: 2.5mm 0 1.2mm 0;
                   text-transform: uppercase; letter-spacing: .6px; color: #5a6375; }
.lecture-body p { margin: 0 0 2mm 0; }
.lecture-body ul { margin: 0 0 2mm 0; }
.chips { margin-top: 2mm; }
.chip { display: inline-block; font-size: 8pt; padding: 0.8mm 2mm; margin-right: 1.5mm;
        background: #f0f3f8; border: 1px solid #dde3ec; border-radius: 2px;
        color: #4a5265; font-family: Consolas, monospace; }
.callout { background: #f4f6fa; border-left: 4px solid #1f6fb2;
           padding: 3mm 4mm; margin: 4mm 0; font-size: 9.5pt; }
code { font-family: Consolas, monospace; font-size: 8.8pt;
       background: #f0f3f8; padding: 0.3mm 1mm; border-radius: 2px; }
pre { background: #f7f9fc; border: 1px solid #e4e8f0; padding: 3mm;
      font-family: Consolas, monospace; font-size: 8.3pt; line-height: 1.45;
      white-space: pre-wrap; }
.toc-row { display: flex; font-size: 9.5pt; padding: 1.2mm 0;
           border-bottom: 1px dotted #dde3ec; }
.toc-row .n { width: 12mm; color: #7a8194; font-weight: 600; }
.toc-row .t { flex: 1; }
.small { font-size: 8.5pt; color: #7a8194; }
"""


def esc(t: str) -> str:
    return html.escape(str(t))


def md_inline(text: str) -> str:
    """Inline `code` only.

    Deliberately does NOT interpret `*` as italics: lecture text contains
    expressions like `2*padding - dilation*(kernel - 1)`, and treating those
    asterisks as markup silently mangles the formula.
    """
    out = esc(text)
    while out.count("`") >= 2:
        out = out.replace("`", "<code>", 1).replace("`", "</code>", 1)
    return out


def md_reading(text: str) -> str:
    """Reading entries do use *italics* for titles, so handle them here."""
    out = md_inline(text)
    while out.count("*") >= 2:
        out = out.replace("*", "<em>", 1).replace("*", "</em>", 1)
    return out


def cover() -> str:
    n_lectures = len(LECTURES)
    n_core = sum(1 for l in LECTURES for t in l["tasks"] if t[0] == "core")
    n_stretch = sum(1 for l in LECTURES for t in l["tasks"] if t[0] == "stretch")
    return f"""
    <div class="cover">
      <div class="code">{esc(COURSE['code'])}</div>
      <h1>{esc(COURSE['title'])}</h1>
      <h2>{esc(COURSE['subtitle'])}</h2>
      <div class="rule"></div>
      <div class="meta">
        <p><strong>Course Module Document</strong></p>
        <p>{esc(COURSE['level'])}<br>{esc(COURSE['credits'])}</p>
        <p><strong>Instructor:</strong> {esc(COURSE['instructor'])}</p>
      </div>
      <div class="footer">
        {n_lectures} lectures (0-16) · {n_lectures} slide decks · {n_lectures} hands-on
        notebooks · {n_core} core tasks + {n_stretch} stretch<br>
        Every notebook runs end to end on a CPU laptop. No downloads, no GPU, no API keys.
      </div>
    </div>
    """


def overview() -> str:
    parts = []
    add = parts.append
    add('<div class="page-break"><h1 class="section">1. Course overview</h1>')
    add("<p>This module takes a student with <strong>no machine learning background at all</strong> to the point of building, diagnosing and deploying modern neural networks. "
        "Lecture 0 is a self-contained primer on classical machine learning; Lectures 1 to 16 "
        "are deep learning. The emphasis is on images, with text introduced where it is needed "
        "to motivate sequence models and attention, and a deployment thread running through "
        "the whole course.</p>")

    add("<p>Curriculum design follows the consensus structure of Stanford CS231n, "
        "NYU DS-GA 1008 (LeCun and Canziani) and the deeplearning.ai Deep Learning "
        "Specialization, re-sequenced for a 16-session semester, prefaced with a machine "
            "learning primer, and extended with the "
        "production engineering those courses leave out.</p>")

    add('<div class="callout"><strong>What makes this module different:</strong> every '
        "technique is implemented from first principles before the library version is "
        "used. Students write backpropagation, convolution, attention, NMS and a "
        "diffusion sampler themselves. They then serve a model behind a tested HTTP API, "
        "because a model nobody can call is not a deliverable.</div>")

    add("<h3>Learning outcomes</h3><p>On completing this module, a student can:</p><ol>")
    for o in COURSE["outcomes"]:
        parts.append(f"<li>{md_inline(o)}</li>")
    add("</ol>")

    add("<h3>Prerequisites</h3><ul>")
    for p in COURSE["prerequisites"]:
        parts.append(f"<li>{md_inline(p)}</li>")
    add("</ul>")

    add("<h3>Technology stack</h3><ul>")
    for s in COURSE["stack"]:
        parts.append(f"<li>{md_inline(s)}</li>")
    add("</ul>")

    add("<h3>Structure</h3>")
    add("<table><tr><th>Part</th><th>Lectures</th><th>Focus</th></tr>")
    focus = {
        0: "Machine learning from zero: features, splits, overfitting, metrics. No deep learning yet.",
        1: "Tensors, linear models, backpropagation, optimisation. Everything built by hand.",
        2: "Convolution, ResNets, transfer learning, augmentation, evaluation and interpretability.",
        3: "Deployment with FastAPI, then detection, segmentation and sequence models.",
        4: "Attention, Vision Transformers, self-supervision, generative models, production systems.",
    }
    for p in (0, 1, 2, 3, 4):
        nums = [l["number"] for l in LECTURES if l["part"] == p]
        parts.append(f"<tr><td><strong>{esc(PARTS[p])}</strong></td>"
                     f"<td class='num'>{nums[0]}–{nums[-1]}</td>"
                     f"<td>{esc(focus[p])}</td></tr>")
    add("</table>")
    add("</div>")
    return "\n".join(parts)


def toc() -> str:
    rows = ['<div class="page-break"><h1 class="section">2. Lecture schedule</h1>']
    current_part = None
    for lec in LECTURES:
        if lec["part"] != current_part:
            current_part = lec["part"]
            rows.append(f'<h2 class="part" style="border-left-color:{PART_COLOURS[current_part]}">'
                        f'{esc(PARTS[current_part])}</h2>')
        rows.append(
            f'<div class="toc-row"><div class="n">{lec["number"]:02d}</div>'
            f'<div class="t"><strong>{esc(lec["title"])}</strong><br>'
            f'<span class="small">{esc(lec["tagline"])}</span></div></div>')
    rows.append("</div>")
    return "\n".join(rows)


def assessment() -> str:
    parts = ['<div class="page-break"><h1 class="section">3. Assessment and policies</h1>']
    parts.append("<h3>Assessment breakdown</h3>")
    parts.append("<table><tr><th>Component</th><th>Weight</th><th>Description</th></tr>")
    for name, weight, desc in COURSE["assessment"]:
        parts.append(f"<tr><td>{esc(name)}</td><td class='num'>{esc(weight)}</td>"
                     f"<td>{esc(desc)}</td></tr>")
    parts.append("</table>")

    parts.append("<h3>Per-task marking rubric</h3>")
    parts.append("<table><tr><th>Criterion</th><th>Weight</th><th>Full marks</th></tr>")
    for c, w, d in [
        ("Correctness", "40%", "Every core task produces the required output, and the numbers are right."),
        ("Code quality", "20%", "Readable, functions documented, no copy-pasted blocks, no dead code."),
        ("Analysis", "25%", "Written answers explain why, not just what. Claims backed by a number or a plot."),
        ("Reproducibility", "15%", "Runs top to bottom from a clean kernel with the seed fixed."),
    ]:
        parts.append(f"<tr><td>{esc(c)}</td><td class='num'>{esc(w)}</td><td>{esc(d)}</td></tr>")
    parts.append("</table>")

    parts.append("<h3>Grading scale</h3><table><tr>")
    for g, r in COURSE["grading_scale"]:
        parts.append(f"<th>{esc(g)}</th>")
    parts.append("</tr><tr>")
    for g, r in COURSE["grading_scale"]:
        parts.append(f"<td class='num'>{esc(r)}</td>")
    parts.append("</tr></table>")

    parts.append("<h3>Policies</h3><ul>")
    for p in COURSE["policies"]:
        parts.append(f"<li>{md_inline(p)}</li>")
    parts.append("</ul>")

    parts.append("<h3>Weekly rhythm</h3>")
    parts.append("<table><tr><th>Block</th><th>Duration</th><th>Activity</th></tr>"
                 "<tr><td>Lecture</td><td class='num'>2 h</td>"
                 "<td>Slide deck, worked derivations, live code from the notebook.</td></tr>"
                 "<tr><td>Lab</td><td class='num'>1 h</td>"
                 "<td>Supervised work on the notebook; instructor circulates.</td></tr>"
                 "<tr><td>Independent</td><td class='num'>4–6 h</td>"
                 "<td>Core tasks, reading, and the running results table.</td></tr></table>")
    parts.append("</div>")
    return "\n".join(parts)


def repository() -> str:
    tree = """DeepLearning/
├── Deep_Learning_Course_Module.pdf     this document
├── README.md                           start here
├── requirements.txt
├── dlcourse/                           shared helpers imported by every notebook
│   ├── data.py                         dataset loaders
│   ├── training.py                     the reusable training loop
│   └── viz.py                          plotting helpers
├── data/                               generated offline; no downloads, ever
│   ├── shapes_32.npz                   4-class 32x32 classification set
│   ├── shapes_imagefolder/             the same data as PNG files
│   ├── digits_8x8.npz                  scikit-learn handwritten digits
│   ├── shapes_det.npz  + annotations   multi-object detection scenes
│   ├── shapes_seg.npz                  per-pixel segmentation masks
│   ├── shapes_pairs.npz                unlabelled pool for self-supervision
│   ├── sentiment.csv, corpus.txt       text corpora
│   └── captions.csv                    image-caption pairs
├── lectures/
│   └── Lecture_01 ... Lecture_16/
│       ├── README.md                   objectives, outline, teaching notes
│       ├── slides/*.pptx               the lecture deck
│       ├── notebooks/*.ipynb           the hands-on lab
│       ├── tasks/*_Tasks.md            student tasks and rubric
│       ├── data/README.md              which datasets this lecture uses
│       └── app/                        FastAPI service (Lectures 9 and 16)
└── tools/                              generators for every artefact above
"""
    return f"""
    <div class="page-break"><h1 class="section">4. Repository layout</h1>
    <pre>{esc(tree)}</pre>
    <h3>Getting started</h3>
    <pre>{esc('''# 1. install dependencies
pip install -r requirements.txt

# 2. generate the datasets (once, ~30 seconds, fully offline)
python tools/build_data.py

# 3. open the first lab
jupyter lab lectures/Lecture_01/notebooks/

# 4. for the serving lectures
cd lectures/Lecture_09/app
python train_and_export.py
uvicorn main:app --reload      # then open http://127.0.0.1:8000/docs''')}</pre>
    <div class="callout"><strong>Design constraint:</strong> every dataset is generated
    procedurally by a seeded script, so the repository is self-contained and reproducible
    byte for byte. Nothing downloads. No notebook needs a GPU, an API key, or an internet
    connection.</div>
    </div>
    """


def lecture_block(lec) -> str:
    colour = PART_COLOURS[lec["part"]]
    core = [t for t in lec["tasks"] if t[0] == "core"]
    stretch = [t for t in lec["tasks"] if t[0] == "stretch"]

    outline = "".join(
        f"<li><strong>{esc(s)}</strong> — {md_inline(b[0])}</li>"
        for s, b in lec["outline"])
    objectives = "".join(f"<li>{md_inline(o)}</li>" for o in lec["objectives"])
    tasks = "".join(f"<li>{md_inline(t[1])}</li>" for t in core)
    reading = "".join(f"<li>{md_reading(r)}</li>" for r in lec["reading"])

    app_chip = ('<span class="chip">FastAPI service</span>' if lec.get("app") else "")

    return f"""
    <div class="lecture">
      <div class="lecture-head" style="background:{colour}">
        <div class="num">LECTURE {lec['number']:02d} · {esc(PARTS[lec['part']])}</div>
        <div class="title">{esc(lec['title'])}</div>
        <div class="tagline">{esc(lec['tagline'])}</div>
      </div>
      <div class="lecture-body">
        <h4>Learning objectives</h4><ul>{objectives}</ul>
        <h4>Lecture outline</h4><ol>{outline}</ol>
        <h4>Hands-on lab</h4><p>{md_inline(lec['lab'])}</p>
        <h4>Core tasks ({len(core)} required, {len(stretch)} stretch)</h4><ol>{tasks}</ol>
        <h4>Reading</h4><ul>{reading}</ul>
        <div class="chips">
          <span class="chip">data: {esc(lec['dataset'])}</span>
          <span class="chip">slides: {len(lec['outline'])} sections</span>
          {app_chip}
        </div>
      </div>
    </div>
    """


def lectures_section() -> str:
    parts = ['<div class="page-break"><h1 class="section">5. Lecture specifications</h1>']
    current = None
    for lec in LECTURES:
        if lec["part"] != current:
            current = lec["part"]
            parts.append(f'<h2 class="part" style="border-left-color:{PART_COLOURS[current]}">'
                         f'{esc(PARTS[current])}</h2>')
        parts.append(lecture_block(lec))
    parts.append("</div>")
    return "\n".join(parts)


def appendix() -> str:
    total_core = sum(1 for l in LECTURES for t in l["tasks"] if t[0] == "core")
    total_stretch = sum(1 for l in LECTURES for t in l["tasks"] if t[0] == "stretch")
    rows = "".join(
        f"<tr><td class='num'>{l['number']:02d}</td><td>{esc(l['title'])}</td>"
        f"<td class='num'>{sum(1 for t in l['tasks'] if t[0]=='core')}</td>"
        f"<td class='num'>{sum(1 for t in l['tasks'] if t[0]=='stretch')}</td>"
        f"<td><code>{esc(l['dataset'])}</code></td></tr>"
        for l in LECTURES)

    return f"""
    <div class="page-break"><h1 class="section">6. Appendix</h1>
    <h3>Task and dataset index</h3>
    <table>
      <tr><th>#</th><th>Lecture</th><th>Core</th><th>Stretch</th><th>Datasets</th></tr>
      {rows}
      <tr><th></th><th>Total</th><th class="num">{total_core}</th>
          <th class="num">{total_stretch}</th><th></th></tr>
    </table>

    <h3>Instructor notes</h3>
    <ul>
      <li>Each notebook maps section-for-section onto its slide deck, so you can present
          the deck and run the matching notebook cells live.</li>
      <li>Cells taking more than about a minute on a laptop are marked in the notebook.
          Reduce the epoch count in class and let students run the full version later.</li>
      <li>Every result quoted in the slides was produced by running the notebook on CPU.
          If a number differs on your machine, the seed and thread count are the first
          things to check.</li>
      <li>Lectures 9 and 16 need no extra setup beyond <code>train_and_export.py</code>;
          the FastAPI service runs in-process via <code>TestClient</code> inside the
          notebook, so no port needs to be free during class.</li>
      <li>The running <code>results.md</code> table students maintain across all 16
          lectures is the single most useful artefact they produce. Insist on it from
          Lecture 1.</li>
    </ul>

    <h3>Sources consulted for curriculum design</h3>
    <ul>
      <li>Stanford CS231n, <em>Deep Learning for Computer Vision</em> — lecture schedule
          and assignment structure.</li>
      <li>NYU DS-GA 1008, <em>Deep Learning</em> (Yann LeCun, Alfredo Canziani) — theme
          organisation and the energy-based perspective.</li>
      <li>deeplearning.ai <em>Deep Learning Specialization</em> (Andrew Ng) — the
          foundations sequence and practical training recipes.</li>
      <li>fast.ai <em>Practical Deep Learning for Coders</em> — the top-down teaching
          approach and transfer-learning emphasis.</li>
      <li>Goodfellow, Bengio and Courville, <em>Deep Learning</em> (MIT Press) — the
          reference text throughout.</li>
    </ul>
    </div>
    """


def build_html() -> str:
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>{esc(COURSE['code'])} — Course Module</title>
    <style>{CSS}</style></head><body>
    {cover()}{overview()}{toc()}{assessment()}{repository()}{lectures_section()}{appendix()}
    </body></html>"""


BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser",
]


def render_with_weasyprint(html_path: Path) -> bool:
    """Preferred on Linux/macOS. On Windows it needs GTK, which is usually absent."""
    try:
        from weasyprint import HTML
    except Exception:
        return False
    try:
        HTML(filename=str(html_path), base_url=str(ROOT)).write_pdf(OUT)
        return True
    except Exception as e:
        print(f"  weasyprint unavailable ({type(e).__name__}), falling back to a browser")
        return False


def render_with_browser(html_path: Path) -> bool:
    """Headless Chrome/Edge honours the print CSS, including @page rules."""
    import shutil
    import subprocess

    for candidate in BROWSERS:
        exe = candidate if Path(candidate).exists() else shutil.which(candidate)
        if not exe:
            continue
        cmd = [
            exe, "--headless", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=10000",
            f"--print-to-pdf={OUT}",
            html_path.as_uri(),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
            if OUT.exists() and OUT.stat().st_size > 10_000:
                print(f"  rendered with {Path(exe).name}")
                return True
        except Exception as e:
            print(f"  {Path(exe).name} failed: {type(e).__name__}")
    return False


def main():
    html_text = build_html()
    html_path = ROOT / "build" / "course_module.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")
    print(f"wrote {html_path.relative_to(ROOT)}")

    if OUT.exists():
        OUT.unlink()

    if not (render_with_weasyprint(html_path) or render_with_browser(html_path)):
        raise SystemExit(
            "Could not render the PDF. Install WeasyPrint's GTK dependencies, or "
            "install Chrome/Edge, or open build/course_module.html and print to PDF."
        )

    print(f"wrote {OUT.name}  ({OUT.stat().st_size / 1024:.0f} KB)")
    try:
        import pypdf
        print(f"pages: {len(pypdf.PdfReader(str(OUT)).pages)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
