"""Write the per-lecture student task sheet and the lecture README."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course_spec import COURSE, LECTURES, PARTS, folder_name, slug  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

RUBRIC = """
| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |
""".strip()


def task_sheet(lec) -> str:
    core = [t for t in lec["tasks"] if t[0] == "core"]
    stretch = [t for t in lec["tasks"] if t[0] == "stretch"]
    n = lec["number"]
    nb = f"L{n:02d}_{slug(lec['title'])[:44]}.ipynb"

    out = []
    add = out.append
    add(f"# Lecture {n:02d} — Tasks")
    add(f"## {lec['title']}")
    add("")
    add(f"> {lec['tagline']}")
    add("")
    add(f"**Course:** {COURSE['code']} {COURSE['title']} · **Part:** {PARTS[lec['part']]}  ")
    add(f"**Weight:** 1.5% of the final grade · **Due:** before Lecture {n + 1:02d}"
        if n < 16 else
        f"**Weight:** 1.5% of the final grade · **Due:** with the capstone submission")
    add("")
    add("---")
    add("")
    add("## What you should be able to do after this")
    add("")
    for o in lec["objectives"]:
        add(f"- {o}")
    add("")
    add("## Before you start")
    add("")
    add("```bash")
    add("# from the repository root")
    add("pip install -r requirements.txt")
    add("python tools/build_data.py          # only needed once")
    add(f"jupyter lab lectures/{folder_name(n)}/notebooks/{nb}")
    add("```")
    add("")
    add(f"**Data used:** `{lec['dataset']}`  ")
    add("**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`")
    add("")
    add("Work through the notebook first — it builds the ideas these tasks assume. "
        "Every task below is marked `TODO` in the notebook at the point where it belongs.")
    add("")
    add("---")
    add("")
    add(f"## Core tasks ({len(core)} required)")
    add("")
    add("All core tasks must be attempted. Each is worth an equal share of the task mark.")
    add("")
    for i, (_, title, desc) in enumerate(core, start=1):
        add(f"### Task {i} — {title}")
        add("")
        add(desc)
        add("")
        add("**Deliverable:** working code in the notebook, plus the requested number, table or plot. "
            "Where the task asks you to explain something, write it in a markdown cell directly beneath "
            "the code — two or three sentences is the right length.")
        add("")
    if stretch:
        add("---")
        add("")
        add(f"## Stretch tasks ({len(stretch)}, optional)")
        add("")
        add("Not marked, but these are where the subject gets interesting. "
            "Attempt at least one over the semester if you are aiming for an A.")
        add("")
        for i, (_, title, desc) in enumerate(stretch, start=1):
            add(f"### Stretch {i} — {title}")
            add("")
            add(desc)
            add("")
    add("---")
    add("")
    add("## Submission checklist")
    add("")
    add("- [ ] Notebook runs top to bottom from a restarted kernel with no errors.")
    add("- [ ] `set_seed(0)` is called before anything random happens.")
    add("- [ ] Every core task is answered, in order, under its own heading.")
    add("- [ ] Every plot has axis labels and a title.")
    add("- [ ] Written answers are in markdown cells, not in code comments.")
    add("- [ ] Results added to your running `results.md` table (carried across all 16 lectures).")
    add("- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.")
    add("")
    add(f"Submit as: `LASTNAME_FIRSTNAME_L{n:02d}.ipynb`")
    add("")
    add("## How this is marked")
    add("")
    add(RUBRIC)
    add("")
    add("## Reading")
    add("")
    for r in lec["reading"]:
        add(f"- {r}")
    add("")
    if n < 16:
        nxt = LECTURES[n]
        add("---")
        add("")
        add(f"**Next lecture —** {nxt['number']:02d}: {nxt['title']}. {nxt['tagline']}")
        add("")
    return "\n".join(out)


def lecture_readme(lec) -> str:
    n = lec["number"]
    nb = f"L{n:02d}_{slug(lec['title'])[:44]}.ipynb"
    core = sum(1 for t in lec["tasks"] if t[0] == "core")
    stretch = sum(1 for t in lec["tasks"] if t[0] == "stretch")

    out = []
    add = out.append
    add(f"# Lecture {n:02d} — {lec['title']}")
    add("")
    add(f"*{lec['tagline']}*")
    add("")
    add(f"{PARTS[lec['part']]} · {COURSE['code']} {COURSE['title']}")
    add("")
    add("## Contents of this folder")
    add("")
    add("| Path | What it is |")
    add("|---|---|")
    add(f"| `slides/Lecture_{n:02d}_{slug(lec['title'])}.pptx` | "
        f"Lecture deck, ready to present |")
    add(f"| `notebooks/{nb}` | Hands-on lab, runs end to end on CPU |")
    add(f"| `tasks/Lecture_{n:02d}_Tasks.md` | {core} core + {stretch} stretch tasks for students |")
    add("| `data/README.md` | Which datasets this lecture uses and how to load them |")
    if lec.get("app"):
        add("| `app/` | FastAPI service built during this lecture |")
    add("")
    add("## Learning objectives")
    add("")
    for o in lec["objectives"]:
        add(f"- {o}")
    add("")
    add("## Lecture outline")
    add("")
    for i, (section, bullets) in enumerate(lec["outline"], start=1):
        add(f"{i}. **{section}** — {bullets[0]}")
    add("")
    add("## The lab")
    add("")
    add(lec["lab"])
    add("")
    add(f"**Data:** `{lec['dataset']}`")
    add("")
    add("## Teaching notes")
    add("")
    add(f"- Suggested timing: 2 hours lecture (sections 1–{len(lec['outline'])}), 1 hour supervised lab.")
    add("- The notebook is written to be run live; each section maps to a slide section.")
    add("- Cells that take longer than ~60 s on a laptop are marked in the notebook.")
    add("- Every figure in the notebook can be dropped straight into the deck if you want to extend it.")
    add("")
    add("## Reading")
    add("")
    for r in lec["reading"]:
        add(f"- {r}")
    add("")
    return "\n".join(out)


DATA_NOTES = {
    "shapes_32.npz": "6000/1000/1000 train/val/test 32x32 RGB images, 4 classes "
                     "(circle, square, triangle, star). `load_shapes()`",
    "shapes_imagefolder/": "The same data as PNG files in class subfolders, for practising "
                           "the file -> tensor path.",
    "digits_8x8.npz": "scikit-learn's handwritten digits, 1347 train / 450 test, 10 classes. "
                      "`load_digits_npz()`",
    "shapes_det.npz": "1500/300 96x96 scenes containing 1-3 shapes. `load_detection()`",
    "shapes_det_annotations.json": "Bounding boxes and labels for the detection scenes.",
    "shapes_seg.npz": "The detection scenes with per-pixel class masks (0 = background). "
                      "`load_segmentation()`",
    "shapes_pairs.npz": "4000 unlabelled 32x32 images for self-supervised pre-training. "
                        "`load_ssl_pool()`",
    "sentiment.csv": "3000 labelled sentences, binary sentiment. `load_sentiment()`",
    "corpus.txt": "~250k characters of text for character-level language modelling. `load_corpus()`",
    "captions.csv": "2000 image-caption pairs aligned with the shapes train split. `load_captions()`",
}


def data_readme(lec) -> str:
    names = [d.strip() for d in lec["dataset"].split(",")]
    out = [f"# Data for Lecture {lec['number']:02d}", "",
           "All datasets live in the repository-level `data/` folder and are generated offline by",
           "`python tools/build_data.py`. Nothing is downloaded and no internet connection is needed.",
           "", "## Used in this lecture", ""]
    for nm in names:
        note = DATA_NOTES.get(nm, "See `dlcourse/data.py`.")
        out.append(f"- **`{nm}`** — {note}")
    out += ["", "## Loading", "",
            "```python",
            "import sys, pathlib",
            "sys.path.append(str(pathlib.Path.cwd().parents[2]))   # repository root",
            "from dlcourse import load_shapes",
            "",
            "d = load_shapes()",
            "X_train, y_train = d['train']      # (6000, 3, 32, 32) float in [0,1], (6000,) int64",
            "print(d['classes'])                # ['circle', 'square', 'triangle', 'star']",
            "```",
            "",
            "## Regenerating",
            "",
            "The generator is seeded, so re-running it reproduces byte-identical files:",
            "",
            "```bash",
            "python tools/build_data.py",
            "```",
            ""]
    return "\n".join(out)


def main():
    for lec in LECTURES:
        n = lec["number"]
        base = ROOT / "lectures" / folder_name(n)
        (base / "tasks").mkdir(parents=True, exist_ok=True)
        (base / "data").mkdir(parents=True, exist_ok=True)
        (base / "tasks" / f"Lecture_{n:02d}_Tasks.md").write_text(task_sheet(lec), encoding="utf-8")
        (base / "README.md").write_text(lecture_readme(lec), encoding="utf-8")
        (base / "data" / "README.md").write_text(data_readme(lec), encoding="utf-8")
        core = sum(1 for t in lec["tasks"] if t[0] == "core")
        stretch = len(lec["tasks"]) - core
        print(f"  L{n:02d}  tasks ({core} core + {stretch} stretch), README, data notes")
    print("Task sheets and READMEs written.")


if __name__ == "__main__":
    main()
