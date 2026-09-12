"""Check that every lecture folder actually contains what the syllabus promises.

    python tools/verify_course.py

Fails loudly rather than quietly shipping a lecture with a missing deck.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course_spec import COURSE, LECTURES, folder_name, slug  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DATA = [
    "shapes_32.npz", "shapes_imagefolder", "digits_8x8.npz", "shapes_det.npz",
    "shapes_det_annotations.json", "shapes_seg.npz", "shapes_pairs.npz",
    "sentiment.csv", "corpus.txt", "captions.csv",
]

problems: list[str] = []
stats = {"slides": 0, "cells": 0, "code_cells": 0, "todos": 0, "tasks": 0}


def check(condition, message):
    if not condition:
        problems.append(message)
    return condition


def check_root():
    for name in ["README.md", "requirements.txt", "Deep_Learning_Course_Module.pdf",
                 "dlcourse/__init__.py", "dlcourse/data.py", "dlcourse/training.py",
                 "dlcourse/viz.py"]:
        check((ROOT / name).exists(), f"missing root file: {name}")
    for name in REQUIRED_DATA:
        check((ROOT / "data" / name).exists(), f"missing dataset: data/{name}")


def check_lecture(lec):
    n = lec["number"]
    base = ROOT / "lectures" / folder_name(n)
    tag = f"L{n:02d}"

    if not check(base.exists(), f"{tag}: folder missing"):
        return

    check((base / "README.md").exists(), f"{tag}: README.md missing")
    check((base / "data" / "README.md").exists(), f"{tag}: data/README.md missing")

    # slides
    deck = base / "slides" / f"Lecture_{n:02d}_{slug(lec['title'])}.pptx"
    if check(deck.exists(), f"{tag}: slide deck missing ({deck.name})"):
        from pptx import Presentation
        prs = Presentation(deck)
        count = len(prs.slides._sldIdLst)
        stats["slides"] += count
        check(count >= 12, f"{tag}: only {count} slides, expected >= 12")

    # notebook
    nb_path = base / "notebooks" / f"L{n:02d}_{slug(lec['title'])[:44]}.ipynb"
    if check(nb_path.exists(), f"{tag}: notebook missing ({nb_path.name})"):
        nb = nbf.read(nb_path, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        todos = [c for c in nb.cells
                 if c.cell_type == "markdown" and "TODO — Task" in c.source]
        stats["cells"] += len(nb.cells)
        stats["code_cells"] += len(code_cells)
        stats["todos"] += len(todos)

        check(len(code_cells) >= 12, f"{tag}: only {len(code_cells)} code cells")
        check(len(todos) >= 4, f"{tag}: only {len(todos)} TODO task cells")

        # Every code cell must at least parse. Executing the notebooks is the real
        # test, but this catches indentation and escaping mistakes in seconds.
        import ast
        for ci, cell in enumerate(code_cells):
            try:
                ast.parse(cell.source)
            except SyntaxError as e:
                problems.append(f"{tag}: code cell {ci} has a syntax error "
                                f"(line {e.lineno}): {e.msg}")

        # Each core task in the spec needs a matching TODO cell in the notebook.
        core_tasks = [t for t in lec["tasks"] if t[0] == "core"]
        for ti, (_, title, _d) in enumerate(core_tasks, start=1):
            marker = f"TODO — Task {ti}: {title}"
            check(any(marker in c.source for c in nb.cells if c.cell_type == "markdown"),
                  f"{tag}: no TODO cell in the notebook for core task {ti} ({title})")
        check(any("from dlcourse import" in c.source for c in code_cells),
              f"{tag}: notebook does not import dlcourse")
        # Shipped notebooks must be output-free so students run them fresh.
        with_output = [c for c in code_cells if c.get("outputs")]
        check(not with_output,
              f"{tag}: {len(with_output)} code cells carry saved output")

    # tasks
    tasks = base / "tasks" / f"Lecture_{n:02d}_Tasks.md"
    if check(tasks.exists(), f"{tag}: task sheet missing"):
        text = tasks.read_text(encoding="utf-8")
        core = sum(1 for t in lec["tasks"] if t[0] == "core")
        stats["tasks"] += len(lec["tasks"])
        check(f"## Core tasks ({core} required)" in text,
              f"{tag}: task sheet core count does not match the spec")
        for marker in ["Submission checklist", "How this is marked", "Reading"]:
            check(marker in text, f"{tag}: task sheet missing '{marker}' section")

    # apps, where the spec promises one
    if lec.get("app"):
        app = base / "app"
        check((app / "main.py").exists(), f"{tag}: app/main.py missing")
        check(any(app.glob("tests/test_*.py")), f"{tag}: app has no tests")
        check((app / "requirements.txt").exists(), f"{tag}: app/requirements.txt missing")


def main():
    check_root()
    for lec in LECTURES:
        check_lecture(lec)

    print(f"{COURSE['code']} — course verification\n" + "=" * 46)
    print(f"lectures        : {len(LECTURES)}")
    print(f"slides total    : {stats['slides']}")
    print(f"notebook cells  : {stats['cells']} ({stats['code_cells']} code)")
    print(f"TODO task cells : {stats['todos']}")
    print(f"tasks in spec   : {stats['tasks']}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1
    print("\nAll lectures complete. Every promised artefact is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
