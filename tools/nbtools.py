"""Helpers for assembling notebooks from Python source.

Writing notebooks as ``.py`` modules keeps them reviewable in git and lets the
build step execute every one before it ships, so a broken cell never reaches a
student.
"""
from __future__ import annotations

import textwrap

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(src).strip("\n"))


def header(lec) -> list:
    """Title block + the import cell every notebook starts with."""
    n = lec["number"]
    return [
        md(f"""
        # Lecture {n:02d} — {lec['title']}

        > *{lec['tagline']}*

        **What this notebook does**

        {lec['lab']}

        **Data:** `{lec['dataset']}`

        ---

        ### How to use this notebook

        **This notebook is your workbook and your submission.** Everything you need is
        here — you do not need to open any other file.

        1. Run the cells in order, top to bottom. The sections match the lecture slides.
        2. Worked cells are there to be **read**, not skimmed. They build the ideas the
           tasks assume.
        3. Cells marked **📝 TODO** are yours. Write your code in the empty cell
           underneath, and your written answer in a markdown cell after that.
        4. When you are done, restart the kernel and run everything once more to check it
           works from clean.

        Nothing here needs a GPU. If a cell is slow on your machine, reduce the epoch
        count at the top of that section and say so in your write-up.

        *(A printable copy of the tasks and the marking rubric is in
        `../tasks/Lecture_{n:02d}_Tasks.md`, but the work itself belongs here.)*
        """),
        code(f"""
        # --- setup: make the repository root importable -------------------------
        import os, sys, pathlib

        ROOT = pathlib.Path.cwd()
        while not (ROOT / "dlcourse").exists() and ROOT != ROOT.parent:
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT))

        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import matplotlib.pyplot as plt

        from dlcourse import (load_shapes, load_digits_npz, load_detection,
                              load_segmentation, load_ssl_pool, load_sentiment,
                              load_corpus, load_captions,
                              train, evaluate, set_seed, count_parameters,
                              show_grid, plot_history, plot_confusion,
                              show_boxes, show_masks)

        set_seed(0)
        # Leave a core or two for the rest of the machine. More threads than cores
        # makes training slower, not faster.
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) - 1)))
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("repo root :", ROOT)
        print("torch     :", torch.__version__)
        print("device    :", DEVICE)
        """),
    ]


def todo(number: str, title: str, body: str) -> nbf.NotebookNode:
    """A task cell: the complete brief, so the notebook stands on its own."""
    body = textwrap.dedent(body).strip()
    indented = "\n    ".join(body.splitlines())
    return md(f"""
    ---

    ## 📝 TODO — Task {number}: {title}

    {indented}

    **Deliverable:** working code in the cell below, plus the number, table or plot the
    task asks for. Where you are asked to explain something, add a markdown cell
    underneath and answer in two or three sentences.
    """)


def todo_cell(hint: str = "") -> nbf.NotebookNode:
    return code(f"""
    # ---- YOUR SOLUTION ----
    {hint}
    """)


def section(title: str, blurb: str = "") -> nbf.NotebookNode:
    body = f"## {title}"
    if blurb:
        body += "\n\n" + textwrap.dedent(blurb).strip()
    return md(body)


def footer(lec) -> list:
    n = lec["number"]
    core = [t for t in lec["tasks"] if t[0] == "core"]
    stretch = [t for t in lec["tasks"] if t[0] == "stretch"]
    core_lines = "\n        ".join(
        f"- [ ] **Task {i}** — {t[1]}" for i, t in enumerate(core, start=1))
    stretch_lines = "\n        ".join(
        f"- [ ] *Stretch {i}* — {t[1]}" for i, t in enumerate(stretch, start=1))
    return [
        md(f"""
        ---

        ## Checklist before you submit

        All {len(core)} core tasks are required. Each is marked **📝 TODO** above, in the
        section it belongs to.

        {core_lines}

        Optional, not marked:

        {stretch_lines}

        ### Final checks

        - [ ] Restart the kernel and **Run All**. It completes with no errors.
        - [ ] `set_seed(0)` runs before anything random.
        - [ ] Every plot has axis labels and a title.
        - [ ] Written answers are in markdown cells, not in code comments.
        - [ ] This lecture's headline numbers are in your running `results.md` table.
        - [ ] AI-assistant use is disclosed in the cell below.

        Submit this notebook as `LASTNAME_FIRSTNAME_L{n:02d}.ipynb`.
        """),
        md("""
        ### AI assistance disclosure

        *Replace this text: state which tools you used and for what. "None" is a valid answer.*
        """),
    ]


def build(lec, cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = header(lec) + cells + footer(lec)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "title": f"Lecture {lec['number']:02d} — {lec['title']}",
    }
    return nb
