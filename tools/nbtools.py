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

        ### How to work through this

        Run the cells in order. Sections match the lecture slides. Cells marked
        **`TODO`** are the ones you complete for the task sheet in
        `../tasks/Lecture_{n:02d}_Tasks.md` — everything else is worked for you and is
        there to be read, not skimmed.

        Nothing here needs a GPU. If a cell is slow on your machine, reduce `EPOCHS`
        at the top of the section and say so in your write-up.
        """),
        code(f"""
        # --- setup: make the repository root importable -------------------------
        import sys, pathlib

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
        torch.set_num_threads(4)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("repo root :", ROOT)
        print("torch     :", torch.__version__)
        print("device    :", DEVICE)
        """),
    ]


def todo(number: str, title: str, body: str) -> nbf.NotebookNode:
    """A task cell: states what the student must do and where it goes."""
    return md(f"""
    ---
    ### 📝 TODO — Task {number}: {title}

    {textwrap.dedent(body).strip()}

    *Write your solution in the cell below, and your explanation in a markdown cell after it.*
    """)


def todo_cell(hint: str = "") -> nbf.NotebookNode:
    return code(f"""
    # Your solution here.
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
    lines = "\n".join(f"{i}. **{t[1]}** — {t[2]}" for i, t in enumerate(core, start=1))
    return [
        md(f"""
        ---

        ## Your tasks

        The full brief, including the stretch tasks and the marking rubric, is in
        `../tasks/Lecture_{n:02d}_Tasks.md`. The core tasks are:

        {lines}

        ### Before you submit

        - Restart the kernel and run everything top to bottom. It must complete with no errors.
        - Check `set_seed(0)` runs before anything random.
        - Every plot needs axis labels and a title.
        - Add this lecture's headline numbers to your running `results.md` table.
        - Disclose any AI-assistant use in the cell below.
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
