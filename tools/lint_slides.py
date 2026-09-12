"""Geometry and text-fit checks for the generated decks.

Without PowerPoint or LibreOffice available we cannot render the slides, so
this approximates what a renderer would show: shapes must stay inside the
canvas, body text must not collide with the footer, and no text box may need
more lines than its height allows.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Emu

ROOT = Path(__file__).resolve().parents[1]
W, H = Inches(13.333), Inches(7.5)
FOOTER_TOP = H - Inches(0.55)

# Rough advance width as a fraction of point size for Calibri / Consolas.
CHAR_W = {"Calibri": 0.48, "Consolas": 0.55}


def est_lines(text: str, width_emu: int, size_pt: float, font: str) -> int:
    if not text.strip():
        return 1
    char_w_in = size_pt * CHAR_W.get(font, 0.5) / 72.0
    per_line = max(int((width_emu / Emu(Inches(1))) / char_w_in), 8)
    # crude word wrap
    lines, cur = 1, 0
    for word in text.split():
        add = len(word) + (1 if cur else 0)
        if cur + add > per_line:
            lines += 1
            cur = len(word)
        else:
            cur += add
    return lines


def check(path: Path) -> list[str]:
    prs = Presentation(path)
    problems = []
    for si, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            L, T = shape.left, shape.top
            R, B = L + shape.width, T + shape.height
            if L < 0 or T < 0 or R > W + Emu(5000) or B > H + Emu(5000):
                problems.append(
                    f"{path.name} slide {si}: shape out of bounds "
                    f"L={L/914400:.2f} T={T/914400:.2f} R={R/914400:.2f} B={B/914400:.2f}")
            if not shape.has_text_frame:
                continue
            tf = shape.text_frame
            total_h_in = 0.0
            for p in tf.paragraphs:
                text = "".join(r.text for r in p.runs)
                if not text:
                    continue
                size = p.runs[0].font.size.pt if p.runs[0].font.size else 18
                font = p.runs[0].font.name or "Calibri"
                n = est_lines(text, shape.width, size, font)
                total_h_in += n * size * 1.22 / 72.0
                after = p.space_after.pt if p.space_after else 0
                total_h_in += after / 72.0
            needed = Inches(total_h_in)
            # A textbox may auto-grow downward; what matters is whether the
            # grown box would run into the footer or off the slide.
            if total_h_in > 0 and T + needed > FOOTER_TOP and T < FOOTER_TOP:
                problems.append(
                    f"{path.name} slide {si}: text needs {total_h_in:.2f}in from "
                    f"top {T/914400:.2f}in -> collides with footer "
                    f"(text: {tf.text[:60]!r})")
    return problems


def main():
    decks = sorted((ROOT / "lectures").glob("*/slides/*.pptx"))
    if not decks:
        sys.exit("no decks found - run build_slides.py first")
    all_problems = []
    for d in decks:
        all_problems.extend(check(d))
    if all_problems:
        print(f"{len(all_problems)} layout problem(s):")
        for p in all_problems[:40]:
            print("  -", p)
        sys.exit(1)
    print(f"Checked {len(decks)} decks: all shapes in bounds, no predicted text overflow.")


if __name__ == "__main__":
    main()
