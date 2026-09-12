"""Assemble every lecture notebook, and optionally execute it to prove it runs.

    python tools/build_notebooks.py                 # build all
    python tools/build_notebooks.py 5 6             # build lectures 5 and 6
    python tools/build_notebooks.py --run 5         # build and execute lecture 5
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lectures"))

from course_spec import LECTURES, folder_name, slug  # noqa: E402
import nbtools  # noqa: E402

ROOT = HERE.parent


def notebook_path(lec) -> Path:
    n = lec["number"]
    return (ROOT / "lectures" / folder_name(n) / "notebooks" /
            f"L{n:02d}_{slug(lec['title'])[:44]}.ipynb")


def build_one(lec) -> Path:
    mod = importlib.import_module(f"l{lec['number']:02d}")
    importlib.reload(mod)
    nb = nbtools.build(lec, mod.cells())
    path = notebook_path(lec)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return path


def execute(path: Path, timeout: int = 2400) -> tuple[bool, str, float]:
    """Run the notebook and verify it completes.

    Shipped notebooks stay output-free so students run them fresh; the executed
    copy is written to ``build/executed/`` so the outputs can be inspected.
    """
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    t0 = time.time()
    try:
        client.execute()
        ok, msg = True, ""
    except CellExecutionError as e:
        # Surface the actual error, not the whole traceback.
        lines = [l for l in str(e).splitlines() if l.strip()]
        ok, msg = False, "\n".join(lines[-6:])
    except Exception as e:                                  # kernel problems etc.
        ok, msg = False, f"{type(e).__name__}: {e}"

    out_dir = ROOT / "build" / "executed"
    out_dir.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_dir / path.name)
    return ok, msg, time.time() - t0


def main(argv):
    run = "--run" in argv
    argv = [a for a in argv if a != "--run"]
    wanted = [int(a) for a in argv] if argv else list(range(1, 17))

    failures = []
    for n in wanted:
        lec = LECTURES[n - 1]
        try:
            path = build_one(lec)
        except ModuleNotFoundError:
            print(f"  L{n:02d}  (no notebook module yet — skipped)")
            continue
        nb = nbf.read(path, as_version=4)
        n_code = sum(1 for c in nb.cells if c.cell_type == "code")
        print(f"  L{n:02d}  {len(nb.cells):3d} cells ({n_code} code)  {path.name}")

        if run:
            ok, msg, secs = execute(path)
            if ok:
                print(f"        executed OK in {secs:.0f}s")
            else:
                print(f"        FAILED after {secs:.0f}s\n{msg}")
                failures.append((n, msg))

    if failures:
        print(f"\n{len(failures)} notebook(s) failed: {[n for n, _ in failures]}")
        return 1
    print("\nAll requested notebooks built." + (" All executed cleanly." if run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
