# Data for Lecture 02

Course datasets live in the repository-level `data/` folder and are generated
offline by `python tools/build_data.py`. A few lectures also use datasets that
ship inside scikit-learn. Either way nothing is downloaded and no internet
connection is needed.

## Used in this lecture

- **`digits_8x8.npz`** — scikit-learn's handwritten digits, 1347 train / 450 test, 10 classes. `load_digits_npz()`
- **`shapes_32.npz`** — 6000/1000/1000 train/val/test 32x32 RGB images, 4 classes (circle, square, triangle, star). `load_shapes()`

## Loading

```python
import sys, pathlib
sys.path.append(str(pathlib.Path.cwd().parents[2]))   # repository root
from dlcourse import load_shapes

d = load_shapes()
X_train, y_train = d['train']      # (6000, 3, 32, 32) float in [0,1], (6000,) int64
print(d['classes'])                # ['circle', 'square', 'triangle', 'star']
```

## Regenerating

The generator is seeded, so re-running it reproduces byte-identical files:

```bash
python tools/build_data.py
```
