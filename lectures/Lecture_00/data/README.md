# Data for Lecture 00

Course datasets live in the repository-level `data/` folder and are generated
offline by `python tools/build_data.py`. A few lectures also use datasets that
ship inside scikit-learn. Either way nothing is downloaded and no internet
connection is needed.

## Used in this lecture

- **`scikit-learn iris (bundled with the library)`** — 150 flowers, 4 measurements, 3 species. Ships inside scikit-learn, so nothing to download. `from sklearn.datasets import load_iris`
- **`digits_8x8.npz`** — scikit-learn's handwritten digits, 1347 train / 450 test, 10 classes. `load_digits_npz()`

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
