# Lecture 01 — Tasks
## Deep Learning Foundations and the Image Data Pipeline

> Why depth works, and how pixels become tensors.

**Course:** DL-601 Deep Learning · **Part:** Part I - Foundations  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 02

---

## What you should be able to do after this

- Place deep learning relative to classical machine learning and explain what changed after 2012.
- Describe an image as a tensor and move fluently between NumPy, PIL and PyTorch layouts.
- Build a Dataset and DataLoader and reason about batching, shuffling and normalisation.
- Establish a baseline and a train/validation/test protocol before touching a model.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_01/notebooks/L01_Deep_Learning_Foundations_and_the_Image_Data.ipynb
```

**Data used:** `shapes_32.npz, shapes_imagefolder/`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (4 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Tensor layout drill

Write round_trip(path) that loads a PNG with PIL, converts it to a normalised NCHW float tensor, converts it back, and asserts the recovered uint8 array matches the original within 1/255. State in one sentence where the rounding error comes from.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Custom Dataset

Implement ShapesFolder(Dataset) that reads shapes_imagefolder/train without using ImageFolder. It must discover class names from the directory listing, sort them for a stable label mapping, and return (CHW float tensor, int label).

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Honest normalisation

Compute per-channel mean and std on the training split only. Then compute them on train+val+test combined. Report both to four decimal places and explain why only the first is admissible.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Baselines that matter

Report majority-class accuracy and multinomial logistic-regression accuracy on the test split. Record both in a results table - every later lecture compares against these numbers.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Batch-size timing study

Time one epoch of the linear baseline at batch sizes 1, 16, 128 and 1024. Plot seconds per epoch against batch size and explain the shape of the curve in terms of Python overhead versus vectorisation.

### Stretch 2 — Find the leak

leaky_split.py builds a train/test split with a deliberate flaw. Find it, quantify how much it inflates the reported accuracy, and fix it.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across all 16 lectures).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L01.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 1 and Section 5.1-5.3.
- Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep CNNs* (AlexNet).
- PyTorch docs: Datasets and DataLoaders tutorial.

---

**Next lecture —** 02: From Linear Models to Neural Networks. The softmax classifier, and the exact point where linearity fails.
