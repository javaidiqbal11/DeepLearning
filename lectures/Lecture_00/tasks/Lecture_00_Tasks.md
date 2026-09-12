# Lecture 00 — Tasks
## Machine Learning Foundations

> What a model is, how it learns, and how you know it worked.

**Course:** DL-601 Deep Learning · **Part:** Part 0 - Machine Learning Foundations  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 01

---

## What you should be able to do after this

- Define machine learning, and say when it is the right tool and when it is not.
- Explain features and labels, and the difference between supervised and unsupervised learning.
- Split data correctly, and explain why the test set is touched exactly once.
- Recognise underfitting and overfitting, and describe the bias-variance trade-off.
- Train and evaluate a classifier with scikit-learn, and read its confusion matrix.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_00/notebooks/L00_Machine_Learning_Foundations.ipynb
```

**Data used:** `scikit-learn iris (bundled with the library), digits_8x8.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (7 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Explore a dataset

Load the iris dataset. Report the number of samples, the number of features, the class names and the count per class. Plot two features against each other coloured by class, and say in one sentence which pair separates the classes best.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Split the data honestly

Split into 60% train, 20% validation and 20% test using stratification. Verify the class proportions are preserved in all three splits, and explain in one sentence why stratification matters on a dataset this small.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Train three classifiers

Train k-nearest neighbours (k=5), a decision tree and logistic regression on the training split. Report training and validation accuracy for each in a small table.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Beat the baseline

Compute the majority-class accuracy. State by how much each of your three models beats it. A model that does not beat this baseline has learned nothing.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Make a model overfit on purpose

Train decision trees with max_depth from 1 to 15. Plot training and validation accuracy against depth on one figure. Mark the depth where overfitting begins and explain how you identified it.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Read a confusion matrix

Produce a confusion matrix for your best model on the test split. Name the two classes it confuses most, and say why that is unsurprising given your plot from Task 1.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 7 — Where hand-designed features run out

Train logistic regression on the raw pixels of digits_8x8 and report test accuracy. Then explain in three sentences why the same approach would break down on a 224x224 colour photograph.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Cross-validation

Replace the single validation split with 5-fold cross-validation. Report the mean and standard deviation of accuracy, and explain what the standard deviation tells you that a single split cannot.

### Stretch 2 — Feature scaling

Train k-nearest neighbours with and without StandardScaler. Report both accuracies and explain why distance-based methods care about feature scale while decision trees do not.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L00.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Geron, *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, Chapters 1-2.
- scikit-learn documentation: *An Introduction to Machine Learning with scikit-learn*.
- Goodfellow, Bengio & Courville, *Deep Learning*, Sections 5.1-5.5.

---

**Next lecture —** 01: Deep Learning Foundations and the Image Data Pipeline. Why depth works, and how pixels become tensors.
