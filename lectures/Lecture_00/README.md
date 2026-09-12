# Lecture 00 — Machine Learning Foundations

*What a model is, how it learns, and how you know it worked.*

Part 0 - Machine Learning Foundations · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_00_Machine_Learning_Foundations.pptx` | Lecture deck, ready to present |
| `notebooks/L00_Machine_Learning_Foundations.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_00_Tasks.md` | 7 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Define machine learning, and say when it is the right tool and when it is not.
- Explain features and labels, and the difference between supervised and unsupervised learning.
- Split data correctly, and explain why the test set is touched exactly once.
- Recognise underfitting and overfitting, and describe the bias-variance trade-off.
- Train and evaluate a classifier with scikit-learn, and read its confusion matrix.

## Lecture outline

1. **What machine learning is** — Traditional programming: you write the rules, the computer applies them.
2. **The three kinds of learning** — Supervised: learn from labelled examples. Classification asks which class, regression asks how much.
3. **Features, labels and the dataset** — A dataset is a table: one row per example, one column per feature, plus a label column.
4. **How a model actually learns** — A model is a function with adjustable numbers inside it, called parameters.
5. **Train, validation and test** — Train fits the parameters. Validation selects hyper-parameters. Test is touched once.
6. **Underfitting, overfitting and the trade-off** — Underfitting: the model is too simple. Both training and test scores are poor.
7. **Measuring success honestly** — Accuracy is the fraction correct, and it is useless when the classes are imbalanced.
8. **From classical ML to deep learning** — Classical models do well on tabular data with good hand-designed features.

## The lab

Walk the complete machine learning workflow on two small real datasets: load and explore the data, split it honestly, train k-nearest neighbours, a decision tree and logistic regression, compare them against a baseline, watch overfitting appear as model complexity grows, read a confusion matrix, and finish by seeing where hand-designed features run out on raw images - which is exactly why the rest of the course exists.

**Data:** `scikit-learn iris (bundled with the library), digits_8x8.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–8), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Geron, *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, Chapters 1-2.
- scikit-learn documentation: *An Introduction to Machine Learning with scikit-learn*.
- Goodfellow, Bengio & Courville, *Deep Learning*, Sections 5.1-5.5.
