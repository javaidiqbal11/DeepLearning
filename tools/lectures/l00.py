"""Lecture 00 notebook — machine learning foundations, before any deep learning."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        md("""
        ## Start here

        This lecture assumes **nothing**. If you have never trained a model before, this is
        the right place to begin. By the end you will have trained three real classifiers,
        measured them honestly, and seen with your own eyes why deep learning had to be
        invented.

        We use small, real datasets that ship with scikit-learn, so everything runs in
        seconds.
        """),
        section("1. What machine learning actually is", """
        **Traditional programming:** you write the rules, the computer applies them.

        **Machine learning:** you supply examples, and the computer works out the rules.

        Start with a problem where you *can* write the rule, so the contrast is clear.
        """),
        code("""
        # Traditional programming: the rule is written by hand.
        def is_even_by_rule(n):
            return n % 2 == 0

        print("rule-based:", [is_even_by_rule(n) for n in range(6)])

        # For "is this number even?" the rule is obvious, so machine learning would be
        # a silly choice. Machine learning earns its keep when a rule genuinely exists
        # but is far too complicated to write down -- "is there a cat in this photo?".
        """),
        md("""
        ### When to use machine learning

        | Use it when | Do not use it when |
        |---|---|
        | The rule is real but too complex to write | A simple rule already works |
        | You have plenty of labelled examples | You have almost no data |
        | Being occasionally wrong is acceptable | Every answer must be provably correct |
        | The pattern is stable over time | The rules change constantly |

        ### The three kinds of learning

        - **Supervised** — learn from labelled examples. *Classification*: which class?
          *Regression*: how much?
        - **Unsupervised** — find structure in unlabelled data (clustering, compression).
        - **Reinforcement** — learn from rewards by acting in an environment.

        This course is almost entirely **supervised learning**, because that is where deep
        learning works best.
        """),
        section("2. A dataset is a table", """
        Every supervised dataset has the same shape: a matrix `X` of features, and a vector
        `y` of labels.

        The iris dataset is the classic first example — 150 flowers, four measurements
        each, three species.
        """),
        code("""
        from sklearn.datasets import load_iris

        iris = load_iris()
        X, y = iris.data, iris.target
        # Plain Python strings: NumPy string scalars print as np.str_('setosa').
        FEATURES = [str(f) for f in iris.feature_names]
        CLASSES = [str(c) for c in iris.target_names]

        print("X (features):", X.shape, " <- (n_samples, n_features)")
        print("y (labels)  :", y.shape, " <- one label per sample")
        print("\\nfeature names:", FEATURES)
        print("class names  :", CLASSES)
        print("\\nfirst three rows of X:")
        print(X[:3])
        print("first three labels:", y[:3], "->", [CLASSES[i] for i in y[:3]])
        """),
        code("""
        import pandas as pd

        df = pd.DataFrame(X, columns=FEATURES)
        df["species"] = [CLASSES[i] for i in y]
        print(df.head())
        print("\\nsamples per class:")
        print(df["species"].value_counts())
        """),
        code("""
        # Plot two features. If the classes separate by eye, a simple model will do well.
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        colours = ["#1f6fb2", "#1b866b", "#c16a1c"]

        for c, name in enumerate(CLASSES):
            m = y == c
            a1.scatter(X[m, 0], X[m, 1], c=colours[c], label=name, s=28, edgecolors="none")
            a2.scatter(X[m, 2], X[m, 3], c=colours[c], label=name, s=28, edgecolors="none")

        a1.set_xlabel(FEATURES[0]); a1.set_ylabel(FEATURES[1])
        a1.set_title("Sepal measurements — classes overlap")
        a2.set_xlabel(FEATURES[2]); a2.set_ylabel(FEATURES[3])
        a2.set_title("Petal measurements — classes separate cleanly")
        for a in (a1, a2):
            a.legend(); a.grid(alpha=.3)
        plt.tight_layout(); plt.show()

        print("Petal length and width almost separate the three species on their own.")
        print("Choosing good features is most of classical machine learning.")
        """),
        section("3. Splitting the data", """
        This is the single most important habit in the whole field.

        - **Training set** — the model fits its parameters on this.
        - **Validation set** — you compare models and settings using this.
        - **Test set** — touched **exactly once**, at the very end, for an honest number.

        Measuring on data the model trained on tells you about memorisation, not learning.
        """),
        code("""
        from sklearn.model_selection import train_test_split

        # Two splits give three sets. `stratify` keeps the class proportions in each.
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=0, stratify=y_temp)

        print(f"train      {len(X_train):>4}  ({len(X_train)/len(X):.0%})")
        print(f"validation {len(X_val):>4}  ({len(X_val)/len(X):.0%})")
        print(f"test       {len(X_test):>4}  ({len(X_test)/len(X):.0%})")

        print("\\nclass counts per split (stratify keeps these balanced):")
        for name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]:
            print(f"  {name:<10} {np.bincount(labels)}")
        """),
        md("""
        ### Data leakage

        Leakage is any information crossing from the test set into training. It makes your
        results look wonderful and your deployed model look broken.

        Common ways it happens:

        - Fitting a scaler on all the data before splitting
        - Duplicate or near-duplicate rows landing in both splits
        - Choosing hyper-parameters by looking at the test score
        - A feature that secretly encodes the answer

        The rule: **split first, then do everything else.**
        """),
        section("4. Your first model", """
        k-nearest neighbours is the simplest classifier worth knowing. It does not really
        "train" at all: it memorises the training set, and to classify a new point it looks
        at the `k` closest training points and takes a vote.
        """),
        code("""
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)             # "training" = storing the data

        train_acc = knn.score(X_train, y_train)
        val_acc = knn.score(X_val, y_val)

        print(f"training accuracy   : {train_acc:.3f}")
        print(f"validation accuracy : {val_acc:.3f}")

        # Look at one prediction in detail.
        sample = X_val[0]
        pred = knn.predict([sample])[0]
        print(f"\\nfirst validation flower: {sample}")
        print(f"  predicted: {CLASSES[pred]}")
        print(f"  actually : {CLASSES[y_val[0]]}")
        """),
        section("5. Always compare against a baseline", """
        A model that beats chance is not evidence of anything. Before celebrating any
        accuracy number, work out what a trivial approach would score.
        """),
        code("""
        from sklearn.dummy import DummyClassifier

        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        baseline = dummy.score(X_val, y_val)

        print(f"always predict the most common class : {baseline:.3f}")
        print(f"random guessing (3 classes)          : {1/3:.3f}")
        print(f"k-nearest neighbours                 : {val_acc:.3f}")
        print(f"\\nkNN beats the baseline by {val_acc - baseline:.3f}. That is the number")
        print("that actually means something.")
        """),
        section("6. Three models, compared fairly", """
        Same data, same splits. Only the algorithm changes.
        """),
        code("""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            "k-nearest neighbours": KNeighborsClassifier(n_neighbors=5),
            "decision tree":        DecisionTreeClassifier(max_depth=3, random_state=0),
            "logistic regression":  LogisticRegression(max_iter=500),
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            results[name] = (model.score(X_train, y_train), model.score(X_val, y_val))

        print(f"{'model':<24}{'train':>8}{'val':>8}{'gap':>8}")
        print("-" * 48)
        print(f"{'baseline (majority)':<24}{dummy.score(X_train, y_train):>8.3f}"
              f"{baseline:>8.3f}{'':>8}")
        for name, (tr, va) in results.items():
            print(f"{name:<24}{tr:>8.3f}{va:>8.3f}{tr - va:>+8.3f}")
        """),
        code("""
        names = list(results)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        pos = np.arange(len(names))
        ax.barh(pos - 0.2, [results[n][0] for n in names], height=0.38,
                label="train", color="#bdc3c7")
        ax.barh(pos + 0.2, [results[n][1] for n in names], height=0.38,
                label="validation", color="#1f6fb2")
        ax.axvline(baseline, color="#c0392b", ls="--", lw=1.5, label="majority baseline")
        ax.set_yticks(pos, names); ax.set_xlim(0, 1.05)
        ax.set_xlabel("accuracy"); ax.set_title("Three classifiers on iris")
        ax.legend(loc="lower right", fontsize=9); ax.invert_yaxis()
        plt.tight_layout(); plt.show()
        """),
        section("7. Underfitting and overfitting", """
        The central tension of the whole subject.

        - **Underfitting** — the model is too simple. Both scores are poor.
        - **Overfitting** — the model memorises the training data. Training score is high,
          validation score is not.

        Iris is too easy to show this: four well-chosen features, three tidy classes, and
        almost any model gets it right. We need a harder problem.

        Switch to **handwritten digits** — 64 raw pixels, 10 classes — and turn one dial:
        the depth of a decision tree.
        """),
        code("""
        digits = load_digits_npz()
        Xd = digits["train"][0].reshape(len(digits["train"][0]), -1).numpy()
        yd = digits["train"][1].numpy()
        DIGIT_CLASSES = digits["classes"]

        Xd_tr, Xd_va, yd_tr, yd_va = train_test_split(
            Xd, yd, test_size=0.25, random_state=0, stratify=yd)

        print("digits train:", Xd_tr.shape, "  validation:", Xd_va.shape)
        show_grid(digits["train"][0][:16], yd[:16], DIGIT_CLASSES, n=16, ncols=8,
                  title="Handwritten digits — 8x8 pixels, 10 classes")
        plt.show()
        """),
        code("""
        depths = list(range(1, 21))
        train_scores, val_scores = [], []

        for d in depths:
            tree = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xd_tr, yd_tr)
            train_scores.append(tree.score(Xd_tr, yd_tr))
            val_scores.append(tree.score(Xd_va, yd_va))

        best_depth = depths[int(np.argmax(val_scores))]
        final_gap = train_scores[-1] - val_scores[-1]

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(depths, train_scores, marker="o", ms=4, lw=2,
                label="training accuracy", color="#7f8c8d")
        ax.plot(depths, val_scores, marker="s", ms=4, lw=2,
                label="validation accuracy", color="#1f6fb2")
        ax.fill_between(depths, val_scores, train_scores, alpha=.15, color="#c0392b")
        ax.axvline(best_depth, color="#27ae60", ls="--",
                   label=f"best validation @ depth {best_depth}")
        ax.set_xlabel("max_depth  (model complexity)"); ax.set_ylabel("accuracy")
        ax.set_xticks(depths[::2]); ax.set_ylim(0, 1.05)
        ax.set_title("Underfitting on the left, overfitting on the right")
        ax.annotate("underfitting\\nboth scores low", xy=(1.6, 0.28), fontsize=9,
                    color="#c0392b")
        ax.annotate("overfitting\\ntraining reaches 1.00,\\nvalidation stops improving",
                    xy=(12.5, 0.50), fontsize=9, color="#c0392b")
        ax.legend(loc="center right"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()

        print(f"{'depth':>6}{'train':>9}{'val':>9}{'gap':>9}")
        print("-" * 33)
        for d in sorted({1, 3, 5, best_depth, 15, 20}):
            i = depths.index(d)
            note = "  <- best" if d == best_depth else ""
            print(f"{d:>6}{train_scores[i]:>9.3f}{val_scores[i]:>9.3f}"
                  f"{train_scores[i] - val_scores[i]:>+9.3f}{note}")

        print(f"\\nAt depth 20 the tree is perfect on data it has seen ({train_scores[-1]:.3f})")
        print(f"and stuck at {val_scores[-1]:.3f} on data it has not. That {final_gap:.3f} gap")
        print("is overfitting, measured rather than asserted.")
        """),
        md("""
        ### Bias and variance

        - **Bias** — error from the model being too simple to represent the truth.
          High bias shows up as underfitting.
        - **Variance** — error from the model being too sensitive to the particular
          training sample it saw. High variance shows up as overfitting.

        Adding capacity lowers bias and raises variance. Finding the balance for *your*
        data is the job. Deep learning does not escape this — Lecture 4 is entirely about
        managing it.
        """),
        section("8. Measuring more carefully than accuracy", """
        Accuracy is one number and it hides a lot. The confusion matrix shows exactly which
        classes are being mixed up.
        """),
        code("""
        from sklearn.metrics import classification_report

        # Back to iris, and back to the rule: choose using validation, then touch
        # the test set exactly once.
        final_model = DecisionTreeClassifier(max_depth=3, random_state=0)
        final_model.fit(X_train, y_train)
        test_pred = final_model.predict(X_test)

        print(f"test accuracy: {final_model.score(X_test, y_test):.3f}\\n")
        print(classification_report(y_test, test_pred,
                                    target_names=CLASSES, digits=3))
        """),
        code("""
        plot_confusion(y_test, test_pred, CLASSES,
                       normalize=False, title="Confusion matrix — counts")
        plt.show()
        """),
        md("""
        ### Reading the report

        - **Precision** — of everything you labelled class C, what fraction really was C?
          (How much do you cry wolf?)
        - **Recall** — of everything that really was class C, what fraction did you find?
          (How much do you miss?)
        - **F1** — their harmonic mean, when you need one number.
        - **Support** — how many test examples of that class there were.

        Why accuracy alone is dangerous: if 99% of emails are not spam, a model that
        predicts "not spam" for everything scores **99% accuracy** and catches **zero**
        spam. Its recall on the class you care about is 0.
        """),
        section("9. Where classical machine learning runs out", """
        Everything above works well because the iris features were *chosen by a botanist*.
        Someone decided that petal length matters.

        Now try a problem with no hand-designed features at all: raw image pixels.
        """),
        code("""
        # The held-out digits test split, still as raw pixels. Nobody designed a
        # single one of these 64 features -- they are just brightness values.
        Xd_test = digits["test"][0].reshape(len(digits["test"][0]), -1).numpy()
        yd_test = digits["test"][1].numpy()

        print("each image is", tuple(digits["train"][0].shape[1:]),
              "-> flattened to", Xd_test.shape[1], "raw pixel values")
        print("train:", Xd.shape, "  test:", Xd_test.shape)
        """),
        code("""
        digit_models = {
            "majority baseline":    DummyClassifier(strategy="most_frequent"),
            "k-nearest neighbours": KNeighborsClassifier(n_neighbors=5),
            "logistic regression":  LogisticRegression(max_iter=2000),
        }

        print(f"{'model':<24}{'test accuracy':>15}")
        print("-" * 39)
        for name, model in digit_models.items():
            model.fit(Xd, yd)
            print(f"{name:<24}{model.score(Xd_test, yd_test):>15.3f}")
        """),
        md("""
        ### So why do we need deep learning at all?

        Classical models do perfectly well on 8x8 digits. The problem is what happens when
        the images get realistic:

        | | 8x8 digits | 224x224 colour photo |
        |---|---|---|
        | Pixels per image | 64 | 150,528 |
        | Weights in one 1000-unit layer | 64,000 | **150 million** |

        And size is not even the main issue. A classifier on raw pixels treats every pixel
        as an independent number. It has **no idea that neighbouring pixels are related**,
        and **no idea that a cat shifted three pixels to the left is still a cat** — it
        would have to learn that separately for every possible position.

        Deep learning fixes exactly this. Instead of you designing features, the network
        learns them — edges in the early layers, then textures, then object parts, then
        whole objects. Nobody programs that hierarchy; it emerges from the data.

        **That is what Lecture 1 onwards is about.**
        """),
        code("""
        # A preview of the gap you are about to close.
        stages = ["chance\\n(4 classes)", "logistic regression\\non raw pixels",
                  "small CNN\\n(Lecture 5)"]
        scores = [0.25, 0.63, 0.99]

        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        bars = ax.bar(stages, scores, color=["#bdc3c7", "#c16a1c", "#1b866b"])
        for b, s in zip(bars, scores):
            ax.text(b.get_x() + b.get_width()/2, s + 0.02, f"{s:.2f}",
                    ha="center", fontsize=11)
        ax.set_ylim(0, 1.15); ax.set_ylabel("accuracy")
        ax.set_title("The shape dataset you will meet in Lecture 1")
        plt.tight_layout(); plt.show()

        print("Classical ML on raw pixels: 0.63. A small CNN: 0.99.")
        print("Closing that gap, and understanding exactly why it closes, is the course.")
        """),
        md("""
        ---

        ## Summary

        | Idea | Why it matters |
        |---|---|
        | Learn rules from examples | Only worth it when the rule is too complex to write |
        | `X` features, `y` labels | Every supervised dataset has this shape |
        | Train / validation / test | The only way to get an honest number |
        | Data leakage | Split first, then do everything else |
        | Baseline first | An accuracy with no baseline means nothing |
        | Underfitting / overfitting | Watch the gap between train and validation |
        | Confusion matrix | Shows *which* errors, not just how many |
        | Feature learning | What deep learning adds, and why it matters on images |

        **Next — Lecture 01:** the data becomes images, and we start building the models
        that learn their own features.
        """),
        todo("1", "Explore a dataset", """
        Load the iris dataset. Report the number of samples, the number of features, the
        class names, and the count per class.

        Then plot two features against each other, coloured by class, and say in one
        sentence which pair separates the classes best.
        """),
        todo_cell("# from sklearn.datasets import load_iris"),
        todo("2", "Split the data honestly", """
        Split into 60% train, 20% validation and 20% test, using `stratify`.

        Verify the class proportions are preserved in all three splits, and explain in one
        sentence why stratification matters on a dataset this small.
        """),
        todo_cell(),
        todo("3", "Train three classifiers", """
        Train k-nearest neighbours (`k=5`), a decision tree, and logistic regression on the
        training split. Report training and validation accuracy for each in a small table.
        """),
        todo_cell(),
        todo("4", "Beat the baseline", """
        Compute the majority-class accuracy. State by how much each of your three models
        beats it.

        A model that does not beat this baseline has learned nothing.
        """),
        todo_cell(),
        todo("5", "Make a model overfit on purpose", """
        Train decision trees with `max_depth` from 1 to 15. Plot training and validation
        accuracy against depth on one figure.

        Mark the depth where overfitting begins, and explain in two sentences how you
        identified it.
        """),
        todo_cell(),
        todo("6", "Read a confusion matrix", """
        Produce a confusion matrix for your best model on the **test** split. Name the two
        classes it confuses most, and say why that is unsurprising given your plot from
        Task 1.
        """),
        todo_cell(),
        todo("7", "Where hand-designed features run out", """
        Train logistic regression on the raw pixels of `digits_8x8` and report test
        accuracy.

        Then explain in three sentences why the same approach would break down on a
        224x224 colour photograph. Mention both the parameter count and the fact that the
        model treats every pixel as independent.
        """),
        todo_cell(),
        todo("8", "Cross-validation *(stretch)*", """
        Replace the single validation split with 5-fold cross-validation
        (`sklearn.model_selection.cross_val_score`).

        Report the mean and standard deviation of accuracy, and explain what the standard
        deviation tells you that a single split cannot.
        """),
        todo_cell(),
        todo("9", "Feature scaling *(stretch)*", """
        Train k-nearest neighbours with and without `StandardScaler`. Report both
        accuracies and explain why distance-based methods care about feature scale while
        decision trees do not.

        Remember: fit the scaler on the training split only.
        """),
        todo_cell(),
    ]
