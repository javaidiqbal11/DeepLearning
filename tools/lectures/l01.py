"""Lecture 01 notebook — foundations and the image data pipeline."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. An image is a tensor", """
        Everything in this course starts here. Before any model exists, you need to be
        fluent in what an image *is* to a computer, and in the two layout conventions
        that will otherwise cost you an afternoon of debugging.
        """),
        code("""
        from PIL import Image

        img_path = ROOT / "data" / "shapes_imagefolder" / "train" / "circle"
        one = sorted(img_path.glob("*.png"))[0]

        pil = Image.open(one)
        arr = np.array(pil)

        print("PIL size (W, H) :", pil.size)
        print("NumPy shape     :", arr.shape, "  <- H, W, C")
        print("dtype           :", arr.dtype, " range", arr.min(), "-", arr.max())
        """),
        md("""
        ### HWC versus NCHW

        NumPy, PIL and matplotlib all think in **HWC** — height, width, channels.
        PyTorch convolutions expect **NCHW** — batch, channels, height, width.

        Converting between them is a `permute`, and forgetting it is the single most
        common day-one error. The symptom is usually a shape mismatch, but sometimes
        it silently "works" and trains to nonsense.
        """),
        code("""
        # HWC uint8  ->  NCHW float32 in [0, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        batch = t.unsqueeze(0)                      # add the batch dimension

        print("HWC numpy :", arr.shape)
        print("CHW tensor:", tuple(t.shape))
        print("NCHW batch:", tuple(batch.shape))

        # ...and back again, for plotting
        back = (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        print("round trip identical:", np.array_equal(arr, back))
        """),
        code("""
        fig, axes = plt.subplots(1, 4, figsize=(11, 3))
        axes[0].imshow(arr);               axes[0].set_title("RGB composite")
        for i, (name, cmap) in enumerate([("red", "Reds"), ("green", "Greens"), ("blue", "Blues")]):
            axes[i + 1].imshow(arr[:, :, i], cmap=cmap, vmin=0, vmax=255)
            axes[i + 1].set_title(f"{name} channel")
        for a in axes:
            a.axis("off")
        fig.suptitle("One image, four views — the channels are just three stacked matrices")
        plt.tight_layout(); plt.show()
        """),
        section("2. Loading the dataset three ways", """
        You will meet all three of these in practice: arrays already in memory, a custom
        `Dataset` class, and a folder of image files on disk.
        """),
        code("""
        # --- way 1: arrays, already in memory ---------------------------------
        data = load_shapes()
        X_train, y_train = data["train"]
        X_val,   y_val   = data["val"]
        X_test,  y_test  = data["test"]
        CLASSES = data["classes"]

        print("classes  :", CLASSES)
        print("train    :", tuple(X_train.shape), tuple(y_train.shape))
        print("val      :", tuple(X_val.shape))
        print("test     :", tuple(X_test.shape))
        print("dtype    :", X_train.dtype, " range %.2f - %.2f" % (X_train.min(), X_train.max()))
        print("per-class counts:", torch.bincount(y_train).tolist())
        """),
        code("""
        show_grid(X_train[:16], y_train[:16], CLASSES, n=16, ncols=8,
                  title="shapes_32 — what the model has to separate")
        plt.show()
        """),
        code("""
        # --- way 2: a Dataset + DataLoader ------------------------------------
        from torch.utils.data import Dataset, DataLoader

        class ShapesTensorDataset(Dataset):
            \"\"\"The minimum a Dataset must implement: __len__ and __getitem__.\"\"\"

            def __init__(self, X, y, transform=None):
                self.X, self.y, self.transform = X, y, transform

            def __len__(self):
                return len(self.X)

            def __getitem__(self, i):
                x = self.X[i]
                if self.transform is not None:
                    x = self.transform(x)
                return x, self.y[i]

        train_ds = ShapesTensorDataset(X_train, y_train)
        # num_workers=0 on Windows: worker spawning misbehaves inside Jupyter.
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

        xb, yb = next(iter(train_dl))
        print("one batch:", tuple(xb.shape), tuple(yb.shape))
        print("batches per epoch:", len(train_dl))
        """),
        code("""
        # --- way 3: real files on disk ----------------------------------------
        folder = ROOT / "data" / "shapes_imagefolder" / "train"
        class_dirs = sorted(p.name for p in folder.iterdir() if p.is_dir())
        counts = {c: len(list((folder / c).glob("*.png"))) for c in class_dirs}

        print("class folders (sorted -> stable label mapping):", class_dirs)
        print("file counts:", counts)
        print("\\nSorting matters: if the label mapping changes between training and")
        print("serving, your model will be confidently, silently wrong.")
        """),
        section("3. Normalisation, done honestly", """
        Standardising inputs puts every channel on a comparable scale, which makes the
        loss surface better conditioned and lets you use a larger learning rate.

        The statistics must come from the **training split only**. Computing them over
        all the data leaks information from validation and test into training — a small
        leak, but the same mistake in a larger form is how published results stop
        reproducing.
        """),
        code("""
        # Correct: statistics from the training split alone.
        train_mean = X_train.mean(dim=(0, 2, 3))
        train_std  = X_train.std(dim=(0, 2, 3))

        # Wrong, shown so you recognise it: statistics over everything.
        everything = torch.cat([X_train, X_val, X_test])
        all_mean = everything.mean(dim=(0, 2, 3))
        all_std  = everything.std(dim=(0, 2, 3))

        print("train-only mean:", [f"{v:.4f}" for v in train_mean.tolist()])
        print("train-only std :", [f"{v:.4f}" for v in train_std.tolist()])
        print("all-data   mean:", [f"{v:.4f}" for v in all_mean.tolist()], " <- leaks")
        print("all-data   std :", [f"{v:.4f}" for v in all_std.tolist()],  " <- leaks")
        print("\\nThe numbers barely differ here. That is exactly why the habit matters:")
        print("you cannot detect the leak by looking at the result.")
        """),
        code("""
        def standardise(X, mean, std):
            return (X - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

        Xtr_n = standardise(X_train, train_mean, train_std)
        Xva_n = standardise(X_val,   train_mean, train_std)
        Xte_n = standardise(X_test,  train_mean, train_std)

        print("before: mean %.3f  std %.3f" % (X_train.mean(), X_train.std()))
        print("after : mean %.3f  std %.3f" % (Xtr_n.mean(), Xtr_n.std()))
        """),
        section("4. Baselines — the numbers everything else must beat", """
        Never train a network before you know what trivial approaches achieve. A model
        that beats chance is not evidence of anything; a model that beats a tuned linear
        classifier is.
        """),
        code("""
        # Baseline 0: always predict the most common class.
        majority = torch.bincount(y_train).argmax().item()
        majority_acc = (y_test == majority).float().mean().item()
        print(f"majority class      : '{CLASSES[majority]}'")
        print(f"majority accuracy   : {majority_acc:.3f}   (chance = {1/len(CLASSES):.3f})")
        """),
        code("""
        # Baseline 1: multinomial logistic regression on raw pixels.
        from sklearn.linear_model import LogisticRegression

        flat_tr = Xtr_n.reshape(len(Xtr_n), -1).numpy()
        flat_te = Xte_n.reshape(len(Xte_n), -1).numpy()

        linear = LogisticRegression(max_iter=400, n_jobs=-1)
        linear.fit(flat_tr[:3000], y_train[:3000].numpy())     # 3000 keeps this under a minute
        linear_acc = linear.score(flat_te, y_test.numpy())
        print(f"linear baseline     : {linear_acc:.3f}")
        """),
        code("""
        results = {
            "chance":            1 / len(CLASSES),
            "majority class":    majority_acc,
            "logistic (pixels)": linear_acc,
        }

        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.barh(list(results), list(results.values()), color=["#bbb", "#888", "#1f6fb2"])
        for i, v in enumerate(results.values()):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
        ax.set_xlim(0, 1); ax.set_xlabel("test accuracy")
        ax.set_title("Lecture 1 baselines — keep these numbers")
        plt.tight_layout(); plt.show()

        print("\\nBy Lecture 5 a small CNN reaches ~0.99 on this same test set.")
        print("That gap is the entire subject of this course.")
        """),
        section("5. Why the linear model fails", """
        Look at *how* it fails, not just how often. The confusion matrix turns one number
        into a map of which classes the model cannot tell apart.
        """),
        code("""
        pred = linear.predict(flat_te)
        plot_confusion(y_test.numpy(), pred, CLASSES,
                       title=f"Linear baseline — {linear_acc:.1%} accuracy")
        plt.show()

        print("Squares, triangles and stars are all 'a blob of colour in the middle' to a")
        print("model that sees each pixel independently. Only a model that can see local")
        print("structure — corners, edges, counts of them — can separate them.")
        """),
        todo("1", "Tensor layout drill", """
        Write `round_trip(path)` that loads a PNG with PIL, converts it to a normalised
        NCHW float tensor, converts it back to a uint8 HWC array, and asserts the result
        matches the original within `1/255`.

        Then state in one sentence where the rounding error comes from.
        """),
        todo_cell("# def round_trip(path): ..."),
        todo("2", "Custom Dataset", """
        Implement `ShapesFolder(Dataset)` that reads `data/shapes_imagefolder/train`
        **without** using `torchvision.datasets.ImageFolder`.

        It must discover class names from the directory listing, sort them for a stable
        label mapping, and return `(CHW float tensor, int label)`.
        """),
        todo_cell("# class ShapesFolder(torch.utils.data.Dataset): ..."),
        todo("3", "Honest normalisation", """
        Compute per-channel mean and std on the training split only, then on
        train+val+test combined. Report both to four decimal places and explain in two
        sentences why only the first is admissible.
        """),
        todo_cell(),
        todo("4", "Baselines that matter", """
        Report majority-class accuracy and logistic-regression accuracy on the test
        split. Start a `results.md` table — you will add a row to it in every remaining
        lecture.
        """),
        todo_cell(),
        todo("5", "Batch-size timing study *(stretch)*", """
        Time one epoch of a linear model at batch sizes 1, 16, 128 and 1024. Plot seconds
        per epoch against batch size on a log x-axis, and explain the shape of the curve
        in terms of Python overhead versus vectorised throughput.
        """),
        todo_cell(),
        section("6. Find the leak (stretch)", """
        `leaky_split.py` in this lecture's folder builds a train/test split with a
        deliberate flaw. Run it, find the flaw, quantify how much it inflates the
        reported accuracy, and fix it.
        """),
        code("""
        leaky = ROOT / "lectures" / "Lecture_01" / "leaky_split.py"
        print(leaky.read_text(encoding="utf-8"))
        """),
        todo("6", "Find the leak", """
        Identify the flaw in `leaky_split.py`, quantify the inflation in reported
        accuracy, and write a corrected version.

        *Hint: compare the reported accuracy against an honest split, and look carefully
        at how the two splits are constructed.*
        """),
        todo_cell(),
    ]
