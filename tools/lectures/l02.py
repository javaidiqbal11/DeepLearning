"""Lecture 02 notebook — linear models to neural networks."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. The linear score function", """
        A linear classifier computes `s = Wx + b`: one score per class, from a flattened
        image. Each **row** of `W` is a template the model compares the image against.
        """),
        code("""
        digits = load_digits_npz()
        Xtr, ytr = digits["train"]
        Xte, yte = digits["test"]
        DIGIT_CLASSES = digits["classes"]

        # Flatten 1x8x8 -> 64 and work in NumPy for this section.
        Xtr_f = Xtr.reshape(len(Xtr), -1).numpy().astype(np.float64)
        Xte_f = Xte.reshape(len(Xte), -1).numpy().astype(np.float64)
        ytr_n, yte_n = ytr.numpy(), yte.numpy()

        print("train:", Xtr_f.shape, " test:", Xte_f.shape, " classes:", len(DIGIT_CLASSES))
        show_grid(Xtr[:16], ytr[:16], DIGIT_CLASSES, n=16, ncols=8,
                  title="scikit-learn digits — 8x8 grayscale, real handwriting")
        plt.show()
        """),
        section("2. Softmax and cross-entropy", """
        Softmax turns scores into a probability distribution; cross-entropy is the
        negative log-likelihood of the correct class.

        The one implementation detail that matters: subtract the row maximum before
        exponentiating. `exp(1000)` is `inf`, and `inf/inf` is `nan`.
        """),
        code("""
        def softmax_naive(scores):
            e = np.exp(scores)
            return e / e.sum(axis=1, keepdims=True)

        def softmax(scores):
            \"\"\"Numerically stable softmax. Shifting by the row max cannot change
            the result, because exp(a-c)/sum(exp(b-c)) = exp(a)/sum(exp(b)).\"\"\"
            shifted = scores - scores.max(axis=1, keepdims=True)
            e = np.exp(shifted)
            return e / e.sum(axis=1, keepdims=True)

        danger = np.array([[1000.0, 1001.0, 999.0]])
        with np.errstate(over="ignore", invalid="ignore"):
            print("naive :", softmax_naive(danger))
        print("stable:", softmax(danger))
        print("\\nBoth are mathematically identical. Only one of them runs.")
        """),
        code("""
        def cross_entropy(probs, y):
            \"\"\"Mean negative log-likelihood of the correct class.\"\"\"
            n = len(y)
            return -np.log(probs[np.arange(n), y] + 1e-12).mean()

        def forward(W, b, X):
            return X @ W + b

        # At initialisation every class is equally likely, so the loss should be
        # -log(1/10). If your initial loss is not this, something is already wrong.
        rng = np.random.default_rng(0)
        W0 = rng.normal(0, 0.001, (64, 10))
        b0 = np.zeros(10)
        p0 = softmax(forward(W0, b0, Xtr_f))
        print(f"initial loss : {cross_entropy(p0, ytr_n):.4f}")
        print(f"-log(1/10)   : {-np.log(1/10):.4f}   <- the sanity check")
        """),
        md("""
        ### The gradient

        For softmax with cross-entropy, the gradient with respect to the scores is
        remarkably clean:

        $$\\frac{\\partial L}{\\partial s} = p - \\mathbf{1}_{y}$$

        The predicted probability minus a one-hot of the truth. Everything else follows
        by the chain rule:

        $$\\frac{\\partial L}{\\partial W} = X^\\top (p - \\mathbf{1}_y) / N
        \\qquad
        \\frac{\\partial L}{\\partial b} = \\text{sum}(p - \\mathbf{1}_y) / N$$
        """),
        code("""
        def gradients(W, b, X, y):
            n = len(y)
            probs = softmax(forward(W, b, X))
            dscores = probs.copy()
            dscores[np.arange(n), y] -= 1.0      # p - onehot(y)
            dscores /= n
            return X.T @ dscores, dscores.sum(axis=0), cross_entropy(probs, y)

        # Gradient check: compare against a central finite difference.
        def numerical_grad(f, x, h=1e-5, n_checks=12):
            grad = np.zeros_like(x)
            idx = [tuple(rng.integers(0, s) for s in x.shape) for _ in range(n_checks)]
            for i in idx:
                old = x[i]
                x[i] = old + h; fp = f()
                x[i] = old - h; fm = f()
                x[i] = old
                grad[i] = (fp - fm) / (2 * h)
            return grad, idx

        Wc = rng.normal(0, 0.01, (64, 10))
        bc = np.zeros(10)
        Xs, ys = Xtr_f[:50], ytr_n[:50]

        dW, db, _ = gradients(Wc, bc, Xs, ys)
        num_dW, checked = numerical_grad(
            lambda: cross_entropy(softmax(forward(Wc, bc, Xs)), ys), Wc)

        errs = [abs(dW[i] - num_dW[i]) / max(abs(dW[i]) + abs(num_dW[i]), 1e-12) for i in checked]
        print(f"worst relative error over {len(checked)} entries: {max(errs):.2e}")
        print("anything below 1e-6 means the analytic gradient is correct")
        """),
        section("3. Training the linear classifier", """
        Mini-batch gradient descent, written out in full so nothing is hidden.
        """),
        code("""
        def train_linear(X, y, Xval, yval, epochs=40, lr=0.5, batch_size=64, seed=0):
            r = np.random.default_rng(seed)
            W = r.normal(0, 0.01, (X.shape[1], 10))
            b = np.zeros(10)
            hist = {"loss": [], "val_acc": []}

            for _ in range(epochs):
                order = r.permutation(len(X))
                epoch_loss = 0.0
                for s in range(0, len(X), batch_size):
                    idx = order[s:s + batch_size]
                    dW, db, loss = gradients(W, b, X[idx], y[idx])
                    W -= lr * dW
                    b -= lr * db
                    epoch_loss += loss * len(idx)
                hist["loss"].append(epoch_loss / len(X))
                acc = (forward(W, b, Xval).argmax(1) == yval).mean()
                hist["val_acc"].append(acc)
            return W, b, hist

        W, b, hist = train_linear(Xtr_f, ytr_n, Xte_f, yte_n)
        print(f"final test accuracy: {hist['val_acc'][-1]:.3f}")
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
        a1.plot(hist["loss"]);     a1.set_xlabel("epoch"); a1.set_ylabel("loss")
        a1.set_title("Training loss"); a1.grid(alpha=.3)
        a2.plot(hist["val_acc"]);  a2.set_xlabel("epoch"); a2.set_ylabel("accuracy")
        a2.set_title("Test accuracy"); a2.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        code("""
        # Each column of W, reshaped to 8x8, is the template for one digit.
        fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))
        for k, ax in enumerate(axes.ravel()):
            ax.imshow(W[:, k].reshape(8, 8), cmap="RdBu_r")
            ax.set_title(f"class {k}", fontsize=10); ax.axis("off")
        fig.suptitle("Learned templates — red = evidence for, blue = evidence against")
        plt.tight_layout(); plt.show()
        """),
        section("4. Where linearity runs out", """
        A linear classifier partitions the input space with hyperplanes. If the classes
        are not linearly separable, no amount of training fixes it — the model simply
        cannot express the boundary.

        Two moons is the cleanest demonstration.
        """),
        code("""
        from sklearn.datasets import make_moons

        Xm, ym = make_moons(n_samples=1000, noise=0.18, random_state=0)
        Xm = torch.tensor(Xm, dtype=torch.float32)
        ym = torch.tensor(ym, dtype=torch.long)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="coolwarm", s=12, edgecolors="none")
        ax.set_title("Two moons — no straight line separates these")
        plt.tight_layout(); plt.show()
        """),
        code("""
        def fit_torch(model, X, y, epochs=400, lr=0.05):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            lf = nn.CrossEntropyLoss()
            for _ in range(epochs):
                opt.zero_grad()
                loss = lf(model(X), y)
                loss.backward()
                opt.step()
            with torch.no_grad():
                acc = (model(X).argmax(1) == y).float().mean().item()
            return acc

        set_seed(0)
        linear_model = nn.Linear(2, 2)
        mlp_model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))

        acc_lin = fit_torch(linear_model, Xm, ym)
        acc_mlp = fit_torch(mlp_model, Xm, ym)
        print(f"linear            : {acc_lin:.3f}")
        print(f"MLP (32 hidden)   : {acc_mlp:.3f}")
        """),
        code("""
        def decision_surface(model, ax, title):
            xs = torch.linspace(Xm[:, 0].min() - .5, Xm[:, 0].max() + .5, 220)
            ys = torch.linspace(Xm[:, 1].min() - .5, Xm[:, 1].max() + .5, 220)
            gx, gy = torch.meshgrid(xs, ys, indexing="xy")
            grid = torch.stack([gx.ravel(), gy.ravel()], dim=1)
            with torch.no_grad():
                zz = model(grid).argmax(1).reshape(gx.shape)
            ax.contourf(gx, gy, zz, alpha=.25, cmap="coolwarm", levels=1)
            ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="coolwarm", s=10, edgecolors="none")
            ax.set_title(title)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        decision_surface(linear_model, a1, f"Linear — {acc_lin:.1%}")
        decision_surface(mlp_model,   a2, f"One hidden layer + ReLU — {acc_mlp:.1%}")
        plt.tight_layout(); plt.show()

        print("One hidden layer and a non-linearity. That is the whole difference.")
        """),
        section("5. Depth without a non-linearity buys nothing", """
        `W2(W1 x) = (W2 W1) x`. Composing linear maps gives another linear map, so a
        five-layer network with no activations has exactly the expressive power of one
        linear layer. Here is the proof by experiment.
        """),
        code("""
        set_seed(0)
        deep_linear = nn.Sequential(
            nn.Linear(2, 64), nn.Linear(64, 64), nn.Linear(64, 64),
            nn.Linear(64, 64), nn.Linear(64, 2),
        )
        acc_deep = fit_torch(deep_linear, Xm, ym)

        print(f"1 linear layer          : {acc_lin:.3f}")
        print(f"5 linear layers, no ReLU: {acc_deep:.3f}")
        print(f"1 hidden layer + ReLU   : {acc_mlp:.3f}")

        # The product of all five weight matrices is a single 2x2 map.
        with torch.no_grad():
            prod = torch.eye(2)
            for layer in deep_linear:
                prod = layer.weight @ prod
        print(f"\\nproduct of all five weight matrices -> shape {tuple(prod.shape)}:")
        print(prod.numpy().round(3))
        print("\\nFive layers collapse to one 2x2 matrix. Depth without non-linearity is decoration.")
        """),
        section("6. Activation functions", """
        ReLU is the default for a reason. Sigmoid and tanh saturate: once a unit's input
        is large in magnitude, the gradient through it is nearly zero and learning stops.
        """),
        code("""
        z = torch.linspace(-6, 6, 400, requires_grad=True)
        acts = {
            "ReLU":      F.relu,
            "Sigmoid":   torch.sigmoid,
            "Tanh":      torch.tanh,
            "LeakyReLU": lambda t: F.leaky_relu(t, 0.1),
            "GELU":      F.gelu,
        }

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        for name, fn in acts.items():
            out = fn(z)
            g, = torch.autograd.grad(out.sum(), z, retain_graph=True)
            a1.plot(z.detach(), out.detach(), label=name)
            a2.plot(z.detach(), g.detach(), label=name)
        a1.set_title("Activation"); a2.set_title("Its derivative")
        for a in (a1, a2):
            a.grid(alpha=.3); a.legend(fontsize=9); a.axhline(0, color="k", lw=.5)
        a2.annotate("sigmoid's gradient is ~0 outside [-4, 4]\\nand peaks at only 0.25",
                    xy=(3.5, 0.05), xytext=(1.0, 0.6), fontsize=9,
                    arrowprops=dict(arrowstyle="->", lw=.8))
        plt.tight_layout(); plt.show()
        """),
        todo("1", "Numerically stable softmax", """
        Implement `softmax(scores)` that handles a row containing `10000` without
        producing `NaN`. Demonstrate the naive version overflowing and yours not.
        """),
        todo_cell(),
        todo("2", "Analytic gradient plus gradient check", """
        Implement the cross-entropy gradient for a linear classifier and verify it
        against a central-difference numerical gradient. Relative error must be below
        `1e-6`. Report the worst element you checked.
        """),
        todo_cell(),
        todo("3", "Train the linear classifier", """
        Train your own NumPy softmax classifier on `digits_8x8` with mini-batch gradient
        descent. Reach at least 90% test accuracy and plot the loss curve.
        """),
        todo_cell(),
        todo("4", "Visualise the templates", """
        Reshape each column of the learned `W` to 8x8 and plot all ten. Describe in two
        sentences what the template for the digit `0` has learned to detect.
        """),
        todo_cell(),
        todo("5", "One hidden layer changes everything", """
        On two-moons, fit a linear classifier and an MLP with one hidden layer of 32 ReLU
        units. Plot both decision boundaries side by side and report both accuracies.
        """),
        todo_cell(),
        todo("6", "Activation comparison *(stretch)*", """
        Train the same MLP on the **shapes** dataset with ReLU, tanh, sigmoid and
        LeakyReLU. Plot all four loss curves on one axis and explain the sigmoid result
        in terms of gradient saturation.
        """),
        todo_cell("# d = load_shapes(); X, y = d['train']"),
        todo("7", "Depth without non-linearity *(stretch)*", """
        Build a five-layer network with no activation functions. Show that its test
        accuracy matches a single linear layer, and confirm that the product of its
        weight matrices is a rank-limited linear map.
        """),
        todo_cell(),
    ]
