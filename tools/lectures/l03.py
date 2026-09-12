"""Lecture 03 notebook — backpropagation and autodiff from scratch."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. The chain rule on a graph", """
        Backpropagation is not a new idea — it is the chain rule, applied in reverse
        topological order over a graph of primitive operations, with each intermediate
        result reused instead of recomputed.

        Work this example by hand before running it:

        `f(x, y, z) = (x + y) · max(z, 0)` at `x = 2, y = -3, z = 4`.
        """),
        code("""
        # Forward pass, with every intermediate value named.
        x, y, z = 2.0, -3.0, 4.0
        a = x + y            # -1
        r = max(z, 0.0)      #  4
        f = a * r            # -4
        print(f"a = x + y     = {a}")
        print(f"r = max(z, 0) = {r}")
        print(f"f = a * r     = {f}")

        # Backward pass. Start with df/df = 1 and walk backwards.
        df = 1.0
        da = df * r                     # multiply: swap the other input
        dr = df * a
        dx = da * 1.0                   # add: distribute unchanged
        dy = da * 1.0
        dz = dr * (1.0 if z > 0 else 0.0)   # max gate: route to the winner

        print(f"\\ndf/dx = {dx}   df/dy = {dy}   df/dz = {dz}")
        """),
        code("""
        # Confirm against PyTorch.
        xt = torch.tensor(2.0, requires_grad=True)
        yt = torch.tensor(-3.0, requires_grad=True)
        zt = torch.tensor(4.0, requires_grad=True)
        (xt + yt) * torch.clamp(zt, min=0)
        out = (xt + yt) * torch.clamp(zt, min=0)
        out.backward()
        print("torch:", xt.grad.item(), yt.grad.item(), zt.grad.item())
        print("ours :", dx, dy, dz)
        """),
        section("2. A reverse-mode autodiff engine", """
        Here is the whole idea in about 90 lines. Each `Tensor` stores its value, its
        gradient, and a closure that knows how to push gradient to its parents. Calling
        `.backward()` sorts the graph topologically and runs those closures in reverse.

        Read this carefully — every framework you will use is this, with more operations
        and far more engineering.
        """),
        code('''
        class Tensor:
            """A minimal reverse-mode autodiff tensor over NumPy arrays."""

            def __init__(self, data, parents=(), op=""):
                self.data = np.asarray(data, dtype=np.float64)
                self.grad = np.zeros_like(self.data)
                self._backward = lambda: None      # how to push grad to parents
                self._parents = set(parents)
                self._op = op

            def __repr__(self):
                return f"Tensor(shape={self.data.shape}, op={self._op!r})"

            # -- helper: undo broadcasting when pushing a gradient back ----------
            @staticmethod
            def _unbroadcast(grad, shape):
                while grad.ndim > len(shape):
                    grad = grad.sum(axis=0)
                for i, s in enumerate(shape):
                    if s == 1 and grad.shape[i] != 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad.reshape(shape)

            def __add__(self, other):
                other = other if isinstance(other, Tensor) else Tensor(other)
                out = Tensor(self.data + other.data, (self, other), "+")

                def _backward():
                    self.grad  += self._unbroadcast(out.grad, self.data.shape)
                    other.grad += self._unbroadcast(out.grad, other.data.shape)
                out._backward = _backward
                return out

            def __mul__(self, other):
                other = other if isinstance(other, Tensor) else Tensor(other)
                out = Tensor(self.data * other.data, (self, other), "*")

                def _backward():
                    self.grad  += self._unbroadcast(other.data * out.grad, self.data.shape)
                    other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)
                out._backward = _backward
                return out

            def matmul(self, other):
                out = Tensor(self.data @ other.data, (self, other), "@")

                def _backward():
                    self.grad  += out.grad @ other.data.T
                    other.grad += self.data.T @ out.grad
                out._backward = _backward
                return out

            def relu(self):
                out = Tensor(np.maximum(self.data, 0), (self,), "relu")

                def _backward():
                    self.grad += (out.data > 0) * out.grad
                out._backward = _backward
                return out

            def sum(self):
                out = Tensor(self.data.sum(), (self,), "sum")

                def _backward():
                    self.grad += np.ones_like(self.data) * out.grad
                out._backward = _backward
                return out

            def mean(self):
                out = Tensor(self.data.mean(), (self,), "mean")
                n = self.data.size

                def _backward():
                    self.grad += np.ones_like(self.data) * out.grad / n
                out._backward = _backward
                return out

            def log_softmax(self):
                """log-softmax over the last axis, computed stably."""
                shifted = self.data - self.data.max(axis=-1, keepdims=True)
                lse = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
                out_data = shifted - lse
                out = Tensor(out_data, (self,), "log_softmax")

                def _backward():
                    probs = np.exp(out_data)
                    self.grad += out.grad - probs * out.grad.sum(axis=-1, keepdims=True)
                out._backward = _backward
                return out

            __radd__ = __add__
            __rmul__ = __mul__

            def __neg__(self):
                return self * -1.0

            def __sub__(self, other):
                return self + (-(other if isinstance(other, Tensor) else Tensor(other)))

            def backward(self):
                """Topological sort, then run every local backward in reverse."""
                order, seen = [], set()

                def visit(t):
                    if t in seen:
                        return
                    seen.add(t)
                    for p in t._parents:
                        visit(p)
                    order.append(t)

                visit(self)
                self.grad = np.ones_like(self.data)
                for t in reversed(order):
                    t._backward()
        '''),
        section("3. Verifying every operation against PyTorch", """
        An autodiff engine you have not tested is an autodiff engine that is wrong. Check
        each operation against PyTorch with identical inputs.
        """),
        code("""
        def check(name, our_fn, torch_fn, *shapes, tol=1e-6):
            rng = np.random.default_rng(0)
            arrays = [rng.normal(size=s) for s in shapes]

            ours = [Tensor(a.copy()) for a in arrays]
            our_fn(*ours).backward()

            theirs = [torch.tensor(a.copy(), requires_grad=True) for a in arrays]
            torch_fn(*theirs).backward()

            worst = max(np.abs(o.grad - t.grad.numpy()).max() for o, t in zip(ours, theirs))
            status = "PASS" if worst < tol else "FAIL"
            print(f"  {status}  {name:<22} max |Δgrad| = {worst:.2e}")
            return worst < tol

        print("gradient checks against PyTorch:")
        ok = []
        ok.append(check("add",     lambda a, b: (a + b).sum(),
                                   lambda a, b: (a + b).sum(), (4, 5), (4, 5)))
        ok.append(check("mul",     lambda a, b: (a * b).sum(),
                                   lambda a, b: (a * b).sum(), (4, 5), (4, 5)))
        ok.append(check("broadcast add", lambda a, b: (a + b).sum(),
                                   lambda a, b: (a + b).sum(), (4, 5), (1, 5)))
        ok.append(check("matmul",  lambda a, b: a.matmul(b).sum(),
                                   lambda a, b: (a @ b).sum(), (4, 6), (6, 3)))
        ok.append(check("relu",    lambda a: a.relu().sum(),
                                   lambda a: torch.relu(a).sum(), (4, 5)))
        ok.append(check("mean",    lambda a: a.mean(),
                                   lambda a: a.mean(), (4, 5)))
        ok.append(check("log_softmax", lambda a: a.log_softmax().sum(),
                                   lambda a: torch.log_softmax(a, -1).sum(), (4, 5)))
        ok.append(check("reuse (fan-out)", lambda a: (a * a).sum(),
                                   lambda a: (a * a).sum(), (4, 5)))
        print(f"\\n{sum(ok)}/{len(ok)} passed")
        """),
        md("""
        ### Note the fan-out test

        `a * a` uses `a` twice. Because every `_backward` **accumulates** with `+=`
        rather than assigning, the two gradient paths add correctly. Had we written
        `self.grad = ...` the result would be wrong by a factor of two — and it would
        still train, just badly. This is why the test exists.
        """),
        section("4. Training a real network with our own engine", """
        No `torch.nn`, no `torch.optim`. Just the `Tensor` class above.
        """),
        code("""
        digits = load_digits_npz()
        Xtr = digits["train"][0].reshape(-1, 64).numpy().astype(np.float64)
        ytr = digits["train"][1].numpy()
        Xte = digits["test"][0].reshape(-1, 64).numpy().astype(np.float64)
        yte = digits["test"][1].numpy()
        print("train", Xtr.shape, " test", Xte.shape)
        """),
        code("""
        def nll_loss(log_probs: Tensor, y: np.ndarray) -> Tensor:
            \"\"\"Negative log-likelihood, expressed with our own operations so the
            gradient flows through the engine we built.\"\"\"
            onehot = np.zeros_like(log_probs.data)
            onehot[np.arange(len(y)), y] = 1.0
            return -(log_probs * Tensor(onehot)).sum() * (1.0 / len(y))

        class OurMLP:
            def __init__(self, sizes, seed=0):
                r = np.random.default_rng(seed)
                self.params = []
                for fan_in, fan_out in zip(sizes, sizes[1:]):
                    # He initialisation: variance 2/fan_in, correct for ReLU.
                    W = Tensor(r.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)))
                    b = Tensor(np.zeros((1, fan_out)))
                    self.params += [W, b]

            def __call__(self, x):
                h = Tensor(x)
                n_layers = len(self.params) // 2
                for i in range(n_layers):
                    W, b = self.params[2 * i], self.params[2 * i + 1]
                    h = h.matmul(W) + b
                    if i < n_layers - 1:
                        h = h.relu()
                return h.log_softmax()

            def zero_grad(self):
                for p in self.params:
                    p.grad = np.zeros_like(p.data)

            def step(self, lr):
                for p in self.params:
                    p.data -= lr * p.grad
        """),
        code("""
        # ~25 s on a laptop. Reduce EPOCHS if you are impatient.
        EPOCHS, BATCH, LR = 30, 64, 0.15

        net = OurMLP([64, 128, 10])
        rng = np.random.default_rng(0)
        losses, accs = [], []

        for epoch in range(EPOCHS):
            order = rng.permutation(len(Xtr))
            total = 0.0
            for s in range(0, len(Xtr), BATCH):
                idx = order[s:s + BATCH]
                net.zero_grad()
                loss = nll_loss(net(Xtr[idx]), ytr[idx])
                loss.backward()
                net.step(LR)
                total += float(loss.data) * len(idx)
            acc = (net(Xte).data.argmax(1) == yte).mean()
            losses.append(total / len(Xtr)); accs.append(acc)
            if (epoch + 1) % 10 == 0:
                print(f"epoch {epoch+1:3d}  loss {losses[-1]:.4f}  test acc {acc:.3f}")

        print(f"\\nfinal test accuracy: {accs[-1]:.3f}  — trained entirely by our own engine")
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
        a1.plot(losses); a1.set_xlabel("epoch"); a1.set_ylabel("NLL loss")
        a1.set_title("Loss — our autodiff engine"); a1.grid(alpha=.3)
        a2.plot(accs);   a2.set_xlabel("epoch"); a2.set_ylabel("accuracy")
        a2.set_title("Test accuracy"); a2.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        section("5. Vanishing and exploding gradients", """
        The gradient reaching layer 1 of an `L`-layer network is a product of `L` local
        Jacobians. Multiply enough numbers below one together and you get zero; above one
        and you get infinity. This is the single reason deep networks were hard to train
        before roughly 2010.

        Measure it directly.
        """),
        code("""
        def gradient_profile(depth=15, activation="sigmoid", residual=False, width=64, seed=0):
            \"\"\"Return the gradient norm at each layer after one backward pass.\"\"\"
            torch.manual_seed(seed)
            act = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}[activation]

            class Block(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin = nn.Linear(width, width)
                    self.act = act()

                def forward(self, x):
                    out = self.act(self.lin(x))
                    return x + out if residual else out

            layers = nn.Sequential(*[Block() for _ in range(depth)], nn.Linear(width, 4))
            x = torch.randn(128, width)
            y = torch.randint(0, 4, (128,))
            nn.CrossEntropyLoss()(layers(x), y).backward()

            return [b.lin.weight.grad.norm().item() for b in layers[:depth]]

        configs = [
            ("sigmoid, plain",   dict(activation="sigmoid", residual=False), "#c0392b"),
            ("tanh, plain",      dict(activation="tanh",    residual=False), "#e67e22"),
            ("ReLU, plain",      dict(activation="relu",    residual=False), "#2980b9"),
            ("ReLU + residual",  dict(activation="relu",    residual=True),  "#27ae60"),
        ]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for label, kw, colour in configs:
            norms = gradient_profile(**kw)
            ax.semilogy(range(1, len(norms) + 1), norms, marker="o", ms=4,
                        label=label, color=colour)
            print(f"{label:<18} layer 1: {norms[0]:.2e}   layer 15: {norms[-1]:.2e}"
                  f"   ratio: {norms[-1]/max(norms[0], 1e-30):.1e}")

        ax.set_xlabel("layer (1 = closest to the input)")
        ax.set_ylabel("gradient norm (log scale)")
        ax.set_title("Gradient norm by depth — 15 layers, one backward pass")
        ax.legend(); ax.grid(alpha=.3, which="both")
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### Read the plot

        The sigmoid curve falls off a cliff: its derivative peaks at 0.25, so fifteen
        layers multiply by at most `0.25^15 ≈ 10^-9`. The early layers receive essentially
        no signal and never learn.

        ReLU's derivative is exactly 1 wherever the unit is active, so nothing shrinks
        systematically. Residual connections do better still — `y = x + F(x)` gives the
        gradient a path with derivative 1 that skips the multiplications entirely. That
        one observation is what made 152-layer networks trainable (Lecture 6).
        """),
        todo("1", "Hand-derive a graph", """
        For `f(x, y, z) = (x + y) · max(z, 0)` at `x=2, y=-3, z=4`, draw the computational
        graph, run the forward pass, and compute all three partial derivatives by hand.
        Show every intermediate value.
        """),
        todo_cell(),
        todo("2", "Finish the autodiff engine", """
        Extend the `Tensor` class with `exp`, `pow`, and `__truediv__`, each with a
        correct `_backward`. Keep the accumulation semantics (`+=`, never `=`).
        """),
        todo_cell(),
        todo("3", "Verify against PyTorch", """
        For each operation you added, build the same expression in PyTorch with
        `requires_grad=True` and assert the gradients agree to within `1e-6`. Use the
        `check()` helper above.
        """),
        todo_cell(),
        todo("4", "Train with your own engine", """
        Train a two-layer MLP on `digits_8x8` using only your engine — no `torch.nn`, no
        `torch.optim`. Reach at least 92% test accuracy.
        """),
        todo_cell(),
        todo("5", "Gradient-norm profile", """
        Build a 15-layer sigmoid MLP and plot the gradient norm of every layer on a log
        scale after one backward pass. Repeat with ReLU and with residual connections,
        and put all three on one figure. Explain the ordering.
        """),
        todo_cell(),
        todo("6", "Add Conv2d *(stretch)*", """
        Extend your engine with a 2D convolution and its backward pass. Verify against
        `torch.nn.functional.conv2d` on a random `(4, 3, 8, 8)` input.
        """),
        todo_cell(),
        todo("7", "Gradient checkpointing *(stretch)*", """
        Modify the engine to recompute activations during the backward pass instead of
        storing them. Measure peak memory and wall-clock time for both versions and
        report the trade-off.
        """),
        todo_cell(),
    ]
