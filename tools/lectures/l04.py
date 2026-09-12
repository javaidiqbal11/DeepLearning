"""Lecture 04 notebook — optimisation and regularisation."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("0. Setup", """
        One dataset and one model definition, reused across every experiment below, so
        that the only thing changing between runs is the thing we are studying.
        """),
        code("""
        data = load_shapes(normalize=True)
        X_train, y_train = data["train"]
        X_val,   y_val   = data["val"]
        X_test,  y_test  = data["test"]
        CLASSES = data["classes"]

        # A deliberately small MLP: big enough to learn, small enough to run many times.
        def make_mlp(hidden=256, dropout=0.0, seed=0):
            torch.manual_seed(seed)
            layers = [nn.Flatten(), nn.Linear(3 * 32 * 32, hidden), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hidden, hidden // 2), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden // 2, len(CLASSES)))
            return nn.Sequential(*layers)

        print("train", tuple(X_train.shape), " val", tuple(X_val.shape))
        print("parameters:", count_parameters(make_mlp()))
        """),
        section("1. Optimiser bake-off", """
        Same architecture, same seed, same number of epochs. The only difference is the
        update rule. This is the only way to compare optimisers honestly.
        """),
        code("""
        # ~60 s total on a laptop.
        EPOCHS = 10

        def run(optimiser_name, lr, **kw):
            model = make_mlp(seed=0)
            opts = {
                "SGD":          lambda p: torch.optim.SGD(p, lr=lr),
                "SGD+momentum": lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9),
                "RMSProp":      lambda p: torch.optim.RMSprop(p, lr=lr),
                "Adam":         lambda p: torch.optim.Adam(p, lr=lr),
                "AdamW":        lambda p: torch.optim.AdamW(p, lr=lr, weight_decay=0.01),
            }
            set_seed(0)
            return train(model, (X_train, y_train), (X_val, y_val),
                         epochs=EPOCHS, optimizer=opts[optimiser_name](model.parameters()),
                         batch_size=128, verbose=False)

        configs = [("SGD", 0.05), ("SGD+momentum", 0.05), ("RMSProp", 1e-3),
                   ("Adam", 1e-3), ("AdamW", 1e-3)]
        histories = {}
        for name, lr in configs:
            histories[name] = run(name, lr)
            best_ep, best = histories[name].best()
            print(f"{name:<14} lr={lr:<7} best val acc {best:.3f} (epoch {best_ep+1})")
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
        for name, h in histories.items():
            ep = range(1, len(h.val_loss) + 1)
            a1.plot(ep, h.val_loss, marker="o", ms=3, label=name)
            a2.plot(ep, h.val_acc,  marker="o", ms=3, label=name)
        a1.set_xlabel("epoch"); a1.set_ylabel("validation loss"); a1.set_title("Validation loss")
        a2.set_xlabel("epoch"); a2.set_ylabel("validation accuracy"); a2.set_title("Validation accuracy")
        for a in (a1, a2):
            a.grid(alpha=.3); a.legend(fontsize=9)
        fig.suptitle("Optimiser comparison — identical model, seed and epoch budget")
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### What to notice

        Adam gets moving fastest — that is its reputation and it is deserved. Plain SGD
        is slowest here, but note that on large-scale vision benchmarks well-tuned SGD
        with momentum often **ends up** at a better final accuracy than Adam. Fast early
        progress and best final result are not the same property.

        AdamW is the right default for anything Transformer-shaped (Lecture 13).
        """),
        section("2. The learning rate range test", """
        The learning rate is the hyper-parameter that matters most, and you can find a
        good one in a single epoch. Increase it exponentially batch by batch and watch
        where the loss turns.
        """),
        code("""
        def lr_range_test(lr_min=1e-5, lr_max=1e0, n_batches=200, batch_size=128):
            model = make_mlp(seed=0)
            opt = torch.optim.SGD(model.parameters(), lr=lr_min, momentum=0.9)
            lf = nn.CrossEntropyLoss()
            mult = (lr_max / lr_min) ** (1 / n_batches)

            lrs, losses = [], []
            set_seed(0)
            perm = torch.randperm(len(X_train))
            lr = lr_min
            for i in range(n_batches):
                idx = perm[(i * batch_size) % len(X_train):][:batch_size]
                if len(idx) < 2:
                    perm = torch.randperm(len(X_train)); continue
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad()
                loss = lf(model(X_train[idx]), y_train[idx])
                loss.backward()
                opt.step()

                lrs.append(lr); losses.append(loss.item())
                lr *= mult
                if loss.item() > 10 * min(losses):      # diverged; stop early
                    break
            return lrs, losses

        lrs, losses = lr_range_test()

        # Smooth the noisy per-batch loss so the turning point is visible.
        smooth, beta, avg = [], 0.9, 0.0
        for i, l in enumerate(losses):
            avg = beta * avg + (1 - beta) * l
            smooth.append(avg / (1 - beta ** (i + 1)))

        best_i = int(np.argmin(smooth))
        suggested = lrs[best_i] / 10          # an order of magnitude below the minimum

        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.semilogx(lrs, smooth, lw=2)
        ax.axvline(lrs[best_i], color="#c0392b", ls="--", label=f"minimum @ {lrs[best_i]:.1e}")
        ax.axvline(suggested, color="#27ae60", ls="--", label=f"suggested @ {suggested:.1e}")
        ax.set_xlabel("learning rate (log scale)"); ax.set_ylabel("smoothed loss")
        ax.set_title("LR range test — one epoch tells you the usable range")
        ax.legend(); ax.grid(alpha=.3, which="both")
        plt.tight_layout(); plt.show()

        print(f"loss minimum at lr = {lrs[best_i]:.2e}")
        print(f"pick roughly an order of magnitude below: {suggested:.2e}")
        """),
        section("3. Initialisation and activation variance", """
        A bad initialisation looks exactly like a bad learning rate: the loss sits flat,
        or explodes immediately. The diagnostic is to measure the variance of each
        layer's activations on a forward pass.

        What you want is a flat line. Anything that decays or grows with depth compounds.
        """),
        code("""
        def activation_variances(init, depth=10, width=256, seed=0):
            torch.manual_seed(seed)
            layers = []
            for _ in range(depth):
                lin = nn.Linear(width, width)
                if init == "zeros":
                    nn.init.zeros_(lin.weight)
                elif init == "normal_0.01":
                    nn.init.normal_(lin.weight, 0, 0.01)
                elif init == "xavier":
                    nn.init.xavier_normal_(lin.weight)
                elif init == "he":
                    nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
                layers += [lin, nn.ReLU()]

            x = torch.randn(512, width)
            variances = []
            with torch.no_grad():
                for layer in layers:
                    x = layer(x)
                    if isinstance(layer, nn.ReLU):
                        variances.append(x.var().item())
            return variances

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for init, colour in [("zeros", "#7f8c8d"), ("normal_0.01", "#c0392b"),
                             ("xavier", "#e67e22"), ("he", "#27ae60")]:
            v = activation_variances(init)
            ax.semilogy([max(x, 1e-30) for x in v], marker="o", ms=4, label=init, color=colour)
            print(f"{init:<14} layer 1: {v[0]:.3e}   layer 10: {v[-1]:.3e}")

        ax.set_xlabel("layer"); ax.set_ylabel("activation variance (log scale)")
        ax.set_title("Activation variance through a 10-layer ReLU network")
        ax.legend(); ax.grid(alpha=.3, which="both")
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### Why He and not Xavier

        Xavier was derived for `tanh`, targeting variance `1/fan_in`. ReLU zeroes half
        its inputs, which halves the variance at every layer — so Xavier decays by `2^-L`
        through an L-layer ReLU stack.

        He initialisation uses `2/fan_in`, and that factor of two is exactly the
        correction. It is why the green line is flat.

        All-zeros never breaks symmetry: every unit in a layer computes the same thing
        and receives the same gradient, forever. The network has one effective unit per
        layer no matter how wide you make it.
        """),
        section("4. Normalisation layers", """
        BatchNorm standardises each channel across the batch, then applies a learned
        scale and shift. It permits larger learning rates and makes the network much less
        sensitive to initialisation.

        It also behaves **differently in train and eval mode**, and forgetting that is a
        genuine production bug — you will meet it again in Lecture 9.
        """),
        code("""
        def small_cnn(batchnorm=True, seed=0):
            torch.manual_seed(seed)
            def block(cin, cout):
                layers = [nn.Conv2d(cin, cout, 3, padding=1)]
                if batchnorm:
                    layers.append(nn.BatchNorm2d(cout))
                layers += [nn.ReLU(), nn.MaxPool2d(2)]
                return layers
            return nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64),
                                 nn.Flatten(), nn.Linear(64 * 4 * 4, len(CLASSES)))

        # BatchNorm lets a much larger learning rate work.
        for bn in (False, True):
            set_seed(0)
            h = train(small_cnn(batchnorm=bn), (X_train, y_train), (X_val, y_val),
                      epochs=6, lr=0.01, batch_size=128, verbose=False)
            print(f"BatchNorm={str(bn):<5} lr=0.01  best val acc {h.best()[1]:.3f}")
        """),
        code("""
        set_seed(0)
        bn_model = small_cnn(batchnorm=True)
        train(bn_model, (X_train, y_train), (X_val, y_val), epochs=6, lr=0.01, verbose=False)

        # The same data, the same weights, two different answers.
        bn_model.train()
        with torch.no_grad():
            acc_train_mode = (bn_model(X_test[:512]).argmax(1) == y_test[:512]).float().mean()
        bn_model.eval()
        with torch.no_grad():
            acc_eval_mode = (bn_model(X_test[:512]).argmax(1) == y_test[:512]).float().mean()

        print(f"test accuracy in .train() mode : {acc_train_mode:.3f}   <- uses batch statistics")
        print(f"test accuracy in .eval()  mode : {acc_eval_mode:.3f}   <- uses running statistics")
        print("\\nOnly the second is a valid measurement. Call model.eval() before you evaluate.")
        """),
        section("5. Regularisation — an ablation", """
        First build something that overfits, then measure what each intervention actually
        buys. A small training subset makes the overfitting unmistakable.
        """),
        code("""
        # 800 training examples is few enough that a 256-wide MLP memorises them.
        SMALL = 800
        Xs, ys = X_train[:SMALL], y_train[:SMALL]

        def ablate(dropout=0.0, weight_decay=0.0, epochs=25):
            set_seed(0)
            model = make_mlp(dropout=dropout, seed=0)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
            h = train(model, (Xs, ys), (X_val, y_val), epochs=epochs,
                      optimizer=opt, batch_size=64, verbose=False)
            _, train_acc = evaluate(model, (Xs, ys))
            return train_acc, h.val_acc[-1], h

        rows = []
        for label, kw in [("none",                dict()),
                          ("weight decay 1e-3",   dict(weight_decay=1e-3)),
                          ("dropout 0.3",         dict(dropout=0.3)),
                          ("both",                dict(dropout=0.3, weight_decay=1e-3))]:
            tr_acc, val_acc, h = ablate(**kw)
            rows.append((label, tr_acc, val_acc, tr_acc - val_acc, h))
            print(f"{label:<20} train {tr_acc:.3f}   val {val_acc:.3f}   gap {tr_acc-val_acc:+.3f}")
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
        for label, _, _, _, h in rows:
            a1.plot(h.val_loss, label=label)
            a2.plot(h.val_acc,  label=label)
        a1.set_xlabel("epoch"); a1.set_ylabel("validation loss"); a1.set_title("Validation loss")
        a2.set_xlabel("epoch"); a2.set_ylabel("validation accuracy"); a2.set_title("Validation accuracy")
        for a in (a1, a2):
            a.grid(alpha=.3); a.legend(fontsize=9)
        fig.suptitle(f"Regularisation ablation on {SMALL} training examples")
        plt.tight_layout(); plt.show()

        labels = [r[0] for r in rows]
        gaps = [r[3] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(labels, gaps, color="#c0392b")
        ax.set_xlabel("train accuracy − validation accuracy  (smaller is better)")
        ax.set_title("Generalisation gap")
        plt.tight_layout(); plt.show()
        """),
        section("6. Reading learning curves", """
        Four shapes, four diagnoses. Learn to recognise them — this is what saves you
        from tuning at random.
        """),
        code("""
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
        ep = np.arange(1, 41)

        # 1. underfitting
        axes[0].plot(ep, 1.2 - 0.25 * np.log1p(ep / 6), label="train")
        axes[0].plot(ep, 1.25 - 0.24 * np.log1p(ep / 6), label="val")
        axes[0].set_title("Underfitting\\nboth high, both flat")

        # 2. overfitting
        axes[1].plot(ep, 1.3 * np.exp(-ep / 7) + 0.02, label="train")
        axes[1].plot(ep, 1.3 * np.exp(-ep / 9) + 0.35 + 0.012 * ep, label="val")
        axes[1].set_title("Overfitting\\ntrain falls, val turns up")

        # 3. healthy
        axes[2].plot(ep, 1.3 * np.exp(-ep / 10) + 0.10, label="train")
        axes[2].plot(ep, 1.3 * np.exp(-ep / 10) + 0.16, label="val")
        axes[2].set_title("Healthy\\nboth fall, small stable gap")

        # 4. LR too high
        rng = np.random.default_rng(0)
        axes[3].plot(ep, 1.0 + 0.45 * rng.normal(size=len(ep)).cumsum() / 6, label="train")
        axes[3].plot(ep, 1.1 + 0.45 * rng.normal(size=len(ep)).cumsum() / 6, label="val")
        axes[3].set_title("Learning rate too high\\nerratic, no trend")

        for a in axes:
            a.set_xlabel("epoch"); a.set_ylabel("loss"); a.legend(fontsize=8); a.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        md("""
        | What you see | Diagnosis | What to do |
        |---|---|---|
        | Both losses high and flat | Underfitting | More capacity, train longer, raise the learning rate |
        | Train falls, validation rises | Overfitting | More data, augmentation (Lecture 7), dropout, weight decay, early stopping |
        | Both fall, gap small and stable | Healthy | Keep going until validation plateaus |
        | Erratic, no trend | Learning rate too high | Lower it, or add warmup |
        | Validation loss rises but accuracy holds | Growing over-confidence | Check calibration (Lecture 8) |
        | Loss becomes `NaN` | Exploding gradients | Gradient clipping, lower learning rate, check for `log(0)` |
        """),
        todo("1", "Optimiser bake-off", """
        Train the same MLP with SGD, SGD+momentum, RMSProp and Adam. Fix the seed and
        every other hyper-parameter. Plot four validation-loss curves on one axis and
        name the winner at 20 epochs.
        """),
        todo_cell(),
        todo("2", "Learning-rate range test", """
        Sweep the learning rate from `1e-5` to `1e0` over one epoch, recording the loss
        after each batch. Plot loss against log learning rate and identify the largest
        rate that is still stable.
        """),
        todo_cell(),
        todo("3", "Initialisation and activation variance", """
        For a 10-layer ReLU MLP initialised with zeros, `N(0, 0.01)`, Xavier and He, plot
        the variance of each layer's activations. Explain why He keeps it flat.
        """),
        todo_cell(),
        todo("4", "Regularisation ablation", """
        Take a model that overfits. Add weight decay, then dropout, then both. Report a
        four-row table of train and validation accuracy and state which intervention paid
        off.
        """),
        todo_cell(),
        todo("5", "BatchNorm in train versus eval", """
        Train a small CNN with BatchNorm. Evaluate the test set once in `train()` mode and
        once in `eval()` mode. Report both numbers and explain the difference in terms of
        batch statistics versus running statistics.
        """),
        todo_cell(),
        todo("6", "Cosine schedule with warmup *(stretch)*", """
        Implement cosine annealing with linear warmup as a `LambdaLR`. Plot the learning
        rate over 50 epochs and compare final accuracy against a constant rate.
        """),
        todo_cell(),
        todo("7", "Batch size and generalisation *(stretch)*", """
        Train at batch sizes 8, 64, 512 and 4096 with the learning rate scaled linearly.
        Report final validation accuracy and comment on the large-batch generalisation gap.
        """),
        todo_cell(),
    ]
