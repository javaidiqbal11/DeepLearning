"""Lecture 07 notebook — augmentation, training recipes, experiment tracking."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("0. Setup", """
        Augmentation only pays off when data is the bottleneck, so this lecture works with
        a deliberately small training subset — 1000 images. That is where the effect is
        visible and where the technique actually matters.
        """),
        code("""
        from dlcourse.data import ArrayImageDataset
        from torch.utils.data import DataLoader, TensorDataset

        data = load_shapes()
        X_train, y_train = data["train"]
        X_val,   y_val   = data["val"]
        X_test,  y_test  = data["test"]
        CLASSES = data["classes"]

        SMALL = 1000
        Xs, ys = X_train[:SMALL], y_train[:SMALL]
        print(f"training on {SMALL} images, validating on {len(X_val)}")
        """),
        section("1. Augmentation, written from tensor operations", """
        No torchvision needed — every transformation here is a few lines on a CHW float
        tensor. Writing them yourself makes the label-preservation question concrete.
        """),
        code('''
        def random_hflip(x, p=0.5):
            """Mirror left-right."""
            return torch.flip(x, dims=[2]) if torch.rand(1).item() < p else x


        def random_vflip(x, p=0.5):
            """Mirror top-bottom. Safe for shapes; NOT safe for digits (6 <-> 9)."""
            return torch.flip(x, dims=[1]) if torch.rand(1).item() < p else x


        def random_crop(x, padding=4):
            """Pad by reflection, then crop back to the original size."""
            c, h, w = x.shape
            xp = F.pad(x.unsqueeze(0), (padding,) * 4, mode="reflect")[0]
            top = torch.randint(0, 2 * padding + 1, (1,)).item()
            left = torch.randint(0, 2 * padding + 1, (1,)).item()
            return xp[:, top:top + h, left:left + w]


        def random_rotation(x, max_deg=20):
            """Rotate via an affine grid."""
            angle = (torch.rand(1).item() * 2 - 1) * max_deg * np.pi / 180
            cos, sin = np.cos(angle), np.sin(angle)
            theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32)
            grid = F.affine_grid(theta.unsqueeze(0), (1, *x.shape), align_corners=False)
            return F.grid_sample(x.unsqueeze(0), grid, align_corners=False,
                                 padding_mode="border")[0]


        def colour_jitter(x, brightness=0.3, contrast=0.3, saturation=0.3):
            """Brightness, contrast and saturation, in that order."""
            if brightness:
                x = x * (1 + (torch.rand(1).item() * 2 - 1) * brightness)
            if contrast:
                mean = x.mean()
                x = (x - mean) * (1 + (torch.rand(1).item() * 2 - 1) * contrast) + mean
            if saturation:
                gray = x.mean(dim=0, keepdim=True)
                x = (x - gray) * (1 + (torch.rand(1).item() * 2 - 1) * saturation) + gray
            return x.clamp(0, 1)


        def random_erasing(x, p=0.5, scale=(0.02, 0.2)):
            """Cutout: occlude a rectangle so the model cannot rely on one region."""
            if torch.rand(1).item() > p:
                return x
            c, h, w = x.shape
            area = h * w * (scale[0] + torch.rand(1).item() * (scale[1] - scale[0]))
            eh = max(1, int(np.sqrt(area)))
            ew = max(1, int(area / eh))
            top = torch.randint(0, max(h - eh, 1), (1,)).item()
            left = torch.randint(0, max(w - ew, 1), (1,)).item()
            x = x.clone()
            x[:, top:top + eh, left:left + ew] = torch.rand(1).item()
            return x


        def compose(*fns):
            def run(x):
                for fn in fns:
                    x = fn(x)
                return x
            return run
        '''),
        code("""
        torch.manual_seed(0)
        original = Xs[1]

        transforms = {
            "original": lambda x: x,
            "hflip": random_hflip,
            "crop": random_crop,
            "rotation": random_rotation,
            "colour jitter": colour_jitter,
            "erasing": random_erasing,
            "all combined": compose(random_hflip, random_crop, random_rotation,
                                    colour_jitter, random_erasing),
        }

        fig, axes = plt.subplots(len(transforms), 8, figsize=(13, 1.65 * len(transforms)))
        for row, (name, fn) in enumerate(transforms.items()):
            for col in range(8):
                out = fn(original)
                axes[row, col].imshow(out.permute(1, 2, 0).clamp(0, 1).numpy())
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(name)
            axes[row, 0].axis("on"); axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
            axes[row, 0].set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
        fig.suptitle(f"Eight samples of each transformation — true class: {CLASSES[ys[1]]}")
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### The label-preservation rule

        A transformation is admissible only if it cannot change the correct label.

        | Transformation | Safe for shapes? | Safe for digits? |
        |---|---|---|
        | Horizontal flip | Yes — a mirrored triangle is a triangle | **No** — mirrored digits are not digits |
        | Vertical flip | Yes | **No** — 6 becomes 9 |
        | Rotation ±20° | Yes | Marginal — large rotations confuse 6/9 |
        | Random crop | Yes, if the shape stays in frame | Yes |
        | Colour jitter | Yes — colour is not the label here | Yes (grayscale anyway) |
        | Erasing | Yes | Yes, in moderation |

        Note the column difference. There is no universal augmentation policy; the policy
        **is** a statement about your data.
        """),
        section("2. Does it actually help?", """
        An ablation: no augmentation, then one transformation at a time, then everything.
        Same model, same seed, same epochs.
        """),
        code("""
        def make_cnn(seed=0):
            torch.manual_seed(seed)
            return nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(64 * 4 * 4, len(CLASSES)),
            )

        def run_with(transform, epochs=20, seed=0):
            set_seed(seed)
            ds = ArrayImageDataset(Xs.numpy(), ys.numpy(), transform=transform)
            dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
            model = make_cnn(seed)
            h = train(model, dl, (X_val, y_val), epochs=epochs, lr=1e-3, verbose=False)
            _, train_acc = evaluate(model, (Xs, ys))
            return model, h, train_acc
        """),
        code("""
        # ~4 minutes for the whole ablation.
        ablation = {}
        policies = {
            "none":           None,
            "+ hflip":        random_hflip,
            "+ crop":         random_crop,
            "+ rotation":     random_rotation,
            "+ colour":       colour_jitter,
            "+ erasing":      random_erasing,
            "all combined":   compose(random_hflip, random_crop, random_rotation,
                                      colour_jitter, random_erasing),
        }

        for name, tf in policies.items():
            model, h, train_acc = run_with(tf)
            _, test_acc = evaluate(model, (X_test, y_test))
            ablation[name] = (train_acc, h.val_acc[-1], test_acc)
            print(f"{name:<14} train {train_acc:.3f}   val {h.val_acc[-1]:.3f}   "
                  f"test {test_acc:.3f}   gap {train_acc - h.val_acc[-1]:+.3f}")
        """),
        code("""
        names = list(ablation)
        train_accs = [ablation[n][0] for n in names]
        val_accs   = [ablation[n][1] for n in names]

        y_pos = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(y_pos - 0.2, train_accs, height=0.38, label="train", color="#bdc3c7")
        ax.barh(y_pos + 0.2, val_accs,   height=0.38, label="validation", color="#1b866b")
        ax.set_yticks(y_pos, names); ax.set_xlim(0, 1.05)
        ax.set_xlabel("accuracy"); ax.legend()
        ax.set_title(f"Augmentation ablation on {SMALL} training images")
        ax.invert_yaxis()
        plt.tight_layout(); plt.show()

        baseline_gap = ablation["none"][0] - ablation["none"][1]
        combined_gap = ablation["all combined"][0] - ablation["all combined"][1]
        print(f"generalisation gap without augmentation: {baseline_gap:+.3f}")
        print(f"generalisation gap with augmentation   : {combined_gap:+.3f}")
        """),
        section("3. Mixup", """
        Mixup trains on convex combinations of pairs of examples, with the labels mixed
        the same way:

        $$\\tilde{x} = \\lambda x_i + (1-\\lambda) x_j, \\qquad
          \\tilde{y} = \\lambda y_i + (1-\\lambda) y_j, \\qquad
          \\lambda \\sim \\text{Beta}(\\alpha, \\alpha)$$

        It encourages the model to behave linearly between training examples, and it
        improves calibration noticeably — which matters for Lecture 8.
        """),
        code("""
        def mixup_batch(x, y, n_classes, alpha=0.2):
            \"\"\"Returns mixed inputs and *soft* labels.\"\"\"
            lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
            idx = torch.randperm(len(x))
            mixed_x = lam * x + (1 - lam) * x[idx]
            onehot = F.one_hot(y, n_classes).float()
            mixed_y = lam * onehot + (1 - lam) * onehot[idx]
            return mixed_x, mixed_y, lam


        def soft_cross_entropy(logits, soft_targets):
            \"\"\"Cross-entropy against a distribution rather than a class index.\"\"\"
            return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


        # See what it produces.
        np.random.seed(3); torch.manual_seed(3)
        mx, my, lam = mixup_batch(Xs[:8], ys[:8], len(CLASSES), alpha=0.4)
        fig, axes = plt.subplots(1, 8, figsize=(13, 2.2))
        for i, ax in enumerate(axes):
            ax.imshow(mx[i].permute(1, 2, 0).clamp(0, 1).numpy()); ax.axis("off")
            top2 = my[i].topk(2)
            ax.set_title(f"{CLASSES[top2.indices[0]][:4]} {top2.values[0]:.2f}\\n"
                         f"{CLASSES[top2.indices[1]][:4]} {top2.values[1]:.2f}", fontsize=8)
        fig.suptitle(f"Mixup, lambda = {lam:.2f} — the labels are mixed too")
        plt.tight_layout(); plt.show()
        """),
        code("""
        def train_mixup(epochs=20, alpha=0.2, seed=0):
            set_seed(seed)
            model = make_cnn(seed)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            ds = TensorDataset(Xs, ys)
            dl = DataLoader(ds, batch_size=64, shuffle=True)
            for _ in range(epochs):
                model.train()
                for xb, yb in dl:
                    mx, my, _ = mixup_batch(xb, yb, len(CLASSES), alpha)
                    opt.zero_grad()
                    soft_cross_entropy(model(mx), my).backward()
                    opt.step()
            return model

        def expected_calibration_error(model, X, y, n_bins=10):
            \"\"\"How far predicted confidence is from observed accuracy.\"\"\"
            model.eval()
            with torch.no_grad():
                probs = F.softmax(model(X), dim=1)
            conf, pred = probs.max(dim=1)
            correct = (pred == y).float()
            ece = 0.0
            for b in range(n_bins):
                lo, hi = b / n_bins, (b + 1) / n_bins
                m = (conf > lo) & (conf <= hi)
                if m.sum() > 0:
                    ece += (m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()).item()
            return ece

        # ~2 minutes.
        plain_model, _, _ = run_with(None)
        mix_model = train_mixup(alpha=0.2)

        for name, m in [("no mixup", plain_model), ("mixup a=0.2", mix_model)]:
            _, acc = evaluate(m, (X_test, y_test))
            ece = expected_calibration_error(m, X_test, y_test)
            print(f"{name:<14} test acc {acc:.3f}   ECE {ece:.4f}")
        print("\\nLower ECE means the confidence numbers can be trusted — see Lecture 8.")
        """),
        section("4. Reproducibility and experiment tracking", """
        A result you cannot reproduce is not a result. The minimum discipline: fix the
        seeds, log the configuration with the metrics, and record the environment.
        """),
        code('''
        import json, hashlib, platform, subprocess, time
        from dataclasses import dataclass, asdict, field


        @dataclass
        class RunConfig:
            lr: float = 1e-3
            batch_size: int = 64
            epochs: int = 20
            dropout: float = 0.0
            weight_decay: float = 0.0
            augment: str = "none"
            seed: int = 0


        class Tracker:
            """Minimal experiment tracker: one JSON file per run."""

            def __init__(self, root):
                self.root = pathlib.Path(root)
                self.root.mkdir(parents=True, exist_ok=True)

            @staticmethod
            def _environment():
                try:
                    commit = subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"],
                        stderr=subprocess.DEVNULL, text=True).strip()
                except Exception:
                    commit = "unknown"
                return {"git_commit": commit,
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                        "platform": platform.platform()}

            def log(self, config: RunConfig, history, extra=None):
                cfg = asdict(config)
                run_id = hashlib.md5(
                    json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
                record = {
                    "run_id": run_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "config": cfg,
                    "environment": self._environment(),
                    "metrics": {
                        "final_train_loss": history.train_loss[-1],
                        "final_val_acc": history.val_acc[-1] if history.val_acc else None,
                        "best_val_acc": history.best()[1],
                        "best_epoch": history.best()[0] + 1,
                        "seconds_per_epoch": float(np.mean(history.epoch_time)),
                    },
                    "curves": {"train_loss": history.train_loss, "val_acc": history.val_acc},
                }
                if extra:
                    record.update(extra)
                (self.root / f"run_{run_id}.json").write_text(json.dumps(record, indent=2))
                return run_id

            def results_table(self):
                import pandas as pd
                rows = []
                for f in sorted(self.root.glob("run_*.json")):
                    r = json.loads(f.read_text())
                    rows.append({**r["config"], **r["metrics"], "run_id": r["run_id"]})
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows).sort_values("best_val_acc", ascending=False)
        '''),
        code("""
        tracker = Tracker(ROOT / "build" / "runs_lecture07")

        set_seed(0)
        cfg = RunConfig(lr=1e-3, epochs=8, augment="none")
        model = make_cnn()
        h = train(model, (Xs, ys), (X_val, y_val), epochs=cfg.epochs, lr=cfg.lr,
                  batch_size=cfg.batch_size, verbose=False)
        run_id = tracker.log(cfg, h)
        print("logged run", run_id)
        print((tracker.root / f"run_{run_id}.json").read_text()[:600], "...")
        """),
        section("5. Random search beats grid search", """
        Bergstra & Bengio's argument: in a grid, every parameter gets the same number of
        distinct values. If only two of your five parameters matter, a 3^5 = 243-point grid
        tries just **3** distinct values of each important one.

        Random search spends the same budget sampling many distinct values of everything.
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
        g = np.linspace(0.08, 0.92, 6)
        gx, gy = np.meshgrid(g, g)
        a1.scatter(gx.ravel(), gy.ravel(), s=26, color="#c0392b")
        a1.set_title(f"Grid search: 36 trials\\nonly {len(g)} distinct values of the important parameter")

        rng_demo = np.random.default_rng(0)
        rx, ry = rng_demo.random(36), rng_demo.random(36)
        a2.scatter(rx, ry, s=26, color="#1b866b")
        a2.set_title("Random search: 36 trials\\n36 distinct values of the important parameter")

        for a in (a1, a2):
            a.set_xlabel("important parameter"); a.set_ylabel("unimportant parameter")
            a.set_xlim(0, 1); a.set_ylim(0, 1); a.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        code("""
        # ~3 minutes for 8 trials.
        N_TRIALS = 8
        rng = np.random.default_rng(0)
        trials = []

        for t in range(N_TRIALS):
            cfg = RunConfig(
                lr=float(10 ** rng.uniform(-4, -2)),           # log-uniform
                weight_decay=float(10 ** rng.uniform(-5, -2)),
                dropout=float(rng.choice([0.0, 0.1, 0.3])),
                augment=str(rng.choice(["none", "crop", "all"])),
                epochs=8, seed=0,
            )
            tf = {"none": None, "crop": random_crop,
                  "all": compose(random_hflip, random_crop, colour_jitter)}[cfg.augment]

            set_seed(cfg.seed)
            ds = ArrayImageDataset(Xs.numpy(), ys.numpy(), transform=tf)
            dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
            model = make_cnn(cfg.seed)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            h = train(model, dl, (X_val, y_val), epochs=cfg.epochs, optimizer=opt, verbose=False)
            tracker.log(cfg, h)
            trials.append((cfg, h.best()[1]))
            print(f"trial {t+1}/{N_TRIALS}  lr={cfg.lr:.1e}  wd={cfg.weight_decay:.1e}  "
                  f"drop={cfg.dropout}  aug={cfg.augment:<5}  best val {h.best()[1]:.3f}")

        best_cfg, best_acc = max(trials, key=lambda p: p[1])
        print(f"\\nbest: val acc {best_acc:.3f} with {best_cfg}")
        """),
        code("""
        table = tracker.results_table()
        print(table[["lr", "weight_decay", "dropout", "augment", "best_val_acc"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        """),
        section("6. Test-time augmentation", """
        Average the prediction over several augmented copies of each test image. It
        reliably buys a little accuracy at a linear cost in inference time.

        State clearly whenever a reported number used TTA — otherwise the comparison
        against a non-TTA number is not honest.
        """),
        code("""
        @torch.no_grad()
        def predict_tta(model, X, n_augs=8, transform=None):
            model.eval()
            probs = F.softmax(model(X), dim=1)          # the un-augmented view
            if transform is None:
                return probs
            for _ in range(n_augs):
                aug = torch.stack([transform(x) for x in X])
                probs = probs + F.softmax(model(aug), dim=1)
            return probs / (n_augs + 1)

        import time
        subset = X_test[:500]; subset_y = y_test[:500]
        tta_transform = compose(random_hflip, random_crop)

        t0 = time.time()
        plain_probs = predict_tta(plain_model, subset, transform=None)
        t_plain = time.time() - t0

        t0 = time.time()
        tta_probs = predict_tta(plain_model, subset, n_augs=8, transform=tta_transform)
        t_tta = time.time() - t0

        acc_plain = (plain_probs.argmax(1) == subset_y).float().mean().item()
        acc_tta   = (tta_probs.argmax(1)   == subset_y).float().mean().item()

        print(f"without TTA : acc {acc_plain:.3f}   {t_plain*1000/len(subset):.2f} ms/image")
        print(f"with TTA x8 : acc {acc_tta:.3f}   {t_tta*1000/len(subset):.2f} ms/image")
        print(f"\\naccuracy change {acc_tta - acc_plain:+.3f} for {t_tta/t_plain:.1f}x the latency")
        """),
        todo("1", "Augmentation from scratch", """
        Implement `random_crop_with_padding`, `random_horizontal_flip`, `random_rotation`
        and `colour_jitter` as functions on CHW float tensors. Show a grid of 16 augmented
        copies of one image.
        """),
        todo_cell(),
        todo("2", "Label-preserving audit", """
        For each of your transformations, state whether it preserves the label for the
        shapes dataset and justify it. Identify one transformation that would break the
        label and explain why.
        """),
        todo_cell(),
        todo("3", "Ablation study", """
        Train with no augmentation, then adding one transformation at a time. Produce a
        table of validation accuracy and identify which transformation contributes most.
        """),
        todo_cell(),
        todo("4", "Mixup", """
        Implement mixup with `Beta(0.2, 0.2)` and the corresponding soft-label
        cross-entropy. Train with it and report the change in validation accuracy **and**
        in expected calibration error.
        """),
        todo_cell(),
        todo("5", "Experiment tracker", """
        Write a `Tracker` class recording config, per-epoch metrics, git commit and
        library versions to a JSON file per run, plus a function loading all runs into a
        sorted results DataFrame.
        """),
        todo_cell(),
        todo("6", "Random search", """
        Run 12 random-search trials over learning rate (log-uniform `1e-4` to `1e-1`),
        weight decay, dropout and augmentation strength. Report the best configuration and
        its validation accuracy.
        """),
        todo_cell(),
        todo("7", "Cutmix *(stretch)*", """
        Implement cutmix — paste a rectangular patch of one image into another and mix the
        labels by patch area. Compare against mixup at equal epoch budget, and show
        example mixed images with their soft labels.
        """),
        todo_cell(),
        todo("8", "Test-time augmentation *(stretch)*", """
        Average predictions over 8 augmented copies at test time. Report the accuracy gain
        and the added latency per image in milliseconds.
        """),
        todo_cell(),
    ]
