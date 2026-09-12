"""Lecture 11 notebook — semantic segmentation with U-Net."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. One prediction per pixel", """
        Segmentation assigns a class to every pixel. Our masks have five classes:
        background plus the four shapes.
        """),
        code("""
        seg = load_segmentation()
        SEG_CLASSES = seg["classes"]
        N_CLASSES = len(SEG_CLASSES)

        # The scenes are 96x96. We work at 48x48 and on a subset of them, because this
        # lecture trains three separate U-Nets to compare losses and run an ablation,
        # and CPU time is the binding constraint. The shapes stay perfectly legible, and
        # every conclusion below holds at full resolution -- it just takes 4x longer.
        SIZE, N_TRAIN = 48, 1200

        def resize(X, M, size=SIZE):
            X = F.interpolate(X, size=(size, size), mode="bilinear", align_corners=False)
            M = F.interpolate(M.unsqueeze(1).float(), size=(size, size),
                              mode="nearest").squeeze(1).long()
            return X, M

        X_train, M_train = resize(*seg["train"])
        X_val,   M_val   = resize(*seg["val"])
        X_train, M_train = X_train[:N_TRAIN], M_train[:N_TRAIN]

        print("images:", tuple(X_train.shape))
        print("masks :", tuple(M_train.shape), " values:", M_train.unique().tolist())
        print("classes:", SEG_CLASSES)
        """),
        code("""
        # Class imbalance is the defining feature of segmentation data.
        pixel_counts = torch.bincount(M_train.flatten(), minlength=N_CLASSES).float()
        fractions = pixel_counts / pixel_counts.sum()

        print(f"{'class':<14}{'pixels':>12}{'fraction':>11}")
        print("-" * 37)
        for name, c, f in zip(SEG_CLASSES, pixel_counts, fractions):
            print(f"{name:<14}{int(c):>12,}{f:>11.4f}")
        print(f"\\nPredicting 'background' everywhere already scores "
              f"{fractions[0]:.1%} pixel accuracy.")
        print("That is why pixel accuracy is not an acceptable metric here.")
        """),
        code("""
        fig, axes = plt.subplots(2, 5, figsize=(14, 5.6))
        for j in range(5):
            axes[0, j].imshow(X_train[j].permute(1, 2, 0).numpy()); axes[0, j].axis("off")
            axes[1, j].imshow(M_train[j], cmap="tab10", vmin=0, vmax=N_CLASSES - 1)
            axes[1, j].axis("off")
        axes[0, 0].set_title("image", fontsize=10); axes[1, 0].set_title("mask", fontsize=10)
        fig.suptitle("Segmentation targets — every pixel is labelled")
        plt.tight_layout(); plt.show()
        """),
        section("2. Why naive upsampling fails", """
        A classification backbone downsamples hard to build semantic depth. Dense
        prediction needs full resolution back. Interpolating a 1/8-scale map straight up
        to full size shows exactly what is lost.
        """),
        code("""
        mask_1 = M_train[0:1].float().unsqueeze(1)
        S = SIZE
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
        axes[0].imshow(M_train[0], cmap="tab10", vmin=0, vmax=N_CLASSES-1)
        axes[0].set_title(f"original {S}x{S}", fontsize=10); axes[0].axis("off")

        for ax, factor in zip(axes[1:], [2, 4, 8, 16]):
            small = F.interpolate(mask_1, scale_factor=1/factor, mode="nearest")
            back = F.interpolate(small, size=(S, S), mode="nearest")
            ax.imshow(back[0, 0], cmap="tab10", vmin=0, vmax=N_CLASSES-1)
            ax.set_title(f"1/{factor} then back up\\n({S//factor}x{S//factor})", fontsize=10)
            ax.axis("off")
        fig.suptitle("What resolution loss costs you — boundaries are the first casualty")
        plt.tight_layout(); plt.show()
        """),
        section("3. U-Net", """
        The answer is an encoder-decoder with **skip connections**: the encoder builds
        semantic content while losing spatial detail, and the skips hand that detail
        directly to the matching decoder stage.

        For upsampling this uses bilinear interpolation followed by a 3x3 convolution
        rather than `ConvTranspose2d`. Transposed convolution produces checkerboard
        artefacts when the kernel size is not divisible by the stride — you will see them
        in the stretch task.
        """),
        code("""
        class DoubleConv(nn.Module):
            \"\"\"(conv -> BN -> ReLU) x 2 — the unit U-Net is built from.\"\"\"

            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)


        class Down(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

            def forward(self, x):
                return self.net(x)


        class Up(nn.Module):
            \"\"\"Upsample, concatenate the skip, then convolve.\"\"\"

            def __init__(self, in_ch, skip_ch, out_ch, use_skip=True):
                super().__init__()
                self.use_skip = use_skip
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                self.conv = DoubleConv(in_ch + (skip_ch if use_skip else 0), out_ch)

            def forward(self, x, skip):
                x = self.up(x)
                if self.use_skip:
                    # Pad if odd input sizes made the shapes disagree by a pixel.
                    dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
                    if dy or dx:
                        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
                    x = torch.cat([skip, x], dim=1)
                return self.conv(x)


        class UNet(nn.Module):
            def __init__(self, n_classes=N_CLASSES, base=16, use_skip=True):
                super().__init__()
                b = base
                # Encoder: three downsamples, so S -> S/2 -> S/4 -> S/8.
                self.inc = DoubleConv(3, b)
                self.d1 = Down(b, b * 2)
                self.d2 = Down(b * 2, b * 4)
                self.d3 = Down(b * 4, b * 8)                  # bottleneck
                # Decoder: three upsamples back to S, each fed by its encoder skip.
                self.u1 = Up(b * 8, b * 4, b * 4, use_skip)
                self.u2 = Up(b * 4, b * 2, b * 2, use_skip)
                self.u3 = Up(b * 2, b, b, use_skip)
                self.outc = nn.Conv2d(b, n_classes, 1)

            def forward(self, x):
                x1 = self.inc(x)
                x2 = self.d1(x1)
                x3 = self.d2(x2)
                x4 = self.d3(x3)
                y = self.u1(x4, x3)
                y = self.u2(y, x2)
                y = self.u3(y, x1)
                return self.outc(y)

        net = UNet()
        out = net(torch.randn(2, 3, SIZE, SIZE))
        print(f"input (2, 3, {SIZE}, {SIZE}) -> output", tuple(out.shape))
        print(f"parameters: {count_parameters(net):,}")
        print("\\nInput and output spatial size match — required for dense prediction.")
        """),
        section("4. Losses for imbalanced dense prediction", """
        Cross-entropy is dominated by background. Dice optimises overlap directly and is
        far less sensitive to imbalance. The reliable default is to use both.
        """),
        code("""
        def dice_loss(logits, target, n_classes=N_CLASSES, eps=1.0):
            \"\"\"Soft Dice over all classes. Differentiable, so it can be a loss.\"\"\"
            probs = F.softmax(logits, dim=1)
            onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
            dims = (0, 2, 3)
            intersection = (probs * onehot).sum(dims)
            cardinality = probs.sum(dims) + onehot.sum(dims)
            dice = (2 * intersection + eps) / (cardinality + eps)
            return 1 - dice.mean()


        def combined_loss(logits, target):
            return F.cross_entropy(logits, target) + dice_loss(logits, target)


        # Class weights inverse to pixel frequency, for weighted cross-entropy.
        weights = 1.0 / (fractions + 1e-6)
        weights = weights / weights.sum() * N_CLASSES
        print("inverse-frequency class weights:")
        for name, w in zip(SEG_CLASSES, weights):
            print(f"  {name:<14}{w:.3f}")
        """),
        section("5. Metrics: IoU and Dice", """
        Accumulate a confusion matrix over the dataset, then derive everything from it.
        Report **per-class** IoU, always — the mean hides a class that has failed
        completely.
        """),
        code("""
        @torch.no_grad()
        def segmentation_metrics(model, X, M, n_classes=N_CLASSES, batch=32):
            model.eval()
            cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
            for i in range(0, len(X), batch):
                pred = model(X[i:i + batch]).argmax(dim=1)
                t = M[i:i + batch].flatten()
                p = pred.flatten()
                idx = t * n_classes + p
                cm += torch.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)

            tp = cm.diag().float()
            fp = cm.sum(0).float() - tp
            fn = cm.sum(1).float() - tp
            iou = tp / (tp + fp + fn).clamp(min=1)
            dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
            pixel_acc = (tp.sum() / cm.sum()).item()
            return {"iou": iou, "dice": dice, "miou": iou.mean().item(),
                    "mdice": dice.mean().item(), "pixel_acc": pixel_acc, "cm": cm}


        def report(metrics, title):
            print(f"\\n{title}")
            print(f"{'class':<14}{'IoU':>8}{'Dice':>8}")
            print("-" * 30)
            for name, i, d in zip(SEG_CLASSES, metrics["iou"], metrics["dice"]):
                print(f"{name:<14}{i:>8.4f}{d:>8.4f}")
            print(f"{'mean':<14}{metrics['miou']:>8.4f}{metrics['mdice']:>8.4f}")
            print(f"pixel accuracy: {metrics['pixel_acc']:.4f}   <- flattering, ignore it")
        """),
        section("6. Training", """
        Three models, three losses, same everything else.
        """),
        code("""
        from torch.utils.data import TensorDataset, DataLoader

        def train_segmenter(loss_fn, epochs=10, use_skip=True, base=16, seed=0, lr=2e-3):
            set_seed(seed)
            model = UNet(base=base, use_skip=use_skip)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loader = DataLoader(TensorDataset(X_train, M_train), batch_size=32, shuffle=True)
            curve = []
            for ep in range(epochs):
                model.train()
                total = 0.0
                for xb, mb in loader:
                    opt.zero_grad()
                    loss = loss_fn(model(xb), mb)
                    loss.backward(); opt.step()
                    total += loss.item() * len(xb)
                curve.append(total / len(X_train))
            return model, curve
        """),
        code("""
        # ~7 minutes for both. This is the longest cell in the course -- two U-Nets
        # trained from scratch. Lower EPOCHS if you need it faster; the ordering of
        # the results holds at 6 epochs, the absolute numbers are just lower.
        EPOCHS = 10
        models, curves, metrics = {}, {}, {}

        for name, fn in [("cross-entropy", F.cross_entropy),
                         ("CE + dice", combined_loss)]:
            m, c = train_segmenter(fn, epochs=EPOCHS)
            models[name], curves[name] = m, c
            metrics[name] = segmentation_metrics(m, X_val, M_val)
            print(f"{name:<15} mIoU {metrics[name]['miou']:.4f}   "
                  f"mDice {metrics[name]['mdice']:.4f}")

        print("\\nTask 4 asks you to add pure Dice loss as a third row.")
        """),
        code("""
        report(metrics["CE + dice"], "CE + Dice — per-class results")
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
        for name, c in curves.items():
            a1.plot(range(1, len(c) + 1), c, marker="o", ms=3, label=name)
        a1.set_xlabel("epoch"); a1.set_ylabel("training loss")
        a1.set_title("Training loss (different scales — compare shape, not height)")
        a1.legend(); a1.grid(alpha=.3)

        width = 0.8 / max(len(metrics), 1)
        x = np.arange(N_CLASSES)
        offset = (len(metrics) - 1) / 2
        for i, (name, m) in enumerate(metrics.items()):
            a2.bar(x + (i - offset) * width, m["iou"].numpy(), width, label=name)
        a2.set_xticks(x, SEG_CLASSES, rotation=20)
        a2.set_ylabel("IoU"); a2.set_title("Per-class IoU by loss function")
        a2.legend(fontsize=9); a2.grid(alpha=.3, axis="y")
        plt.tight_layout(); plt.show()
        """),
        code("""
        best = models["CE + dice"]
        best.eval()
        with torch.no_grad():
            preds = best(X_val[:6]).argmax(dim=1)

        show_masks(X_val[:6], M_val[:6], preds, n=6, classes=SEG_CLASSES)
        plt.show()
        """),
        section("7. What the skip connections are doing", """
        Remove them and retrain. The model keeps the semantics — it still knows *what* is
        in the image — and loses the boundaries.
        """),
        code("""
        # ~1 minute.
        no_skip, _ = train_segmenter(combined_loss, epochs=EPOCHS, use_skip=False)
        m_no_skip = segmentation_metrics(no_skip, X_val, M_val)
        m_skip = metrics["CE + dice"]

        print(f"with skips   : mIoU {m_skip['miou']:.4f}")
        print(f"without skips: mIoU {m_no_skip['miou']:.4f}")
        print(f"difference   : {m_skip['miou'] - m_no_skip['miou']:+.4f}")
        """),
        code("""
        with torch.no_grad():
            p_skip = best(X_val[:4]).argmax(dim=1)
            p_none = no_skip(X_val[:4]).argmax(dim=1)

        fig, axes = plt.subplots(4, 4, figsize=(11, 11))
        titles = ["image", "ground truth", "with skips", "without skips"]
        for i in range(4):
            panels = [X_val[i].permute(1, 2, 0).numpy(), M_val[i], p_skip[i], p_none[i]]
            for j, panel in enumerate(panels):
                axes[i, j].imshow(panel, cmap=None if j == 0 else "tab10",
                                  vmin=None if j == 0 else 0,
                                  vmax=None if j == 0 else N_CLASSES - 1)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(titles[j], fontsize=11)
        fig.suptitle("Skip connections carry the boundaries")
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### The mechanism

        The bottleneck at 12x12 knows there is a triangle and roughly where. It cannot
        know which pixel is the edge — that information was discarded by the pooling
        layers.

        The skip connection hands the decoder the 96x96 feature map from before any
        pooling happened. The decoder combines "what" from below with "where exactly" from
        the side. That is the entire design.
        """),
        section("8. Class weighting for the rare classes", """
        Cross-entropy counts every pixel equally, so background -- 91% of the pixels --
        dominates the gradient. Inverse-frequency weights rebalance it.

        The weights are computed below. Training the weighted model and measuring what
        it changes is **Task 6**: this lecture already trains two U-Nets from scratch,
        and a third belongs in your own run rather than in the worked text.
        """),
        code("""
        # The weights you will use in Task 6, ready to pass straight to cross_entropy.
        print(f"{'class':<14}{'pixel fraction':>16}{'weight':>10}")
        print("-" * 40)
        for name, f, w in zip(SEG_CLASSES, fractions, weights):
            print(f"{name:<14}{f:>16.4f}{w:>10.3f}")

        print("\\nbackground is", f"{fractions[0] / fractions[-1]:.0f}x",
              "more common than the rarest shape, so it gets",
              f"{weights[0] / weights[-1]:.3f}x", "the weight.")
        print("\\nTask 6: train with these weights and report the change in per-class IoU.")
        """),
        todo("1", "U-Net implementation", """
        Implement `DoubleConv`, `Down`, `Up` and `OutConv` blocks and assemble a U-Net with
        depth 3. Verify that input and output spatial dimensions match.
        """),
        todo_cell(),
        todo("2", "Train and visualise", """
        Train for 15 epochs and produce a figure with image, ground-truth mask and
        prediction for six validation scenes.
        """),
        todo_cell(),
        todo("3", "mIoU and Dice from scratch", """
        Implement both metrics with a confusion-matrix accumulator. Report per-class IoU
        and the mean, and state how you handle the background class.
        """),
        todo_cell(),
        todo("4", "Loss comparison", """
        Train three models with cross-entropy, Dice, and CE + Dice. Report mIoU for each
        and state which classes each loss favours.
        """),
        todo_cell(),
        todo("5", "Skip connections ablation", """
        Remove the skip connections and retrain. Report the mIoU drop and show a
        side-by-side prediction demonstrating the loss of boundary detail.
        """),
        todo_cell(),
        todo("6", "Class imbalance", """
        Report the pixel frequency of each class. Apply inverse-frequency weights to
        cross-entropy and report the change in per-class IoU for the rarest class.
        """),
        todo_cell(),
        todo("7", "Checkerboard artefacts *(stretch)*", """
        Replace the bilinear upsampling with `ConvTranspose2d(k=3, stride=2)` and show the
        checkerboard artefacts in the output. Then fix them with `k=4` and explain why
        divisibility of kernel by stride matters.
        """),
        todo_cell(),
        todo("8", "Boundary IoU *(stretch)*", """
        Implement boundary IoU — IoU restricted to a narrow band around object boundaries.
        Report it alongside mIoU and explain what it reveals that mIoU does not.
        """),
        todo_cell(),
    ]
