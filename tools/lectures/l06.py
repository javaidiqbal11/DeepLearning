"""Lecture 06 notebook — modern architectures and transfer learning."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("0. Setup"),
        code("""
        data = load_shapes(normalize=True)
        X_train, y_train = data["train"]
        X_val,   y_val   = data["val"]
        X_test,  y_test  = data["test"]
        CLASSES = data["classes"]
        print("train", tuple(X_train.shape), "classes", CLASSES)
        """),
        section("1. The degradation problem", """
        In 2015 He et al. reported something that looked like a bug: a 56-layer plain
        network had **higher training error** than a 20-layer one.

        That is not overfitting — overfitting would show as higher *test* error with lower
        training error. It is an optimisation failure. The deeper network could in
        principle learn the identity in its extra layers and match the shallow one
        exactly, and it fails to find that solution.

        Reproduce it here, at a scale that runs on a laptop.
        """),
        code("""
        class PlainBlock(nn.Module):
            \"\"\"conv -> BN -> ReLU -> conv -> BN -> ReLU. No skip.\"\"\"

            def __init__(self, channels):
                super().__init__()
                self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                self.b1 = nn.BatchNorm2d(channels)
                self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                self.b2 = nn.BatchNorm2d(channels)

            def forward(self, x):
                out = F.relu(self.b1(self.c1(x)))
                return F.relu(self.b2(self.c2(out)))


        class ResidualBlock(nn.Module):
            \"\"\"The same, plus the identity shortcut: y = ReLU(F(x) + x).\"\"\"

            def __init__(self, channels):
                super().__init__()
                self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                self.b1 = nn.BatchNorm2d(channels)
                self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                self.b2 = nn.BatchNorm2d(channels)

            def forward(self, x):
                out = F.relu(self.b1(self.c1(x)))
                out = self.b2(self.c2(out))
                return F.relu(out + x)          # <- the entire idea


        def deep_net(n_blocks, residual, width=24, seed=0):
            torch.manual_seed(seed)
            Block = ResidualBlock if residual else PlainBlock
            return nn.Sequential(
                nn.Conv2d(3, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(),
                *[Block(width) for _ in range(n_blocks)],
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(width, len(CLASSES)),
            )
        """),
        code("""
        # ~2 minutes. Four networks, deliberately restricted to a small subset so the
        # optimisation difficulty — not the data — is what we are measuring.
        SUB = 1500
        Xs, ys = X_train[:SUB], y_train[:SUB]
        EPOCHS = 6

        degradation = {}
        for residual in (False, True):
            for depth in (3, 8):
                set_seed(0)
                model = deep_net(depth, residual)
                h = train(model, (Xs, ys), (X_val, y_val), epochs=EPOCHS,
                          lr=1e-3, batch_size=128, verbose=False)
                _, train_acc = evaluate(model, (Xs, ys))
                key = ("residual" if residual else "plain", depth * 2 + 2)
                degradation[key] = (h.train_loss[-1], train_acc, h.val_acc[-1])
                print(f"{key[0]:<9} {key[1]:>2} layers   "
                      f"train_loss {h.train_loss[-1]:.4f}   "
                      f"train_acc {train_acc:.3f}   val_acc {h.val_acc[-1]:.3f}")
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for i, kind in enumerate(["plain", "residual"]):
            depths = [d for (k, d) in degradation if k == kind]
            losses = [degradation[(kind, d)][0] for d in depths]
            accs   = [degradation[(kind, d)][2] for d in depths]
            a1.plot(depths, losses, marker="o", lw=2, label=kind)
            a2.plot(depths, accs,   marker="o", lw=2, label=kind)
        a1.set_xlabel("depth (layers)"); a1.set_ylabel("final TRAINING loss")
        a1.set_title("Training loss vs depth\\n(plain gets worse — that is degradation)")
        a2.set_xlabel("depth (layers)"); a2.set_ylabel("validation accuracy")
        a2.set_title("Validation accuracy vs depth")
        for a in (a1, a2):
            a.grid(alpha=.3); a.legend()
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### Why the shortcut fixes it

        A plain block must learn a mapping `H(x)` from scratch. If the best thing it could
        do is nothing at all, it has to learn the identity — and stacked non-linear layers
        are surprisingly bad at representing the identity exactly.

        A residual block computes `H(x) = F(x) + x`. To do nothing it only needs
        `F(x) = 0`, which is easy: drive the weights toward zero. **The identity is the
        default behaviour, and the block learns the deviation from it.**

        The second benefit is the one you measured in Lecture 3: `d(x + F(x))/dx = 1 + dF/dx`.
        That `1` gives gradients a path to the early layers that skips all the
        multiplications.
        """),
        section("2. A proper ResNet", """
        Real ResNets change channel count and spatial size as they go, so the shortcut
        needs a 1x1 projection whenever the shapes do not line up. That detail is where
        most from-scratch implementations go wrong.
        """),
        code("""
        class BasicBlock(nn.Module):
            \"\"\"ResNet basic block with an optional projection shortcut.\"\"\"
            expansion = 1

            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)

                # Identity only works when shape and channels match; otherwise project.
                self.shortcut = nn.Identity()
                if stride != 1 or in_ch != out_ch:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_ch),
                    )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return F.relu(out + self.shortcut(x))


        # Check both paths.
        same = BasicBlock(16, 16, stride=1)
        down = BasicBlock(16, 32, stride=2)
        t = torch.randn(2, 16, 32, 32)
        print("identity shortcut :", tuple(t.shape), "->", tuple(same(t).shape))
        print("projection shortcut:", tuple(t.shape), "->", tuple(down(t).shape))
        print("\\nshortcut type when shapes match    :", type(same.shortcut).__name__)
        print("shortcut type when they do not     :", type(down.shortcut).__name__)
        """),
        code("""
        class SmallResNet(nn.Module):
            \"\"\"ResNet-style network sized for 32x32 inputs.\"\"\"

            def __init__(self, blocks_per_stage=2, widths=(16, 32, 64), n_classes=4):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
                    nn.BatchNorm2d(widths[0]), nn.ReLU(),
                )
                stages, in_ch = [], widths[0]
                for i, w in enumerate(widths):
                    for b in range(blocks_per_stage):
                        stride = 2 if (b == 0 and i > 0) else 1
                        stages.append(BasicBlock(in_ch, w, stride))
                        in_ch = w
                self.stages = nn.Sequential(*stages)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, n_classes))

            def forward(self, x):
                return self.head(self.stages(self.stem(x)))

        set_seed(0)
        resnet = SmallResNet()
        print(f"parameters: {count_parameters(resnet):,}")
        print("output shape:", tuple(resnet(torch.randn(2, 3, 32, 32)).shape))
        """),
        code("""
        # ~2 minutes.
        set_seed(0)
        resnet = SmallResNet()
        hist = train(resnet, (X_train, y_train), (X_val, y_val),
                     epochs=6, lr=2e-3, batch_size=128)
        _, resnet_acc = evaluate(resnet, (X_test, y_test))
        print(f"\\nSmallResNet test accuracy: {resnet_acc:.4f}  "
              f"({count_parameters(resnet):,} parameters)")
        """),
        code("""
        plot_history(hist, "SmallResNet on shapes")
        plt.show()
        """),
        section("3. Transfer learning", """
        The real test of transfer learning is a target task with very few labels — that is
        the situation where it matters.

        The setup: pre-train on **circle vs square**, then transfer to **triangle vs star**
        with only 100 labelled examples. Three strategies, same budget.
        """),
        code("""
        # Split the dataset into a source task and a target task with disjoint classes.
        src_mask_tr = (y_train == 0) | (y_train == 1)      # circle, square
        tgt_mask_tr = (y_train == 2) | (y_train == 3)      # triangle, star
        tgt_mask_te = (y_test == 2) | (y_test == 3)

        Xsrc, ysrc = X_train[src_mask_tr], y_train[src_mask_tr]          # labels 0/1
        Xtgt_all, ytgt_all = X_train[tgt_mask_tr], y_train[tgt_mask_tr] - 2
        Xtgt_te,  ytgt_te  = X_test[tgt_mask_te],  y_test[tgt_mask_te] - 2

        N_LABELS = 100
        set_seed(0)
        pick = torch.randperm(len(Xtgt_all))[:N_LABELS]
        Xtgt, ytgt = Xtgt_all[pick], ytgt_all[pick]

        print(f"source task (circle vs square) : {len(Xsrc)} examples")
        print(f"target task (triangle vs star) : {len(Xtgt)} labelled, "
              f"{len(Xtgt_te)} test")
        """),
        code("""
        # Pre-train the source model. ~60 s.
        set_seed(0)
        source_model = SmallResNet(n_classes=2)
        train(source_model, (Xsrc, ysrc), epochs=5, lr=2e-3, batch_size=128, verbose=False)
        _, src_acc = evaluate(source_model, (Xsrc, ysrc))
        print(f"source task accuracy: {src_acc:.3f}")
        """),
        code("""
        import copy

        def target_model_from(source, mode):
            \"\"\"mode: 'scratch' | 'frozen' | 'finetune'.\"\"\"
            if mode == "scratch":
                set_seed(1)
                return SmallResNet(n_classes=2), 2e-3

            model = copy.deepcopy(source)
            model.head[-1] = nn.Linear(model.head[-1].in_features, 2)   # fresh head
            if mode == "frozen":
                for p in model.stem.parameters():
                    p.requires_grad = False
                for p in model.stages.parameters():
                    p.requires_grad = False
                return model, 2e-3
            return model, 2e-4          # fine-tune: 10x lower learning rate

        transfer_results = {}
        for mode in ("scratch", "frozen", "finetune"):
            set_seed(2)
            model, lr = target_model_from(source_model, mode)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.Adam(params, lr=lr)
            train(model, (Xtgt, ytgt), epochs=25, optimizer=opt,
                  batch_size=32, verbose=False)
            _, acc = evaluate(model, (Xtgt_te, ytgt_te))
            transfer_results[mode] = acc
            trainable = sum(p.numel() for p in params)
            print(f"{mode:<10} target test acc {acc:.3f}   ({trainable:,} trainable parameters)")
        """),
        code("""
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        labels = ["from scratch", "frozen backbone\\n+ new head", "full fine-tune\\n(lr/10)"]
        vals = [transfer_results[m] for m in ("scratch", "frozen", "finetune")]
        bars = ax.barh(labels, vals, color=["#c0392b", "#e67e22", "#1b866b"])
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=11)
        ax.set_xlim(0, 1.05); ax.set_xlabel("target-task test accuracy")
        ax.set_title(f"Transfer learning with only {N_LABELS} labelled target examples")
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### Choosing a strategy

        | Target data | Domain distance | What to do |
        |---|---|---|
        | Small | Close | Freeze the backbone, train a linear head |
        | Large | Close | Fine-tune everything |
        | Small | Far | Fine-tune the middle layers; very late features may not transfer |
        | Large | Far | Fine-tuning still usually converges faster than random init |

        The learning rate for fine-tuning is conventionally 10x lower than you would use
        from scratch. The pre-trained weights are already good; a large learning rate
        destroys them in the first few batches — a failure mode with its own name,
        *catastrophic forgetting*.
        """),
        section("4. How much should you unfreeze?", """
        Not a binary choice. Sweep it.
        """),
        code("""
        def unfreeze_last_k(source, k):
            model = copy.deepcopy(source)
            model.head[-1] = nn.Linear(model.head[-1].in_features, 2)
            blocks = list(model.stages)
            for p in model.stem.parameters():
                p.requires_grad = False
            for i, block in enumerate(blocks):
                trainable = i >= len(blocks) - k
                for p in block.parameters():
                    p.requires_grad = trainable
            return model

        n_blocks = len(list(source_model.stages))
        sweep = {}
        for k in range(0, n_blocks + 1):
            set_seed(3)
            model = unfreeze_last_k(source_model, k)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.Adam(params, lr=5e-4)
            train(model, (Xtgt, ytgt), epochs=25, optimizer=opt, batch_size=32, verbose=False)
            _, acc = evaluate(model, (Xtgt_te, ytgt_te))
            sweep[k] = (acc, sum(p.numel() for p in params))
            print(f"unfrozen blocks: {k}/{n_blocks}   acc {acc:.3f}   "
                  f"trainable {sweep[k][1]:,}")
        """),
        code("""
        ks = list(sweep)
        accs = [sweep[k][0] for k in ks]
        trainables = [sweep[k][1] for k in ks]

        fig, ax1 = plt.subplots(figsize=(8, 4.2))
        ax1.plot(ks, accs, marker="o", color="#1b866b", lw=2)
        ax1.set_xlabel("number of unfrozen backbone blocks")
        ax1.set_ylabel("target test accuracy", color="#1b866b")
        ax1.grid(alpha=.3)
        ax2 = ax1.twinx()
        ax2.plot(ks, trainables, marker="s", color="#7f8c8d", ls="--")
        ax2.set_ylabel("trainable parameters", color="#7f8c8d")
        ax1.set_title("How much to unfreeze — accuracy against cost")
        plt.tight_layout(); plt.show()
        """),
        section("5. Architecture building blocks worth knowing", """
        Two ideas you will meet constantly: the 1x1 bottleneck, and depthwise separable
        convolution. Both are about buying the same receptive field for fewer operations.
        """),
        code("""
        def count(m):
            return sum(p.numel() for p in m.parameters())

        C = 256
        plain = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.Conv2d(C, C, 3, padding=1))
        bottleneck = nn.Sequential(
            nn.Conv2d(C, C // 4, 1),                    # reduce
            nn.Conv2d(C // 4, C // 4, 3, padding=1),    # process cheaply
            nn.Conv2d(C // 4, C, 1),                    # expand
        )
        depthwise = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C),    # one filter per channel
            nn.Conv2d(C, C, 1),                         # mix channels
        )

        print(f"{'block':<32}{'parameters':>14}{'ratio':>10}")
        print("-" * 56)
        base = count(plain)
        for name, m in [("plain 3x3 + 3x3", plain),
                        ("1x1 -> 3x3 -> 1x1 bottleneck", bottleneck),
                        ("depthwise separable", depthwise)]:
            print(f"{name:<32}{count(m):>14,}{count(m)/base:>10.2f}x")

        x = torch.randn(1, C, 16, 16)
        for name, m in [("plain", plain), ("bottleneck", bottleneck), ("depthwise", depthwise)]:
            print(f"\\n{name} output: {tuple(m(x).shape)}")
        """),
        todo("1", "Residual block from scratch", """
        Implement `BasicBlock` with two 3x3 convolutions, BatchNorm and a skip connection,
        including the 1x1 projection needed when channel count or stride changes. Verify
        output shapes for both cases.
        """),
        todo_cell(),
        todo("2", "Reproduce the degradation problem", """
        Train a 20-layer and a 40-layer plain CNN. Show the deeper one has higher
        **training** loss. Add skip connections to both and show the ordering reverses.
        """),
        todo_cell(),
        todo("3", "Small ResNet on shapes", """
        Assemble a ResNet-style network from your blocks. Reach at least 99% test accuracy
        and report parameter count and training time against the Lecture 5 CNN.
        """),
        todo_cell(),
        todo("4", "Feature extraction versus fine-tuning", """
        Pre-train on circle/square only. Transfer to a triangle/star task with just 100
        labelled examples, three ways: from scratch, frozen backbone, full fine-tuning at
        `lr/10`. Report all three accuracies.
        """),
        todo_cell(),
        todo("5", "How many layers to unfreeze", """
        Sweep the number of unfrozen backbone blocks from 0 to all. Plot target-task
        accuracy against that number and state where the curve flattens.
        """),
        todo_cell(),
        todo("6", "1x1 bottleneck cost analysis *(stretch)*", """
        Compare parameters and FLOPs for a plain 3x3-3x3 block against a 1x1-3x3-1x1
        bottleneck of equal input/output width. Report the ratio and verify with a
        forward-pass timing.
        """),
        todo_cell(),
        todo("7", "Depthwise separable convolution *(stretch)*", """
        Implement it and substitute it into your ResNet. Report the accuracy lost and the
        parameters saved.
        """),
        todo_cell(),
    ]
