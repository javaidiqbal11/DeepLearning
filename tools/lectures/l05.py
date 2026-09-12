"""Lecture 05 notebook — convolutional neural networks."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. Why not an MLP", """
        Start with the arithmetic, because it settles the argument before any accuracy
        number is involved.
        """),
        code("""
        # A modest ImageNet-sized input into one hidden layer.
        H, W, C, hidden = 224, 224, 3, 1000
        mlp_params = H * W * C * hidden

        # A convolution producing 64 channels from a 3x3 patch.
        conv_params = 3 * 3 * C * 64 + 64

        print(f"MLP  224x224x3 -> {hidden:,} units : {mlp_params:,} weights in ONE layer")
        print(f"Conv 3x3, 3 -> 64 channels        : {conv_params:,} weights")
        print(f"ratio                              : {mlp_params / conv_params:,.0f}x")
        print("\\nAnd the convolution's count does not change if the image gets bigger.")
        """),
        section("2. What convolution actually does", """
        Before learning any kernels, apply a few by hand. A convolution is a local
        weighted sum slid across the image — nothing more exotic than that.
        """),
        code("""
        # Standardised inputs, so the CNN and the MLP later get exactly the same
        # treatment — a comparison is only informative if nothing else differs.
        data = load_shapes(normalize=True)
        X_train, y_train = data["train"]
        X_val,   y_val   = data["val"]
        X_test,  y_test  = data["test"]
        CLASSES = data["classes"]

        raw = load_shapes()                      # unstandardised, for display only
        img = raw["train"][0][1:2]               # keep the batch dimension
        gray = img.mean(dim=1, keepdim=True)     # 1x1x32x32

        kernels = {
            "identity":      [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "vertical edge": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],      # Sobel x
            "horizontal edge": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],    # Sobel y
            "blur":          [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]],
            "sharpen":       [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        }

        fig, axes = plt.subplots(1, len(kernels) + 1, figsize=(15, 2.9))
        axes[0].imshow(gray[0, 0], cmap="gray"); axes[0].set_title("input"); axes[0].axis("off")
        for ax, (name, k) in zip(axes[1:], kernels.items()):
            w = torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3)
            out = F.conv2d(gray, w, padding=1)
            ax.imshow(out[0, 0], cmap="gray"); ax.set_title(name, fontsize=10); ax.axis("off")
        fig.suptitle("Hand-designed 3x3 kernels — a CNN learns these in layer 1, unprompted")
        plt.tight_layout(); plt.show()
        """),
        section("3. Convolution arithmetic", """
        You should be able to compute an output shape without running anything:

        $$\\text{out} = \\left\\lfloor \\frac{\\text{in} + 2p - d(k-1) - 1}{s} \\right\\rfloor + 1$$

        where `k` is kernel size, `s` stride, `p` padding and `d` dilation.
        """),
        code("""
        def conv_out(in_size, k, stride=1, padding=0, dilation=1):
            return (in_size + 2 * padding - dilation * (k - 1) - 1) // stride + 1

        configs = [
            ("Conv2d(3,16,3,pad=1)",            dict(k=3, padding=1)),
            ("Conv2d(3,16,3,pad=0)",            dict(k=3)),
            ("Conv2d(3,16,5,stride=2,pad=2)",   dict(k=5, stride=2, padding=2)),
            ("Conv2d(3,16,3,dilation=2,pad=2)", dict(k=3, padding=2, dilation=2)),
            ("Conv2d(3,16,7,stride=2,pad=3)",   dict(k=7, stride=2, padding=3)),
        ]

        print(f"{'layer':<36}{'formula':>9}{'actual':>9}")
        print("-" * 54)
        for name, kw in configs:
            predicted = conv_out(32, **kw)
            layer = nn.Conv2d(3, 16, kw["k"], stride=kw.get("stride", 1),
                              padding=kw.get("padding", 0), dilation=kw.get("dilation", 1))
            actual = layer(torch.zeros(1, 3, 32, 32)).shape[-1]
            flag = "OK" if predicted == actual else "MISMATCH"
            print(f"{name:<36}{predicted:>9}{actual:>9}  {flag}")
        """),
        md("""
        **The rule worth memorising:** for an odd kernel `k` with stride 1, padding
        `p = (k-1)/2` keeps the spatial size unchanged. That is why almost every modern
        architecture uses 3x3 with `padding=1`.
        """),
        section("4. Convolution from scratch", """
        Two implementations: explicit loops, which show what is happening, and `im2col`,
        which is roughly how it is actually done — reshape the problem into one large
        matrix multiplication and let BLAS do the work.
        """),
        code("""
        def conv2d_naive(x, w, b=None, stride=1, padding=0):
            \"\"\"Explicit-loop convolution. Correct, and slow enough to feel it.\"\"\"
            N, C, H, W = x.shape
            F_out, _, KH, KW = w.shape
            xp = F.pad(x, (padding,) * 4)
            H_out = (H + 2 * padding - KH) // stride + 1
            W_out = (W + 2 * padding - KW) // stride + 1

            out = torch.zeros(N, F_out, H_out, W_out)
            for n in range(N):
                for f in range(F_out):
                    for i in range(H_out):
                        for j in range(W_out):
                            patch = xp[n, :, i*stride:i*stride+KH, j*stride:j*stride+KW]
                            out[n, f, i, j] = (patch * w[f]).sum()
            if b is not None:
                out += b.view(1, -1, 1, 1)
            return out


        def conv2d_im2col(x, w, b=None, stride=1, padding=0):
            \"\"\"Unfold every patch into a column, then one matrix multiply.\"\"\"
            N, C, H, W = x.shape
            F_out, _, KH, KW = w.shape
            H_out = (H + 2 * padding - KH) // stride + 1
            W_out = (W + 2 * padding - KW) // stride + 1

            cols = F.unfold(x, (KH, KW), stride=stride, padding=padding)   # N, C*KH*KW, L
            wcol = w.reshape(F_out, -1)                                    # F, C*KH*KW
            out = wcol @ cols                                              # N, F, L
            out = out.reshape(N, F_out, H_out, W_out)
            if b is not None:
                out += b.view(1, -1, 1, 1)
            return out
        """),
        code("""
        import time

        torch.manual_seed(0)
        x = torch.randn(4, 3, 16, 16)
        w = torch.randn(8, 3, 3, 3)
        b = torch.randn(8)

        ref = F.conv2d(x, w, b, stride=1, padding=1)

        t0 = time.time(); naive = conv2d_naive(x, w, b, 1, 1);  t_naive = time.time() - t0
        t0 = time.time(); fast  = conv2d_im2col(x, w, b, 1, 1); t_fast  = time.time() - t0

        print(f"naive  max |Δ| vs F.conv2d : {(naive - ref).abs().max():.2e}   {t_naive*1000:7.1f} ms")
        print(f"im2col max |Δ| vs F.conv2d : {(fast  - ref).abs().max():.2e}   {t_fast*1000:7.1f} ms")
        print(f"\\nim2col speed-up: {t_naive / t_fast:.0f}x")
        print("Both are the same mathematics. One of them is a matrix multiply.")
        """),
        section("5. Pooling", """
        Max pooling takes the strongest response in each window. Its backward pass routes
        the entire gradient to the winning position and nothing anywhere else — a fact
        worth verifying rather than believing.
        """),
        code("""
        small = torch.tensor([[[[1., 3., 2., 4.],
                                [5., 6., 1., 2.],
                                [0., 1., 8., 3.],
                                [2., 1., 4., 7.]]]], requires_grad=True)

        pooled = F.max_pool2d(small, 2)
        print("input:\\n", small[0, 0].detach().numpy())
        print("\\nmax-pooled 2x2:\\n", pooled[0, 0].detach().numpy())

        pooled.sum().backward()
        print("\\ngradient (1 at each window's argmax, 0 elsewhere):\\n",
              small.grad[0, 0].numpy())
        """),
        section("6. Training a CNN", """
        Three conv-ReLU-pool blocks and a dense head. Compare it against the Lecture 1
        baselines on exactly the same test split.
        """),
        code("""
        def make_cnn(seed=0):
            torch.manual_seed(seed)
            return nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 -> 8
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8  -> 4
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
                nn.Linear(128, len(CLASSES)),
            )

        cnn = make_cnn()
        print(cnn)
        print("\\nparameters:", f"{count_parameters(cnn):,}")

        # Trace the shape through the network — do this whenever you build one.
        t = torch.zeros(1, 3, 32, 32)
        print("\\nshape trace:")
        for layer in cnn:
            t = layer(t)
            print(f"  {layer.__class__.__name__:<12} -> {tuple(t.shape)}")
        """),
        code("""
        # ~60 s on a laptop.
        set_seed(0)
        cnn = make_cnn()
        hist = train(cnn, (X_train, y_train), (X_val, y_val), epochs=10, lr=1e-3, batch_size=128)

        test_loss, test_acc = evaluate(cnn, (X_test, y_test))
        print(f"\\nCNN test accuracy: {test_acc:.3f}")
        """),
        code("""
        plot_history(hist, "CNN on shapes")
        plt.show()
        """),
        code("""
        # The honest comparison: same data, same standardisation, same epochs.
        set_seed(0)
        mlp = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32, 256), nn.ReLU(),
                            nn.Linear(256, len(CLASSES)))
        train(mlp, (X_train, y_train), (X_val, y_val), epochs=10, lr=1e-3,
              batch_size=128, verbose=False)
        _, mlp_acc = evaluate(mlp, (X_test, y_test))

        rows = [
            ("chance",               0.25,     0),
            ("logistic (Lecture 1)", 0.633,    3*32*32*4 + 4),
            ("MLP, 256 hidden",      mlp_acc,  count_parameters(mlp)),
            ("CNN",                  test_acc, count_parameters(cnn)),
        ]

        print(f"{'model':<24}{'test acc':>10}{'parameters':>14}")
        print("-" * 48)
        for name, a, params in rows:
            print(f"{name:<24}{a:>10.3f}{params:>14,}")
        print(f"\\nThe CNN uses {count_parameters(mlp)/count_parameters(cnn):.1f}x fewer "
              f"parameters than the MLP.")
        """),
        md("""
        ### Read that table carefully

        On **this** dataset the MLP does well too — the shapes sit near the centre of
        every frame, so a fully-connected layer can memorise "circle-ish pixels live
        here". The accuracy gap is real but modest.

        It would be easy, and dishonest, to stop here and claim CNNs are simply more
        accurate. The genuine advantages are different, and both are measurable:

        1. **Parameter efficiency** — visible in the table above.
        2. **Translation equivariance** — the CNN's actual inductive bias.

        The second one does not show up at all on centred test images. So test for it
        directly.
        """),
        section("6b. The property that actually distinguishes them", """
        Train both on the centred data, then evaluate on test images where the shape has
        been shifted. Nothing about the object changed — only its position.

        A CNN applies the same kernels everywhere, so a shift mostly shifts its features.
        An MLP has a separate weight for every pixel position and has learned nothing
        about the shifted location.
        """),
        code("""
        shifts = [0, 2, 4, 6, 8]
        robustness = {}

        for name, model in [("MLP", mlp), ("CNN", cnn)]:
            accs = []
            for s in shifts:
                X_shifted = torch.roll(X_test, shifts=(s, s), dims=(2, 3))
                _, a = evaluate(model, (X_shifted, y_test))
                accs.append(a)
            robustness[name] = accs
            print(f"{name}: " + "  ".join(f"shift {s}px: {a:.3f}" for s, a in zip(shifts, accs)))
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2),
                                     gridspec_kw={"width_ratios": [1.1, 1]})

        for name, colour in [("MLP", "#e67e22"), ("CNN", "#1b866b")]:
            a1.plot(shifts, robustness[name], marker="o", lw=2, label=name, color=colour)
        a1.set_xlabel("test-image shift (pixels)"); a1.set_ylabel("test accuracy")
        a1.set_title("Accuracy under translation\\n(neither model saw shifted images in training)")
        a1.axhline(0.25, color="k", ls=":", lw=1, label="chance")
        a1.set_ylim(0, 1.05); a1.legend(); a1.grid(alpha=.3)

        # Show what a shifted input actually looks like.
        raw_test = raw["test"][0]
        for j, s in enumerate([0, 4, 8]):
            ax = a2.inset_axes([j * 0.34, 0.25, 0.3, 0.5])
            ax.imshow(torch.roll(raw_test[3], shifts=(s, s), dims=(1, 2)).permute(1, 2, 0).numpy())
            ax.set_title(f"shift {s}px", fontsize=9); ax.axis("off")
        a2.axis("off"); a2.set_title("The same object, moved")
        plt.tight_layout(); plt.show()

        drop_mlp = robustness["MLP"][0] - robustness["MLP"][2]
        drop_cnn = robustness["CNN"][0] - robustness["CNN"][2]
        print(f"accuracy lost at a 4px shift — MLP: {drop_mlp:.3f}   CNN: {drop_cnn:.3f}")
        """),
        md("""
        ### That is the inductive bias, made visible

        A four-pixel shift — an eighth of the image — costs the MLP most of its accuracy
        and costs the CNN far less. Neither model was trained on shifted images; the
        difference comes entirely from how the architecture is wired.

        This is what "the right prior for images" means, concretely. It is also why the
        CNN's advantage would grow, not shrink, at 224x224 where objects genuinely appear
        anywhere in the frame.

        (Neither model is *invariant* — pooling gives only limited tolerance, and by 8px
        both are struggling. Genuine invariance comes from data augmentation, which is
        Lecture 7.)
        """),
        section("7. What the first layer learned", """
        Plot the learned kernels. Compare them against the hand-designed edge detectors
        from section 2 — nobody told the network to find edges.
        """),
        code("""
        w1 = cnn[0].weight.detach()                      # 16, 3, 3, 3
        w1n = (w1 - w1.min()) / (w1.max() - w1.min())

        fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(w1n[i].permute(1, 2, 0).numpy()); ax.axis("off")
            ax.set_title(f"filter {i}", fontsize=8)
        fig.suptitle("Learned first-layer kernels")
        plt.tight_layout(); plt.show()
        """),
        code("""
        # Their responses on one test image.
        test_img = X_test[3:4]                   # standardised, for the model
        display_img = raw["test"][0][3]          # unstandardised, for display
        with torch.no_grad():
            acts = F.relu(cnn[0](test_img))

        fig, axes = plt.subplots(2, 9, figsize=(14, 3.4))
        axes[0, 0].imshow(display_img.permute(1, 2, 0).numpy()); axes[0, 0].set_title("input", fontsize=9)
        axes[1, 0].axis("off")
        for i in range(16):
            ax = axes[(i + 1) // 9, (i + 1) % 9]
            ax.imshow(acts[0, i], cmap="viridis"); ax.set_title(f"ch {i}", fontsize=8)
        for ax in axes.ravel():
            ax.axis("off")
        fig.suptitle(f"Layer-1 activations — true class: {CLASSES[y_test[3]]}")
        plt.tight_layout(); plt.show()
        """),
        section("8. Receptive field", """
        A unit deep in the network sees a region of the input, not a pixel. If that
        region does not cover the object, the network physically cannot classify it.

        Two stacked 3x3 layers see a 5x5 region using 18 parameters per channel pair;
        one 5x5 layer sees the same region using 25. That is why VGG used only 3x3.
        """),
        code("""
        def receptive_field(layers):
            \"\"\"layers: list of (kernel, stride). Returns RF size after each layer.\"\"\"
            rf, jump, out = 1, 1, []
            for k, s in layers:
                rf = rf + (k - 1) * jump
                jump = jump * s
                out.append(rf)
            return out

        arch = [(3, 1), (2, 2), (3, 1), (2, 2), (3, 1), (2, 2)]
        names = ["conv1", "pool1", "conv2", "pool2", "conv3", "pool3"]
        rfs = receptive_field(arch)

        for n, r in zip(names, rfs):
            bar = "#" * min(r, 60)
            print(f"{n:<7} RF = {r:>3} px  {bar}")
        print(f"\\ninput is 32x32; the RF of one unit after pool3 is {rfs[-1]} px.")
        print("That is most of the image but not all of it — a unit in the corner of the")
        print("final feature map still cannot see the opposite corner. The flatten-and-dense")
        print("head is what finally combines all spatial positions.")
        """),
        md("""
        ### Measuring it empirically, and one trap

        The obvious check is to backpropagate from a single output unit and see which
        input pixels receive a non-zero gradient.

        That works for convolutions, but **max pooling breaks the measurement**: it routes
        gradient only to the argmax of each window, so you measure the receptive field of
        one selected path rather than of the architecture. The result is an
        underestimate.

        Swapping max pooling for average pooling — which distributes gradient to every
        element — measures what the formula predicts.
        """),
        code("""
        def empirical_rf(use_avg_pool):
            \"\"\"Backpropagate from one unit and see which input pixels are reachable.

            No ReLU here: the receptive field is a property of the architecture, and a
            ReLU that happens to be inactive at the probed unit would zero the gradient
            and confuse the measurement with a question about activations.
            \"\"\"
            pool = nn.AvgPool2d(2) if use_avg_pool else nn.MaxPool2d(2)
            torch.manual_seed(0)
            probe_net = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1), pool,
                nn.Conv2d(8, 8, 3, padding=1), pool,
            )
            probe = torch.randn(1, 3, 32, 32, requires_grad=True)
            feat = probe_net(probe)
            feat[0, 0, feat.shape[2] // 2, feat.shape[3] // 2].backward()

            influence = (probe.grad[0].abs().sum(0) > 0).float()
            ys, xs = torch.nonzero(influence, as_tuple=True)
            return influence, int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)

        predicted = receptive_field([(3, 1), (2, 2), (3, 1), (2, 2)])[-1]
        inf_max, h_max, w_max = empirical_rf(use_avg_pool=False)
        inf_avg, h_avg, w_avg = empirical_rf(use_avg_pool=True)

        print(f"formula predicts            : {predicted} x {predicted} px")
        print(f"measured with average pool  : {h_avg} x {w_avg} px   <- matches")
        print(f"measured with max pool      : {h_max} x {w_max} px   <- underestimate,")
        print("                                gradient follows only the argmax path")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, inf, title in [(axes[0], inf_avg, f"average pool — {h_avg}x{w_avg}"),
                               (axes[1], inf_max, f"max pool — {h_max}x{w_max}")]:
            ax.imshow(inf, cmap="Greys"); ax.set_title(title, fontsize=10); ax.axis("off")
        fig.suptitle("Input pixels that can influence one mid-network unit")
        plt.tight_layout(); plt.show()
        """),
        todo("1", "Shape arithmetic without running code", """
        For a 32x32x3 input, compute the output shape **and parameter count** for:
        `Conv2d(3,16,3,pad=1)`; `MaxPool2d(2)`; `Conv2d(16,32,5,stride=2,pad=2)`;
        `Conv2d(32,32,3,dilation=2,pad=2)`. Then verify each with a forward pass.
        """),
        todo_cell(),
        todo("2", "Convolution from scratch", """
        Implement `conv2d_naive` and `conv2d_im2col` yourself. Both must match
        `F.conv2d` to within `1e-5`. Report the im2col speed-up.
        """),
        todo_cell(),
        todo("3", "Max pooling forward and backward", """
        Implement max pooling and its backward pass without using
        `F.max_pool2d`. Verify the gradient routes entirely to the argmax position.
        """),
        todo_cell(),
        todo("4", "Train a CNN on shapes", """
        Build a three-block CNN and train it to at least 97% test accuracy. Add the
        result to your running `results.md` table beside the Lecture 1 baselines.
        """),
        todo_cell(),
        todo("5", "Parameter accounting", """
        Compare your CNN against an MLP that reaches similar accuracy. Report parameter
        counts for both and explain the ratio in terms of weight sharing.
        """),
        todo_cell(),
        todo("6", "Visualise layer-1 filters", """
        Plot all first-layer kernels as RGB images. Identify at least two that respond to
        oriented edges and show their activation maps on one test image.
        """),
        todo_cell(),
        todo("7", "Receptive field calculator *(stretch)*", """
        Write `receptive_field(layers)` returning the RF at each layer for a list of
        `(kernel, stride, dilation)` tuples. Verify empirically by finding which input
        pixels affect one output unit.
        """),
        todo_cell(),
        todo("8", "Translation equivariance test *(stretch)*", """
        Shift a test image by 1, 4 and 8 pixels. Measure how the CNN's and the MLP's
        predictions change, and quantify the difference.
        """),
        todo_cell(),
    ]
