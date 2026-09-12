"""Lecture 14 notebook — Vision Transformers and self-supervised learning."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. An image as a sequence of patches", """
        ViT's whole trick: cut the image into fixed patches, flatten each, project it
        linearly, and hand the resulting sequence to a standard Transformer encoder.

        A 32x32 image with patch size 4 becomes a sequence of 64 tokens.
        """),
        code("""
        data = load_shapes(normalize=True)
        X_train, y_train = data["train"]
        X_val,   y_val   = data["val"]
        X_test,  y_test  = data["test"]
        CLASSES = data["classes"]
        raw = load_shapes()["train"][0]
        """),
        code("""
        PATCH = 4
        IMG = 32
        N_PATCHES = (IMG // PATCH) ** 2
        print(f"{IMG}x{IMG} image, patch {PATCH} -> {N_PATCHES} patches "
              f"of {PATCH*PATCH*3} values each")

        # Visualise the decomposition.
        img = raw[1]
        fig = plt.figure(figsize=(11, 4))
        a1 = fig.add_subplot(1, 2, 1)
        a1.imshow(img.permute(1, 2, 0).numpy())
        for i in range(0, IMG + 1, PATCH):
            a1.axhline(i - .5, color="w", lw=.8); a1.axvline(i - .5, color="w", lw=.8)
        a1.set_title(f"{IMG}x{IMG} image, cut into {PATCH}x{PATCH} patches")
        a1.axis("off")

        grid = IMG // PATCH
        for r in range(grid):
            for c in range(grid):
                ax = fig.add_subplot(grid, 2 * grid, r * 2 * grid + grid + c + 1)
                ax.imshow(img[:, r*PATCH:(r+1)*PATCH, c*PATCH:(c+1)*PATCH]
                          .permute(1, 2, 0).numpy())
                ax.axis("off")
        fig.suptitle("The Transformer sees the right-hand panel — an unordered set of patches,\\n"
                     "until positional embeddings tell it where each one came from")
        plt.tight_layout(); plt.show()
        """),
        code("""
        class PatchEmbed(nn.Module):
            \"\"\"Patch extraction and projection, done as one strided convolution.

            A conv with kernel = stride = patch size computes exactly 'cut into
            non-overlapping patches, flatten, apply a linear layer'.
            \"\"\"

            def __init__(self, img_size=IMG, patch_size=PATCH, in_ch=3, embed_dim=64):
                super().__init__()
                self.n_patches = (img_size // patch_size) ** 2
                self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)

            def forward(self, x):
                x = self.proj(x)                    # B, D, H/P, W/P
                return x.flatten(2).transpose(1, 2)  # B, N, D

        pe = PatchEmbed(embed_dim=64)
        out = pe(torch.randn(2, 3, 32, 32))
        print("input (2, 3, 32, 32) -> patches", tuple(out.shape))
        print(f"  {out.shape[1]} tokens, each a {out.shape[2]}-dimensional embedding")
        """),
        section("2. The Vision Transformer", """
        Patch embeddings, a learnable `[CLS]` token prepended, positional embeddings added,
        then the same encoder blocks from Lecture 13.
        """),
        code("""
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads, dropout=0.0):
                super().__init__()
                self.d_model, self.n_heads = d_model, n_heads
                self.d_head = d_model // n_heads
                self.qkv = nn.Linear(d_model, 3 * d_model)
                self.proj = nn.Linear(d_model, d_model)
                self.drop = nn.Dropout(dropout)

            def forward(self, x, return_weights=False):
                B, L, D = x.shape
                qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                w = F.softmax(q @ k.transpose(-2, -1) / self.d_head ** 0.5, dim=-1)
                out = (self.drop(w) @ v).transpose(1, 2).reshape(B, L, D)
                out = self.proj(out)
                return (out, w) if return_weights else out


        class Block(nn.Module):
            def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
                super().__init__()
                self.norm1 = nn.LayerNorm(d_model)
                self.attn = MultiHeadAttention(d_model, n_heads, dropout)
                self.norm2 = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(mlp_ratio * d_model, d_model))

            def forward(self, x, return_weights=False):
                if return_weights:
                    a, w = self.attn(self.norm1(x), return_weights=True)
                else:
                    a, w = self.attn(self.norm1(x)), None
                x = x + a
                x = x + self.mlp(self.norm2(x))
                return (x, w) if return_weights else x


        class ViT(nn.Module):
            def __init__(self, img_size=IMG, patch_size=PATCH, embed_dim=64,
                         depth=4, n_heads=4, n_classes=4, dropout=0.1):
                super().__init__()
                self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
                n = self.patch_embed.n_patches

                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
                nn.init.trunc_normal_(self.cls_token, std=0.02)
                nn.init.trunc_normal_(self.pos_embed, std=0.02)

                self.blocks = nn.ModuleList(
                    [Block(embed_dim, n_heads, dropout=dropout) for _ in range(depth)])
                self.norm = nn.LayerNorm(embed_dim)
                self.head = nn.Linear(embed_dim, n_classes)

            def forward_features(self, x, return_weights=False):
                B = x.shape[0]
                x = self.patch_embed(x)
                cls = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls, x], dim=1) + self.pos_embed
                weights = []
                for blk in self.blocks:
                    if return_weights:
                        x, w = blk(x, return_weights=True)
                        weights.append(w)
                    else:
                        x = blk(x)
                return self.norm(x), weights

            def forward(self, x, return_weights=False):
                feats, weights = self.forward_features(x, return_weights)
                logits = self.head(feats[:, 0])       # classify from the [CLS] token
                return (logits, weights) if return_weights else logits

        vit = ViT()
        print("output:", tuple(vit(torch.randn(2, 3, 32, 32)).shape))
        print(f"parameters: {count_parameters(vit):,}")
        """),
        code("""
        # ~2 minutes.
        set_seed(0)
        vit = ViT()
        opt = torch.optim.AdamW(vit.parameters(), lr=1e-3, weight_decay=0.05)
        hist_vit = train(vit, (X_train, y_train), (X_val, y_val),
                         epochs=10, optimizer=opt, batch_size=128)
        _, vit_acc = evaluate(vit, (X_test, y_test))
        print(f"\\nViT test accuracy: {vit_acc:.4f}  ({count_parameters(vit):,} parameters)")
        """),
        section("3. Inductive bias and the data requirement", """
        A CNN knows, by construction, that nearby pixels are related and that a feature is
        the same feature wherever it appears. A ViT knows neither and must learn both from
        data.

        That is why ViT loses to ResNets on ImageNet-1k alone and wins after pre-training
        on 300 million images. Reproduce the shape of that result here, by varying the
        training-set size.
        """),
        code("""
        def small_cnn(n_classes=4, seed=0):
            torch.manual_seed(seed)
            return nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(48 * 16, 64), nn.ReLU(), nn.Linear(64, n_classes))

        print(f"CNN parameters: {count_parameters(small_cnn()):,}")
        print(f"ViT parameters: {count_parameters(ViT()):,}")
        """),
        code("""
        # ~4 minutes. Reduce SIZES if that is too slow.
        SIZES = [250, 1000, 6000]
        curve = {"CNN": [], "ViT": []}

        for n in SIZES:
            Xs, ys = X_train[:n], y_train[:n]
            # Roughly equal *gradient steps* across sizes, capped so the small-n
            # runs do not balloon into hundreds of epochs.
            epochs = min(40, max(10, int(6000 / n * 5)))

            set_seed(0)
            cnn = small_cnn()
            train(cnn, (Xs, ys), epochs=epochs, lr=2e-3, batch_size=64, verbose=False)
            _, a = evaluate(cnn, (X_test, y_test))
            curve["CNN"].append(a)

            set_seed(0)
            v = ViT()
            o = torch.optim.AdamW(v.parameters(), lr=1e-3, weight_decay=0.05)
            train(v, (Xs, ys), epochs=epochs, optimizer=o, batch_size=64, verbose=False)
            _, b = evaluate(v, (X_test, y_test))
            curve["ViT"].append(b)

            print(f"n={n:>5} ({epochs} epochs)   CNN {a:.3f}   ViT {b:.3f}   "
                  f"gap {a - b:+.3f}")
        """),
        code("""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.semilogx(SIZES, curve["CNN"], marker="o", lw=2, label="CNN", color="#1b866b")
        ax.semilogx(SIZES, curve["ViT"], marker="s", lw=2, label="ViT", color="#7a3ea8")
        ax.set_xlabel("training examples"); ax.set_ylabel("test accuracy")
        ax.set_title("Inductive bias is worth most when data is scarce")
        ax.set_xticks(SIZES, [str(s) for s in SIZES])
        ax.legend(); ax.grid(alpha=.3, which="both")
        plt.tight_layout(); plt.show()

        print("The CNN's built-in assumptions about images act like free training data.")
        print("As the real data grows, that advantage shrinks — and at internet scale,")
        print("ViT's freedom from those assumptions starts to pay instead.")
        """),
        section("4. What the attention is looking at", """
        Mean attention distance — how far apart, in pixels, the patches a head connects
        are — is the ViT analogue of receptive field. CNNs start strictly local by
        construction; ViTs choose.
        """),
        code("""
        vit.eval()
        with torch.no_grad():
            _, weights = vit(X_test[:64], return_weights=True)

        grid = IMG // PATCH
        coords = torch.stack(torch.meshgrid(torch.arange(grid), torch.arange(grid),
                                            indexing="ij"), -1).reshape(-1, 2).float() * PATCH
        dist = torch.cdist(coords, coords)              # N x N pixel distances

        print(f"{'layer':<8}{'head 0':>9}{'head 1':>9}{'head 2':>9}{'head 3':>9}{'mean':>9}")
        print("-" * 53)
        layer_means = []
        for li, w in enumerate(weights):
            patch_attn = w[:, :, 1:, 1:].mean(0)        # drop [CLS], average over batch
            per_head = [(patch_attn[h] * dist).sum(-1).mean().item()
                        for h in range(patch_attn.shape[0])]
            layer_means.append(np.mean(per_head))
            print(f"{li+1:<8}" + "".join(f"{d:>9.2f}" for d in per_head)
                  + f"{np.mean(per_head):>9.2f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(layer_means) + 1), layer_means, marker="o", lw=2, color="#7a3ea8")
        ax.set_xlabel("layer"); ax.set_ylabel("mean attention distance (pixels)")
        ax.set_title("How far each layer looks")
        ax.set_xticks(range(1, len(layer_means) + 1)); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        code("""
        # Where the [CLS] token attends — a rough saliency map.
        idx = [int((y_test == c).nonzero()[0]) for c in range(4)]
        raw_test = load_shapes()["test"][0]

        with torch.no_grad():
            _, w = vit(X_test[idx], return_weights=True)

        cls_attn = w[-1][:, :, 0, 1:].mean(1)            # B, N — averaged over heads
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for j, i in enumerate(idx):
            axes[0, j].imshow(raw_test[i].permute(1, 2, 0).numpy())
            axes[0, j].set_title(CLASSES[y_test[i]], fontsize=10); axes[0, j].axis("off")
            heat = cls_attn[j].reshape(grid, grid)
            axes[1, j].imshow(raw_test[i].permute(1, 2, 0).numpy())
            axes[1, j].imshow(F.interpolate(heat[None, None], size=(32, 32),
                                            mode="bilinear", align_corners=False)[0, 0],
                              cmap="jet", alpha=.5)
            axes[1, j].set_title("[CLS] attention", fontsize=10); axes[1, j].axis("off")
        plt.tight_layout(); plt.show()
        """),
        section("5. Self-supervised learning: SimCLR", """
        Labels are expensive; images are not. SimCLR manufactures supervision from the
        data itself: two augmented views of the same image should have similar
        representations, and views of different images should not.

        The augmentation policy is not an implementation detail — it **defines** what the
        model is told to ignore.
        """),
        code("""
        pool = load_ssl_pool()
        X_pool = pool["X"]                     # 4000 unlabelled images
        y_hidden = pool["y_hidden"]            # ONLY for the final linear probe
        print("unlabelled pool:", tuple(X_pool.shape))
        print("labels exist but must not be used during pre-training.")
        """),
        code("""
        def simclr_augment(x):
            \"\"\"Random crop + flip + colour jitter + grayscale. The standard recipe.\"\"\"
            c, h, w = x.shape
            # random resized crop
            scale = 0.5 + 0.5 * torch.rand(1).item()
            size = max(8, int(h * scale))
            top = torch.randint(0, h - size + 1, (1,)).item()
            left = torch.randint(0, w - size + 1, (1,)).item()
            x = x[:, top:top + size, left:left + size]
            x = F.interpolate(x.unsqueeze(0), size=(h, w), mode="bilinear",
                              align_corners=False)[0]
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            # colour jitter
            x = x * (0.6 + 0.8 * torch.rand(1).item())
            mean = x.mean()
            x = (x - mean) * (0.6 + 0.8 * torch.rand(1).item()) + mean
            if torch.rand(1).item() < 0.2:
                x = x.mean(0, keepdim=True).expand(3, -1, -1)
            return x.clamp(0, 1)

        torch.manual_seed(0)
        base = X_pool[3]
        fig, axes = plt.subplots(2, 8, figsize=(14, 3.8))
        axes[0, 0].imshow(base.permute(1, 2, 0).numpy()); axes[0, 0].set_title("original", fontsize=9)
        for j in range(1, 8):
            axes[0, j].imshow(simclr_augment(base).permute(1, 2, 0).numpy())
            axes[0, j].set_title("view A", fontsize=8)
        for j in range(8):
            axes[1, j].imshow(simclr_augment(base).permute(1, 2, 0).numpy())
            axes[1, j].set_title("view B", fontsize=8)
        for ax in axes.ravel():
            ax.axis("off")
        fig.suptitle("Every one of these must map to the same representation")
        plt.tight_layout(); plt.show()
        """),
        code("""
        def nt_xent(z1, z2, temperature=0.5):
            \"\"\"Normalised temperature-scaled cross-entropy.

            For a batch of N, build 2N representations. Each has exactly one positive
            (its other view) and 2N-2 negatives.
            \"\"\"
            N = z1.shape[0]
            z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
            sim = (z @ z.T) / temperature

            # A representation must not count itself as a negative.
            sim.fill_diagonal_(float("-inf"))

            # Positive for index i is i+N (and vice versa).
            targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
            return F.cross_entropy(sim, targets)


        # Sanity check on a batch we know the answer for.
        torch.manual_seed(0)
        perfect = torch.randn(8, 16)
        print(f"loss when the two views are identical : {nt_xent(perfect, perfect.clone()):.4f}")
        print(f"loss when the two views are unrelated : {nt_xent(perfect, torch.randn(8, 16)):.4f}")
        print(f"theoretical minimum for N=8 is near 0, chance is log(2*8-1) = "
              f"{np.log(15):.4f}")
        """),
        code("""
        class Encoder(nn.Module):
            \"\"\"Backbone plus a projection head. The head is discarded afterwards —
            representations *before* the projection transfer better, which was one of
            SimCLR's more surprising findings.\"\"\"

            def __init__(self, feat_dim=64, proj_dim=32):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten())
                self.projection = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, proj_dim))

            def forward(self, x, project=True):
                f = self.backbone(x)
                return self.projection(f) if project else f
        """),
        code("""
        # ~3 minutes. No labels are touched anywhere in this cell.
        set_seed(0)
        encoder = Encoder()
        opt = torch.optim.Adam(encoder.parameters(), lr=2e-3)
        BATCH, EPOCHS = 128, 10
        ssl_losses = []

        for epoch in range(EPOCHS):
            perm = torch.randperm(len(X_pool))
            total, nb = 0.0, 0
            encoder.train()
            for i in range(0, len(X_pool) - BATCH + 1, BATCH):
                batch = X_pool[perm[i:i + BATCH]]
                v1 = torch.stack([simclr_augment(x) for x in batch])
                v2 = torch.stack([simclr_augment(x) for x in batch])
                opt.zero_grad()
                loss = nt_xent(encoder(v1), encoder(v2))
                loss.backward(); opt.step()
                total += loss.item(); nb += 1
            ssl_losses.append(total / nb)
            if (epoch + 1) % 5 == 0:
                print(f"epoch {epoch+1:2d}  NT-Xent loss {ssl_losses[-1]:.4f}")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(range(1, EPOCHS + 1), ssl_losses, marker="o", ms=3, color="#7a3ea8")
        ax.axhline(np.log(2 * BATCH - 1), color="k", ls=":", label="chance")
        ax.set_xlabel("epoch"); ax.set_ylabel("NT-Xent loss")
        ax.set_title("Contrastive pre-training — no labels used")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        section("6. Linear probe: was the representation worth anything?", """
        The standard evaluation. Freeze the encoder, train only a linear classifier on a
        small number of labels, and compare against training the same architecture from
        scratch on those same labels.
        """),
        code("""
        LABEL_BUDGETS = [25, 100, 250]
        probe_results = {"SimCLR + linear probe": [], "supervised from scratch": []}

        for n_labels in LABEL_BUDGETS:
            set_seed(1)
            idx = torch.randperm(len(X_pool))[:n_labels]
            Xl, yl = X_pool[idx], y_hidden[idx]

            # --- frozen encoder + linear probe ---
            encoder.eval()
            with torch.no_grad():
                feats = encoder(Xl, project=False)
                test_feats = encoder(X_test, project=False)
            set_seed(0)
            probe = nn.Linear(feats.shape[1], 4)
            train(probe, (feats, yl), epochs=80, lr=1e-2, batch_size=32, verbose=False)
            _, probe_acc = evaluate(probe, (test_feats, y_test))

            # --- same architecture, trained from scratch on the same labels ---
            set_seed(0)
            scratch = nn.Sequential(Encoder().backbone, nn.Linear(64, 4))
            train(scratch, (Xl, yl), epochs=80, lr=2e-3, batch_size=32, verbose=False)
            _, scratch_acc = evaluate(scratch, (X_test, y_test))

            probe_results["SimCLR + linear probe"].append(probe_acc)
            probe_results["supervised from scratch"].append(scratch_acc)
            print(f"{n_labels:>4} labels   probe {probe_acc:.3f}   scratch {scratch_acc:.3f}"
                  f"   advantage {probe_acc - scratch_acc:+.3f}")
        """),
        code("""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, colour in [("SimCLR + linear probe", "#7a3ea8"),
                             ("supervised from scratch", "#c0392b")]:
            ax.plot(LABEL_BUDGETS, probe_results[name], marker="o", lw=2,
                    label=name, color=colour)
        ax.axhline(0.25, color="k", ls=":", lw=1, label="chance")
        ax.set_xlabel("number of labelled examples"); ax.set_ylabel("test accuracy")
        ax.set_title("Pre-training on unlabelled data, then probing with few labels")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()

        print("The encoder never saw a label during pre-training. A linear classifier on")
        print("top of its frozen features is competitive with — often better than — a")
        print("whole network trained from scratch on the same tiny label budget.")
        """),
        code("""
        # What the representation space looks like, coloured by the labels it never saw.
        encoder.eval()
        with torch.no_grad():
            feats = encoder(X_test[:800], project=False).numpy()

        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(feats)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for c, name in enumerate(CLASSES):
            m = (y_test[:800] == c).numpy()
            ax.scatter(proj[m, 0], proj[m, 1], s=10, label=name, alpha=.7)
        ax.set_title("Self-supervised features, PCA\\n(colours are labels the encoder never saw)")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### Beyond contrastive

        - **BYOL / DINO** — no negatives at all. A momentum-updated teacher plus a
          stop-gradient prevents the collapse everyone expected.
        - **MAE** — mask 75% of patches and reconstruct them. Beautifully simple, scales
          extremely well, and pairs naturally with ViT.

        The high mask ratio in MAE is the crucial hyper-parameter: at 25% you can inpaint
        from neighbouring pixels without understanding anything. At 75% you cannot.
        """),
        todo("1", "Patch embedding", """
        Implement `PatchEmbed` with a strided convolution. For a 32x32 input with patch
        size 4, verify the output is `(B, 64, embed_dim)` and explain the 64.
        """),
        todo_cell(),
        todo("2", "Vision Transformer", """
        Assemble a ViT with the `[CLS]` token, learned positional embeddings and the
        encoder blocks from Lecture 13. Train on shapes and report accuracy and parameter
        count.
        """),
        todo_cell(),
        todo("3", "ViT versus CNN under data scarcity", """
        Train both at matched parameter counts on 500, 2000 and 6000 training examples.
        Plot accuracy against training-set size and explain the gap in terms of inductive
        bias.
        """),
        todo_cell(),
        todo("4", "Attention distance", """
        For each ViT layer, compute the mean attention distance in pixels. Plot it against
        depth and compare the early layers to a CNN's receptive field.
        """),
        todo_cell(),
        todo("5", "SimCLR augmentations and NT-Xent", """
        Implement the two-view augmentation pipeline and the NT-Xent loss with temperature
        0.5. Verify the loss on a hand-constructed batch where you know the right answer.
        """),
        todo_cell(),
        todo("6", "Pre-train and probe", """
        Pre-train the encoder on `shapes_pairs.npz` for 20 epochs using no labels at all.
        Then train a linear probe on 100 labelled examples and compare against supervised
        training on the same 100 examples.
        """),
        todo_cell(),
        todo("7", "Augmentation ablation for SimCLR *(stretch)*", """
        Remove colour jitter, then remove random cropping. Report linear-probe accuracy for
        each and explain why cropping matters most.
        """),
        todo_cell(),
        todo("8", "Masked autoencoder *(stretch)*", """
        Implement a minimal MAE: mask 75% of patches and reconstruct with a light decoder.
        Visualise reconstructions and report linear-probe accuracy against your SimCLR
        result.
        """),
        todo_cell(),
    ]
