"""Lecture 16 notebook — multimodal learning and production ML systems."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. A CLIP-style dual encoder", """
        CLIP trains an image encoder and a text encoder so that matched image-text pairs
        land close together in a shared embedding space, and mismatched pairs land far
        apart.

        The payoff is **zero-shot classification**: to classify into new categories you
        just write them down as text. No retraining.
        """),
        code("""
        data = load_shapes(normalize=True)
        X_img, y_img = data["train"]
        X_test, y_test = data["test"]
        CLASSES = data["classes"]
        caption_rows = load_captions()

        print(f"{len(caption_rows)} image-caption pairs")
        for idx, cap in caption_rows[:5]:
            print(f"  image {idx:>4} ({CLASSES[y_img[idx]]:<8}): {cap}")
        """),
        code("""
        # Word-level vocabulary over the captions.
        words = sorted({w for _, c in caption_rows for w in c.lower().split()})
        vocab = ["<pad>"] + words
        stoi = {w: i for i, w in enumerate(vocab)}
        MAX_LEN = max(len(c.split()) for _, c in caption_rows)

        def encode_text(text, max_len=MAX_LEN):
            ids = [stoi.get(w, 0) for w in text.lower().split()][:max_len]
            return ids + [0] * (max_len - len(ids))

        pair_img_idx = torch.tensor([r[0] for r in caption_rows])
        pair_text = torch.tensor([encode_text(r[1]) for r in caption_rows])
        pair_images = X_img[pair_img_idx]
        pair_labels = y_img[pair_img_idx]

        print(f"vocabulary {len(vocab)} words, max caption length {MAX_LEN}")
        print("images:", tuple(pair_images.shape), " text:", tuple(pair_text.shape))
        """),
        code("""
        EMBED = 64

        class ImageEncoder(nn.Module):
            def __init__(self, dim=EMBED):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.AdaptiveMaxPool2d(2), nn.Flatten(), nn.Linear(64 * 4, dim))

            def forward(self, x):
                return F.normalize(self.net(x), dim=-1)     # project onto the unit sphere


        class TextEncoder(nn.Module):
            def __init__(self, vocab_size, dim=EMBED, embed=48):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
                self.fc = nn.Sequential(nn.Linear(embed, dim), nn.ReLU(), nn.Linear(dim, dim))

            def forward(self, ids):
                mask = (ids != 0).unsqueeze(-1).float()
                pooled = (self.embed(ids) * mask).sum(1) / mask.sum(1).clamp(min=1)
                return F.normalize(self.fc(pooled), dim=-1)


        class DualEncoder(nn.Module):
            def __init__(self, vocab_size, dim=EMBED):
                super().__init__()
                self.image_encoder = ImageEncoder(dim)
                self.text_encoder = TextEncoder(vocab_size, dim)
                # Learned temperature, stored as a log so it stays positive.
                self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

            def forward(self, images, texts):
                zi = self.image_encoder(images)
                zt = self.text_encoder(texts)
                scale = self.logit_scale.exp().clamp(max=100)
                return scale * zi @ zt.T          # image-text similarity matrix
        """),
        md("""
        ### Symmetric InfoNCE

        For a batch of `N` pairs, the similarity matrix is `N x N`. The diagonal holds the
        matched pairs; everything off-diagonal is a negative.

        Read each **row** as "which caption belongs to this image" and each **column** as
        "which image belongs to this caption". Both are `N`-way classification problems
        whose correct answer is the diagonal — so the loss is cross-entropy in both
        directions, averaged.
        """),
        code("""
        def clip_loss(logits):
            \"\"\"Symmetric cross-entropy over the image-text similarity matrix.\"\"\"
            targets = torch.arange(len(logits), device=logits.device)
            loss_i2t = F.cross_entropy(logits, targets)     # rows
            loss_t2i = F.cross_entropy(logits.T, targets)   # columns
            return (loss_i2t + loss_t2i) / 2


        # Sanity check against a case with a known answer.
        perfect = torch.eye(8) * 20
        chance = torch.zeros(8, 8)
        print(f"loss for a perfectly aligned batch : {clip_loss(perfect):.6f}")
        print(f"loss for a uniform (no signal) batch: {clip_loss(chance):.6f}")
        print(f"chance is log(8) = {np.log(8):.6f}")
        """),
        code("""
        # ~3 minutes.
        from torch.utils.data import TensorDataset, DataLoader

        set_seed(0)
        clip = DualEncoder(len(vocab))
        opt = torch.optim.AdamW(clip.parameters(), lr=2e-3, weight_decay=0.01)
        loader = DataLoader(TensorDataset(pair_images, pair_text), batch_size=128,
                            shuffle=True, drop_last=True)

        losses = []
        for epoch in range(25):
            clip.train(); total, nb = 0.0, 0
            for imgs, txts in loader:
                opt.zero_grad()
                loss = clip_loss(clip(imgs, txts))
                loss.backward(); opt.step()
                total += loss.item(); nb += 1
            losses.append(total / nb)
            if (epoch + 1) % 5 == 0:
                print(f"epoch {epoch+1:2d}  loss {losses[-1]:.4f}  "
                      f"temperature {1/clip.logit_scale.exp().item():.4f}")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(losses, color="#7a3ea8")
        ax.axhline(np.log(128), color="k", ls=":", label="chance (batch 128)")
        ax.set_xlabel("epoch"); ax.set_ylabel("symmetric InfoNCE")
        ax.set_title("Contrastive vision-language training")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        section("2. Zero-shot classification", """
        No classifier head was ever trained. Embed the class names as text prompts, embed
        the image, and take the nearest.
        """),
        code("""
        @torch.no_grad()
        def zero_shot(model, images, class_names, template="a photo of a {}"):
            model.eval()
            prompts = torch.tensor([encode_text(template.format(c)) for c in class_names])
            text_emb = model.text_encoder(prompts)               # C, D
            img_emb = model.image_encoder(images)                # N, D
            return (img_emb @ text_emb.T).argmax(dim=1)

        pred = zero_shot(clip, X_test, CLASSES)
        zs_acc = (pred == y_test).float().mean().item()
        print(f"zero-shot accuracy: {zs_acc:.4f}")
        print("The model was never trained on a classification objective.")
        """),
        code("""
        plot_confusion(y_test.numpy(), pred.numpy(), CLASSES,
                       title=f"Zero-shot classification — {zs_acc:.1%}")
        plt.show()
        """),
        code("""
        # Prompt sensitivity — a real and underrated deployment risk.
        templates = [
            "a photo of a {}",
            "{}",
            "a {}",
            "picture of a {}",
            "an image containing a single {}",
            "this is a {} shape",
        ]
        prompt_accs = {}
        for t in templates:
            p = zero_shot(clip, X_test, CLASSES, template=t)
            prompt_accs[t] = (p == y_test).float().mean().item()
            print(f'{t:<36} {prompt_accs[t]:.4f}')

        spread = max(prompt_accs.values()) - min(prompt_accs.values())
        print(f"\\nspread across prompts: {spread:.4f}")

        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.barh(list(prompt_accs), list(prompt_accs.values()), color="#7a3ea8")
        ax.set_xlabel("zero-shot accuracy"); ax.set_xlim(0, 1.05)
        ax.set_title("The same model, the same images, different wording")
        ax.invert_yaxis()
        plt.tight_layout(); plt.show()
        print("\\nIf your deployed accuracy depends on a wording choice, that wording is a")
        print("hyper-parameter and belongs in version control with everything else.")
        """),
        section("3. Retrieval in both directions", """
        A shared embedding space gives text-to-image and image-to-text search for free.
        """),
        code("""
        @torch.no_grad()
        def retrieve(model, query_text, images, k=8):
            model.eval()
            q = model.text_encoder(torch.tensor([encode_text(query_text)]))
            emb = model.image_encoder(images)
            sims = (q @ emb.T)[0]
            top = sims.topk(k)
            return top.indices, top.values

        raw_test = load_shapes()["test"][0]
        queries = ["a circle in the middle of the frame", "picture of a star",
                   "a single triangle, centred"]

        fig, axes = plt.subplots(len(queries), 6, figsize=(12, 2.2 * len(queries)))
        for r, q in enumerate(queries):
            idx, sims = retrieve(clip, q, X_test, k=6)
            for c in range(6):
                axes[r, c].imshow(raw_test[idx[c]].permute(1, 2, 0).numpy())
                axes[r, c].axis("off")
                axes[r, c].set_title(f"{CLASSES[y_test[idx[c]]]}\\n{sims[c]:.3f}", fontsize=8)
            axes[r, 0].set_ylabel(q, fontsize=8)
        fig.suptitle("Text-to-image retrieval — query on the left, top 6 results")
        plt.tight_layout(); plt.show()
        """),
        code("""
        @torch.no_grad()
        def recall_at_k(model, images, texts, ks=(1, 5)):
            \"\"\"Recall@K in both directions over a held-out set of pairs.\"\"\"
            model.eval()
            zi = model.image_encoder(images)
            zt = model.text_encoder(texts)
            sims = zi @ zt.T
            n = len(sims)
            target = torch.arange(n)
            out = {}
            for k in ks:
                i2t = (sims.topk(k, dim=1).indices == target[:, None]).any(1).float().mean()
                t2i = (sims.T.topk(k, dim=1).indices == target[:, None]).any(1).float().mean()
                out[f"image->text R@{k}"] = i2t.item()
                out[f"text->image R@{k}"] = t2i.item()
            return out

        # Note: many captions are identical strings, so exact-pair recall understates
        # how well retrieval works semantically. Report it honestly anyway.
        sample = torch.randperm(len(pair_images))[:256]
        for name, v in recall_at_k(clip, pair_images[sample], pair_text[sample]).items():
            print(f"{name:<20} {v:.4f}")
        print(f"\\nchance at R@1 for 256 candidates: {1/256:.4f}")
        """),
        section("4. A model registry", """
        Now the production half. A registry versions every model artefact with its metrics,
        config and approval state — so "which model produced this prediction" always has an
        answer.
        """),
        code('''
        import json, hashlib, time
        from dataclasses import dataclass, asdict, field

        REGISTRY = ROOT / "build" / "registry"


        @dataclass
        class ModelRecord:
            version: str
            created_at: str
            metrics: dict
            config: dict
            stage: str = "staging"          # staging -> production -> archived
            notes: str = ""


        class ModelRegistry:
            """Versioned model store with an explicit promotion step."""

            def __init__(self, root=REGISTRY):
                self.root = pathlib.Path(root)
                (self.root / "artefacts").mkdir(parents=True, exist_ok=True)
                self.index_path = self.root / "index.json"
                if not self.index_path.exists():
                    self.index_path.write_text("{}")

            def _index(self):
                return json.loads(self.index_path.read_text())

            def _write(self, idx):
                self.index_path.write_text(json.dumps(idx, indent=2))

            def register(self, model, version, metrics, config, notes=""):
                torch.save(model.state_dict(), self.root / "artefacts" / f"{version}.pt")
                idx = self._index()
                idx[version] = asdict(ModelRecord(
                    version=version,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    metrics=metrics, config=config, notes=notes))
                self._write(idx)
                return version

            def promote(self, version):
                """Exactly one version is in production at a time."""
                idx = self._index()
                if version not in idx:
                    raise KeyError(f"unknown version {version}")
                for v, rec in idx.items():
                    if rec["stage"] == "production":
                        rec["stage"] = "archived"
                idx[version]["stage"] = "production"
                self._write(idx)

            def production(self):
                for v, rec in self._index().items():
                    if rec["stage"] == "production":
                        return rec
                return None

            def list_models(self):
                return sorted(self._index().values(), key=lambda r: r["created_at"])
        '''),
        code("""
        registry = ModelRegistry()

        # Register two versions: a weak model and a better one.
        set_seed(0)
        weak = ImageEncoder()
        weak_head = nn.Sequential(weak, nn.Linear(EMBED, 4))
        train(weak_head, (X_img[:300], y_img[:300]), epochs=4, lr=2e-3, verbose=False)
        _, weak_acc = evaluate(weak_head, (X_test, y_test))

        set_seed(0)
        strong = ImageEncoder()
        strong_head = nn.Sequential(strong, nn.Linear(EMBED, 4))
        train(strong_head, (X_img, y_img), epochs=6, lr=2e-3, verbose=False)
        _, strong_acc = evaluate(strong_head, (X_test, y_test))

        registry.register(weak_head, "0.9.0", {"test_accuracy": round(weak_acc, 4)},
                          {"n_train": 300, "epochs": 4}, notes="first attempt")
        registry.register(strong_head, "1.0.0", {"test_accuracy": round(strong_acc, 4)},
                          {"n_train": len(X_img), "epochs": 6}, notes="full training set")
        registry.promote("1.0.0")

        print(f"{'version':<10}{'stage':<12}{'test acc':>10}   notes")
        print("-" * 55)
        for rec in registry.list_models():
            print(f"{rec['version']:<10}{rec['stage']:<12}"
                  f"{rec['metrics']['test_accuracy']:>10.4f}   {rec['notes']}")
        print(f"\\nserving: version {registry.production()['version']}")
        """),
        section("5. Drift detection", """
        The hard truth of production ML: labels arrive late, or never. You usually cannot
        measure accuracy live.

        What you *can* measure without any labels is whether the **distribution** of
        inputs or predictions has moved. The Population Stability Index quantifies that:

        $$\\text{PSI} = \\sum_i (p_i - q_i)\\ln\\frac{p_i}{q_i}$$

        Rules of thumb: below 0.1 stable, 0.1-0.25 investigate, above 0.25 act.
        """),
        code("""
        def population_stability_index(baseline, current, eps=1e-6):
            \"\"\"PSI between two discrete distributions (they must be normalised).\"\"\"
            p = np.asarray(baseline, dtype=float) + eps
            q = np.asarray(current, dtype=float) + eps
            p, q = p / p.sum(), q / q.sum()
            return float(np.sum((p - q) * np.log(p / q)))


        @torch.no_grad()
        def prediction_distribution(model, X, n_classes=4):
            model.eval()
            preds = model(X).argmax(1)
            return (torch.bincount(preds, minlength=n_classes).float() / len(preds)).numpy()


        baseline = prediction_distribution(strong_head, X_test)
        print("baseline prediction distribution:")
        for c, f in zip(CLASSES, baseline):
            print(f"  {c:<10} {f:.4f}")
        """),
        code("""
        # Simulate the corruptions a real deployment actually meets.
        def corrupt(X, kind, severity):
            if kind == "gaussian noise":
                return (X + torch.randn_like(X) * severity * 2).clamp(-5, 5)
            if kind == "brightness":
                return X + severity * 3
            if kind == "blur":
                k = torch.ones(3, 1, 3, 3) / 9
                out = X
                for _ in range(max(1, int(severity * 6))):
                    out = F.conv2d(out, k, padding=1, groups=3)
                return out
            if kind == "class shift":
                # Only circles arrive — the world changed, the model did not.
                return X[(y_test == 0)][:len(X)]
            return X

        rows = []
        for kind in ["gaussian noise", "brightness", "blur", "class shift"]:
            for sev in ([0.2, 0.5, 1.0] if kind != "class shift" else [1.0]):
                Xc = corrupt(X_test, kind, sev)
                dist = prediction_distribution(strong_head, Xc)
                psi = population_stability_index(baseline, dist)
                if kind == "class shift":
                    acc = float("nan")
                else:
                    _, acc = evaluate(strong_head, (Xc, y_test))
                status = "OK" if psi < 0.1 else ("INVESTIGATE" if psi < 0.25 else "ALERT")
                rows.append((kind, sev, psi, acc, status))
                print(f"{kind:<16} sev {sev:<5} PSI {psi:>7.4f}  "
                      f"acc {acc:>6.3f}  {status}")
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))

        labels = [f"{k}\\n{s}" for k, s, _, _, _ in rows]
        psis = [r[2] for r in rows]
        colours = ["#1b866b" if p < 0.1 else "#e67e22" if p < 0.25 else "#c0392b" for p in psis]
        a1.bar(range(len(rows)), psis, color=colours)
        a1.axhline(0.10, color="#e67e22", ls="--", label="investigate (0.10)")
        a1.axhline(0.25, color="#c0392b", ls="--", label="alert (0.25)")
        a1.set_xticks(range(len(rows)), labels, rotation=60, ha="right", fontsize=7)
        a1.set_ylabel("PSI"); a1.set_title("Drift signal — needs no labels")
        a1.legend(fontsize=8)

        valid = [(r[2], r[3]) for r in rows if not np.isnan(r[3])]
        a2.scatter([v[0] for v in valid], [v[1] for v in valid], s=60, color="#7a3ea8")
        a2.set_xlabel("PSI (measurable in production)")
        a2.set_ylabel("accuracy (usually NOT measurable in production)")
        a2.set_title("PSI is a usable proxy for the thing you cannot see")
        a2.grid(alpha=.3)
        plt.tight_layout(); plt.show()

        if len(valid) > 2:
            corr = np.corrcoef([v[0] for v in valid], [v[1] for v in valid])[0, 1]
            print(f"correlation between PSI and accuracy: {corr:.3f}")
            print("Negative and strong: as drift rises, accuracy falls. That is exactly")
            print("what makes PSI worth alerting on.")
        """),
        section("6. Pre-release audit", """
        Before anything ships: robustness across corruptions, calibration, and a written
        record of what the model is and is not for.
        """),
        code("""
        corruptions = ["gaussian noise", "brightness", "blur"]
        severities = [0.2, 0.5, 1.0]
        grid = np.zeros((len(corruptions), len(severities)))

        for i, kind in enumerate(corruptions):
            for j, sev in enumerate(severities):
                _, a = evaluate(strong_head, (corrupt(X_test, kind, sev), y_test))
                grid[i, j] = a

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(severities)), [str(s) for s in severities])
        ax.set_yticks(range(len(corruptions)), corruptions)
        ax.set_xlabel("severity"); ax.set_title("Robustness benchmark")
        for i in range(len(corruptions)):
            for j in range(len(severities)):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, label="accuracy")
        plt.tight_layout(); plt.show()

        worst = np.unravel_index(grid.argmin(), grid.shape)
        print(f"clean accuracy: {strong_acc:.4f}")
        print(f"weakest axis  : {corruptions[worst[0]]} at severity "
              f"{severities[worst[1]]} -> {grid.min():.4f}")
        """),
        code("""
        # A model card is written before release, not after an incident.
        # Built from a list of lines so the f-string expressions stay readable.
        robustness_rows = [
            f"  - {k} at severity {s}: {grid[i, j]:.4f}"
            for i, k in enumerate(corruptions)
            for j, s in enumerate(severities)
        ]

        card_lines = [
            "# Model Card - Shape Classifier",
            "",
            "## Overview",
            "- **Version:** 1.0.0",
            "- **Type:** 4-class image classifier (circle, square, triangle, star)",
            "- **Input:** 32x32 RGB, standardised with the training-split statistics",
            "- **Output:** probability distribution over 4 classes",
            "",
            "## Intended use",
            "Classifying a single centred geometric shape on a pale background, inside",
            "the DL-601 teaching pipeline.",
            "",
            "## Out-of-scope use",
            "- Photographs, or any natural image",
            "- Images containing more than one shape (use the Lecture 10 detector)",
            "- Shapes outside the four training classes. There is no 'unknown' output:",
            "  the model will confidently pick one of the four.",
            "- Any decision that affects a person",
            "",
            "## Training data",
            f"- {len(X_img)} synthetic images from `tools/build_data.py` (seeded)",
            "- Balanced across the four classes",
            "- Colour is randomised and deliberately decorrelated from the label",
            "",
            "## Evaluation",
            f"- Clean test accuracy: **{strong_acc:.4f}** on {len(X_test)} held-out images",
            "- Robustness under corruption:",
            *robustness_rows,
            f"- Weakest axis: {corruptions[worst[0]]} at severity {severities[worst[1]]}",
            "",
            "## Known limitations",
            "- Synthetic training data; no claim of transfer to photographs",
            "- No out-of-distribution detection. Confidence stays high on inputs the",
            "  model has never seen anything resembling.",
            "- Calibration was measured on in-distribution data only",
            "",
            "## Monitoring",
            "- `/metrics` exposes the prediction distribution and latency percentiles",
            "- PSI against the training baseline; alert above 0.25",
            "- Review on: PSI alert, p99 latency above 200 ms, or error rate above 1%",
            "",
            "## Rollback",
            "`POST /models/<previous_version>/promote`. The previous artefact stays in",
            "the registry and is never deleted on promotion.",
            "",
        ]

        card = "\\n".join(card_lines)
        card_path = ROOT / "build" / "MODEL_CARD.md"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(card, encoding="utf-8")
        print(card)
        """),
        md("""
        ### What this course did not cover, and where to go next

        - **Scaling laws** — loss as a predictable function of compute, data and parameters.
        - **Efficient adaptation** — LoRA, adapters, prompt tuning: change 0.1% of the
          weights instead of all of them.
        - **Efficiency** — quantisation, distillation, pruning. Usually the difference
          between a demo and a product.
        - **RLHF and preference optimisation** — how a language model is aligned after
          pre-training.
        - **Open problems** — reasoning, sample efficiency, calibrated uncertainty,
          verifiable alignment.

        The foundations you now have — gradients, convolution, attention, and the
        discipline of measuring things honestly — are what every one of those is built on.
        """),
        todo("1", "Dual encoder", """
        Implement an image encoder (small CNN) and a text encoder (embedding + mean
        pooling, or a small Transformer) projecting into a shared 64-dimensional
        L2-normalised space.
        """),
        todo_cell(),
        todo("2", "Symmetric contrastive loss", """
        Implement InfoNCE in both directions with a learnable temperature. Verify that a
        perfectly aligned batch gives near-zero loss.
        """),
        todo_cell(),
        todo("3", "Zero-shot classification", """
        Classify test images using only the text prompts `"a photo of a {class}"`. Report
        accuracy and compare against the supervised Lecture 6 model.
        """),
        todo_cell(),
        todo("4", "Prompt sensitivity", """
        Try five different prompt templates. Report the accuracy spread and comment on what
        it implies for deploying a zero-shot system.
        """),
        todo_cell(),
        todo("5", "Bidirectional retrieval", """
        Implement text-to-image and image-to-text retrieval. Report Recall@1 and Recall@5
        for both directions.
        """),
        todo_cell(),
        todo("6", "Model registry and serving", """
        Build a registry that versions checkpoints with their metrics and config. Serve the
        latest approved version from FastAPI with `GET /models` and
        `POST /models/{version}/promote`. Extend the Lecture 9 service.
        """),
        todo_cell(),
        todo("7", "Drift detection", """
        Implement a `/metrics` endpoint tracking the prediction distribution and PSI against
        the training baseline. Feed it corrupted images and show the alarm fire.
        """),
        todo_cell(),
        todo("8", "Model card", """
        Write a complete model card: intended use, out-of-scope use, training data,
        evaluation results, subgroup performance, calibration, known limitations and the
        rollback procedure.
        """),
        todo_cell(),
        todo("9", "Robustness benchmark *(stretch)*", """
        Evaluate under five corruptions (Gaussian noise, blur, brightness, contrast,
        rotation) at three severities. Produce a 5x3 accuracy table and identify the
        weakest axis.
        """),
        todo_cell(),
        todo("10", "Quantisation *(stretch)*", """
        Apply dynamic quantisation to the served model. Report model size, p95 latency and
        accuracy before and after, and state whether you would ship it.
        """),
        todo_cell(),
    ]
