"""Lecture 08 notebook — evaluation, interpretability, debugging."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("0. A model to interrogate", """
        Train a CNN first — everything in this lecture is about examining a model you
        already have.
        """),
        code("""
        data = load_shapes()
        X_train, y_train = data["train"]
        X_val,   y_val   = data["val"]
        X_test,  y_test  = data["test"]
        CLASSES = data["classes"]

        class SmallCNN(nn.Module):
            \"\"\"Named blocks, because Grad-CAM needs to hook a specific layer.\"\"\"

            def __init__(self, n_classes=4):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
                self.block2 = nn.Sequential(
                    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
                self.block3 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
                # 2x2 spatial pooling rather than global average pooling: shape is a
                # *spatial* property, and GAP throws that away. With GAP this same
                # network plateaus near 49%.
                self.pool = nn.AdaptiveMaxPool2d(2)
                self.fc = nn.Linear(64 * 4, n_classes)

            def forward(self, x):
                x = self.block3(self.block2(self.block1(x)))
                return self.fc(self.pool(x).flatten(1))

        # Deliberately a small training budget. A model at ~93% has real errors to
        # analyse; a model at 100% has nothing to teach about error analysis.
        N_TRAIN = 500
        set_seed(0)
        model = SmallCNN()
        train(model, (X_train[:N_TRAIN], y_train[:N_TRAIN]), (X_val, y_val),
              epochs=6, lr=2e-3, batch_size=64, verbose=False)
        _, acc = evaluate(model, (X_test, y_test))
        print(f"test accuracy: {acc:.4f}  (trained on {N_TRAIN} images)")
        """),
        section("1. Accuracy is one number, and rarely the one you need", """
        Compute precision, recall and F1 by hand first. Knowing what the numbers mean
        matters more than knowing which function returns them.
        """),
        code("""
        @torch.no_grad()
        def get_predictions(model, X, y, batch=256):
            model.eval()
            probs, preds = [], []
            for i in range(0, len(X), batch):
                p = F.softmax(model(X[i:i + batch]), dim=1)
                probs.append(p); preds.append(p.argmax(1))
            return torch.cat(probs), torch.cat(preds), y

        probs, preds, truth = get_predictions(model, X_test, y_test)


        def per_class_metrics(y_true, y_pred, n_classes):
            \"\"\"precision, recall, F1 and support per class, computed directly.\"\"\"
            rows = []
            for c in range(n_classes):
                tp = ((y_pred == c) & (y_true == c)).sum().item()
                fp = ((y_pred == c) & (y_true != c)).sum().item()
                fn = ((y_pred != c) & (y_true == c)).sum().item()
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                rows.append((precision, recall, f1, tp + fn))
            return rows

        rows = per_class_metrics(truth, preds, len(CLASSES))
        print(f"{'class':<10}{'precision':>11}{'recall':>9}{'f1':>8}{'support':>9}")
        print("-" * 47)
        for c, (p, r, f1, sup) in zip(CLASSES, rows):
            print(f"{c:<10}{p:>11.3f}{r:>9.3f}{f1:>8.3f}{sup:>9}")

        macro_f1 = np.mean([r[2] for r in rows])
        micro_acc = (preds == truth).float().mean().item()
        print(f"\\nmacro F1 (classes weighted equally): {macro_f1:.3f}")
        print(f"micro / accuracy (samples weighted): {micro_acc:.3f}")
        """),
        code("""
        # Verify against sklearn -- do this once, then trust your own implementation.
        from sklearn.metrics import classification_report
        print(classification_report(truth.numpy(), preds.numpy(),
                                    target_names=CLASSES, digits=3))
        """),
        md("""
        ### When accuracy actively misleads

        With 99% negatives, "always predict negative" scores 99% accuracy and is useless.
        Macro-averaging weights every class equally and exposes this; micro-averaging
        does not.

        - **Imbalanced classes** → macro F1, or per-class recall
        - **Ranking / threshold-free** → ROC-AUC, or PR-AUC when positives are rare
        - **Asymmetric error costs** → pick the threshold from the cost matrix, not 0.5
        """),
        section("2. The confusion matrix and the error gallery", """
        The confusion matrix shows *which* classes are being confused. The error gallery —
        the mistakes the model was most confident about — shows *why*.
        """),
        code("""
        plot_confusion(truth.numpy(), preds.numpy(), CLASSES,
                       title=f"Confusion matrix — {acc:.1%} accuracy")
        plt.show()
        """),
        code("""
        confidence, _ = probs.max(dim=1)
        wrong = (preds != truth).nonzero(as_tuple=True)[0]
        if len(wrong) == 0:
            print("No errors on the test set. Try a harder subset or fewer training epochs.")
        else:
            order = wrong[confidence[wrong].argsort(descending=True)][:16]
            fig, axes = plt.subplots(2, 8, figsize=(14, 4))
            for ax, i in zip(axes.ravel(), order):
                ax.imshow(X_test[i].permute(1, 2, 0).numpy()); ax.axis("off")
                ax.set_title(f"true {CLASSES[truth[i]][:4]}\\npred {CLASSES[preds[i]][:4]} "
                             f"({confidence[i]:.2f})", fontsize=8)
            for ax in axes.ravel()[len(order):]:
                ax.axis("off")
            fig.suptitle("Most confident mistakes — the most informative images you have")
            plt.tight_layout(); plt.show()
            print(f"{len(wrong)} errors out of {len(truth)}; "
                  f"highest wrong confidence: {confidence[wrong].max():.3f}")
        """),
        section("3. Calibration", """
        A well-calibrated model that says 0.8 is right 80% of the time. Modern networks
        are systematically over-confident, and anything downstream that acts on the
        confidence number needs to know that.
        """),
        code("""
        def reliability_data(probs, truth, n_bins=10):
            conf, pred = probs.max(dim=1)
            correct = (pred == truth).float()
            edges = torch.linspace(0, 1, n_bins + 1)
            bins = []
            ece = 0.0
            for b in range(n_bins):
                m = (conf > edges[b]) & (conf <= edges[b + 1])
                if m.sum() == 0:
                    bins.append((edges[b].item() + 0.5 / n_bins, np.nan, np.nan, 0))
                    continue
                acc_b, conf_b, n = correct[m].mean().item(), conf[m].mean().item(), int(m.sum())
                bins.append((edges[b].item() + 0.5 / n_bins, acc_b, conf_b, n))
                ece += (n / len(conf)) * abs(acc_b - conf_b)
            return bins, ece

        bins, ece = reliability_data(probs, truth)
        print(f"Expected Calibration Error: {ece:.4f}")


        def plot_reliability(bins, ece, title):
            centres = [b[0] for b in bins]
            accs = [b[1] for b in bins]
            counts = [b[3] for b in bins]
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 6),
                                         gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
            a1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
            a1.bar(centres, [0 if np.isnan(a) else a for a in accs], width=0.09,
                   color="#1f6fb2", alpha=.8, label="observed accuracy")
            a1.set_ylabel("accuracy"); a1.set_ylim(0, 1)
            a1.set_title(f"{title}\\nECE = {ece:.4f}"); a1.legend(fontsize=9); a1.grid(alpha=.3)
            a2.bar(centres, counts, width=0.09, color="#7f8c8d")
            a2.set_xlabel("confidence"); a2.set_ylabel("count"); a2.grid(alpha=.3)
            plt.tight_layout()
            return fig

        plot_reliability(bins, ece, "Reliability — before temperature scaling")
        plt.show()
        """),
        code("""
        # Temperature scaling: divide the logits by one learned scalar, fit on validation.
        @torch.no_grad()
        def get_logits(model, X, batch=256):
            model.eval()
            return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)])

        val_logits = get_logits(model, X_val)
        log_T = torch.zeros(1, requires_grad=True)      # optimise log T to keep T > 0
        opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=60)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(val_logits / log_T.exp(), y_val)
            loss.backward()
            return loss

        opt.step(closure)
        T = log_T.exp().item()
        print(f"fitted temperature: T = {T:.3f}")
        print("T > 1 means the model was over-confident and the logits are being softened.")
        """),
        code("""
        test_logits = get_logits(model, X_test)
        probs_scaled = F.softmax(test_logits / T, dim=1)

        bins_after, ece_after = reliability_data(probs_scaled, truth)
        acc_after = (probs_scaled.argmax(1) == truth).float().mean().item()

        print(f"ECE before : {ece:.4f}")
        print(f"ECE after  : {ece_after:.4f}")
        print(f"accuracy before {acc:.4f} -> after {acc_after:.4f}  "
              f"(unchanged: dividing by a constant cannot reorder the logits)")

        plot_reliability(bins_after, ece_after, f"Reliability — after temperature scaling (T={T:.2f})")
        plt.show()
        """),
        section("4. Grad-CAM", """
        Grad-CAM weights the final convolutional feature maps by how much the target class
        score depends on each of them, then sums. The result is coarse — the resolution of
        the last conv layer — but it is class-discriminative and it is robust.
        """),
        code('''
        class GradCAM:
            """Grad-CAM via forward and backward hooks on a chosen layer."""

            def __init__(self, model, target_layer):
                self.model = model
                self.activations = None
                self.gradients = None
                target_layer.register_forward_hook(self._save_activation)
                target_layer.register_full_backward_hook(self._save_gradient)

            def _save_activation(self, module, inp, out):
                self.activations = out.detach()

            def _save_gradient(self, module, grad_in, grad_out):
                self.gradients = grad_out[0].detach()

            def __call__(self, x, class_idx=None):
                self.model.eval()
                self.model.zero_grad()
                logits = self.model(x)
                if class_idx is None:
                    class_idx = logits.argmax(dim=1)
                score = logits[torch.arange(len(x)), class_idx].sum()
                score.backward()

                # One weight per channel: the mean gradient over spatial positions.
                weights = self.gradients.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

                # Normalise each map to [0, 1] for display.
                b = cam.shape[0]
                flat = cam.view(b, -1)
                lo = flat.min(dim=1).values.view(b, 1, 1, 1)
                hi = flat.max(dim=1).values.view(b, 1, 1, 1)
                return ((cam - lo) / (hi - lo + 1e-8))[:, 0], class_idx
        '''),
        code("""
        cam_tool = GradCAM(model, model.block3)

        # One example of each class.
        picks = [int((y_test == c).nonzero()[0]) for c in range(len(CLASSES))]
        batch = X_test[picks]
        cams, cls = cam_tool(batch.clone())

        fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6))
        for j, i in enumerate(picks):
            img = X_test[i].permute(1, 2, 0).numpy()
            axes[0, j].imshow(img); axes[0, j].axis("off")
            axes[0, j].set_title(f"true {CLASSES[y_test[i]]}\\npred {CLASSES[cls[j]]}", fontsize=10)
            axes[1, j].imshow(img); axes[1, j].imshow(cams[j], cmap="jet", alpha=.5)
            axes[1, j].axis("off"); axes[1, j].set_title("Grad-CAM", fontsize=10)
        fig.suptitle("Where the model is looking")
        plt.tight_layout(); plt.show()
        """),
        section("5. Catching a shortcut", """
        This is the important part of the lecture.

        Models exploit whatever correlates with the label. If your data collection put a
        marker on every positive example, the model will read the marker and your
        validation accuracy will look excellent — right up until deployment.

        Build that failure deliberately, then catch it.
        """),
        code("""
        def add_shortcut(X, y, n_classes=4, size=5, strength=1.0):
            \"\"\"Stamp a class-coded coloured square into a corner of every image.\"\"\"
            X = X.clone()
            corners = [(0, 0), (0, 32 - size), (32 - size, 0), (32 - size, 32 - size)]
            for i in range(len(X)):
                c = int(y[i])
                top, left = corners[c % 4]
                colour = torch.zeros(3)
                colour[c % 3] = strength
                X[i, :, top:top + size, left:left + size] = colour.view(3, 1, 1)
            return X

        X_short_train = add_shortcut(X_train[:N_TRAIN], y_train[:N_TRAIN])
        X_short_val   = add_shortcut(X_val, y_val)

        show_grid(X_short_train[:8], y_train[:8], CLASSES, n=8, ncols=8,
                  title="Corrupted training data — note the corner marker")
        plt.show()
        """),
        code("""
        set_seed(0)
        shortcut_model = SmallCNN()
        train(shortcut_model, (X_short_train, y_train[:N_TRAIN]), (X_short_val, y_val),
              epochs=6, lr=2e-3, batch_size=64, verbose=False)

        _, acc_corrupted = evaluate(shortcut_model, (add_shortcut(X_test, y_test), y_test))
        _, acc_clean     = evaluate(shortcut_model, (X_test, y_test))

        print(f"accuracy on corrupted test set (marker present) : {acc_corrupted:.3f}")
        print(f"accuracy on the CLEAN test set (no marker)      : {acc_clean:.3f}")
        print(f"\\ncollapse: {acc_corrupted - acc_clean:+.3f}")
        print("\\nValidation accuracy said the model was excellent. It had learned to read")
        print("a marker that will not exist in production.")
        """),
        code("""
        cam_short = GradCAM(shortcut_model, shortcut_model.block3)
        corrupted_batch = add_shortcut(X_test[picks], y_test[picks])
        cams_s, cls_s = cam_short(corrupted_batch.clone())

        fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6))
        for j in range(len(picks)):
            img = corrupted_batch[j].permute(1, 2, 0).numpy()
            axes[0, j].imshow(img); axes[0, j].axis("off")
            axes[0, j].set_title(f"pred {CLASSES[cls_s[j]]}", fontsize=10)
            axes[1, j].imshow(img); axes[1, j].imshow(cams_s[j], cmap="jet", alpha=.5)
            axes[1, j].axis("off"); axes[1, j].set_title("Grad-CAM", fontsize=10)
        fig.suptitle("The evidence: attention sits on the corner marker, not the shape")
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### The signature of shortcut learning

        Grad-CAM sitting **outside the object** is the tell. Real examples of exactly this
        failure:

        - Pneumonia classifiers reading the hospital's scanner token in the image corner
        - A husky-vs-wolf classifier that had learned to detect snow
        - Models reading dataset watermarks on scraped images

        The defence is not a better architecture. It is: hold out data where the shortcut
        is broken, and look at where the model is looking before you ship.
        """),
        section("6. A debugging protocol", """
        When a model does not work, resist the urge to change the learning rate. Work
        through this in order — it finds the cause far faster than random search.
        """),
        code("""
        # Step 1: can the model overfit a single batch? If not, the bug is in the model,
        # the loss, or the label alignment — not in the hyper-parameters.
        set_seed(0)
        probe = SmallCNN()
        xb, yb = X_train[:16], y_train[:16]
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)

        print("overfitting a single batch of 16:")
        for step in range(120):
            opt.zero_grad()
            loss = F.cross_entropy(probe(xb), yb)
            loss.backward(); opt.step()
            if step % 30 == 0 or step == 119:
                acc_b = (probe(xb).argmax(1) == yb).float().mean().item()
                print(f"  step {step:3d}  loss {loss.item():.5f}  acc {acc_b:.3f}")
        print("\\nLoss near zero and accuracy 1.0 means the plumbing is correct.")
        """),
        code("""
        # Step 2: is the initial loss what theory predicts?
        set_seed(0)
        fresh = SmallCNN()
        with torch.no_grad():
            initial = F.cross_entropy(fresh(X_train[:512]), y_train[:512]).item()
        expected = float(np.log(len(CLASSES)))
        print(f"initial loss : {initial:.4f}")
        print(f"-log(1/{len(CLASSES)})     : {expected:.4f}")
        print(f"difference   : {abs(initial - expected):.4f}")
        print("\\nA large difference means the final layer is badly initialised, or the")
        print("labels are not what you think they are.")
        """),
        md("""
        ### The protocol

        1. **Overfit one batch.** Loss must reach ~0. If not: bug in the model, the loss,
           or label alignment. Nothing else matters until this passes.
        2. **Check the initial loss** equals `-log(1/n_classes)` for balanced classes.
        3. **Verify shapes and label alignment** — print one batch and look at it.
        4. **Check normalisation matches** between training and inference. (Lecture 9.)
        5. **Then** tune: learning rate first, then capacity, then regularisation.

        Steps 1–4 cost ten minutes and find most bugs. Step 5 costs a day and finds none
        of them.
        """),
        todo("1", "Full metric report", """
        Compute per-class precision, recall and F1, plus macro and micro averages, without
        `sklearn.metrics`. Verify against sklearn afterwards.
        """),
        todo_cell(),
        todo("2", "Error gallery", """
        Plot the 16 misclassified test images with the highest predicted confidence,
        captioned with true and predicted labels. Write three sentences on what they have
        in common.
        """),
        todo_cell(),
        todo("3", "Reliability diagram and ECE", """
        Implement a 10-bin reliability diagram and expected calibration error. Report your
        model's ECE.
        """),
        todo_cell(),
        todo("4", "Temperature scaling", """
        Fit a single temperature on the validation set by minimising NLL. Report ECE before
        and after, and confirm accuracy is unchanged.
        """),
        todo_cell(),
        todo("5", "Grad-CAM", """
        Implement Grad-CAM using forward and backward hooks on the last convolutional
        block. Produce overlays for one correct and one incorrect prediction per class.
        """),
        todo_cell(),
        todo("6", "Catch the shortcut", """
        Use `add_shortcut` to corrupt the training data. Train on it, observe the high
        validation accuracy, then use Grad-CAM to prove the model is reading the marker.
        Report accuracy on the clean test set.
        """),
        todo_cell(),
        todo("7", "Occlusion sensitivity *(stretch)*", """
        Implement occlusion sensitivity with a sliding grey patch. Compare its map against
        Grad-CAM on the same image and comment on where they disagree.
        """),
        todo_cell(),
        todo("8", "Saliency sanity check *(stretch)*", """
        Reproduce the Adebayo et al. model-randomisation test: compare saliency from a
        trained model against a randomly initialised one. Report whether your method
        passes.
        """),
        todo_cell(),
    ]
