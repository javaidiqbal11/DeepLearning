"""Lecture 10 notebook — object detection."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. The detection task", """
        Classification answers "what". Detection answers "what, and exactly where" — and
        the number of answers varies per image, which a fixed-size classification head
        simply cannot express.
        """),
        code("""
        det = load_detection()
        X_train_np, ann_train = det["train"]
        X_val_np,   ann_val   = det["val"]
        DET_CLASSES = det["classes"]

        print("train scenes:", X_train_np.shape, " val:", X_val_np.shape)
        print("classes:", DET_CLASSES)

        counts = np.bincount([len(a["boxes"]) for a in ann_train])
        for n, c in enumerate(counts):
            if c:
                print(f"  scenes with {n} object(s): {c}")
        """),
        code("""
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
        for ax, i in zip(axes.ravel(), range(8)):
            show_boxes(X_train_np[i], ann_train[i]["boxes"], ann_train[i]["labels"],
                       DET_CLASSES, ax=ax)
        fig.suptitle("Detection scenes — 96x96, one to three objects each")
        plt.tight_layout(); plt.show()
        """),
        section("2. Intersection over Union", """
        IoU is the matching criterion the whole field is built on: a prediction counts as
        correct if it overlaps the truth by more than a threshold.
        """),
        code("""
        def box_iou(boxes_a, boxes_b):
            \"\"\"Pairwise IoU. boxes are [x1, y1, x2, y2].

            Returns an (N, M) matrix. Vectorised — never loop over boxes.
            \"\"\"
            a = torch.as_tensor(boxes_a, dtype=torch.float32).reshape(-1, 4)
            b = torch.as_tensor(boxes_b, dtype=torch.float32).reshape(-1, 4)

            area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
            area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

            lt = torch.max(a[:, None, :2], b[None, :, :2])      # top-left of overlap
            rb = torch.min(a[:, None, 2:], b[None, :, 2:])      # bottom-right of overlap
            wh = (rb - lt).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]

            union = area_a[:, None] + area_b[None, :] - inter
            return inter / union.clamp(min=1e-9)


        # Tests, because an IoU bug silently ruins every metric downstream.
        box = [[0., 0., 10., 10.]]
        checks = [
            ("identical",          [[0., 0., 10., 10.]],   1.0),
            ("disjoint",           [[20., 20., 30., 30.]], 0.0),
            ("contained (quarter)",[[0., 0., 5., 5.]],     0.25),
            ("half overlap",       [[5., 0., 15., 10.]],   1/3),
            ("touching edges",     [[10., 0., 20., 10.]],  0.0),
        ]
        for name, other, expected in checks:
            got = box_iou(box, other).item()
            flag = "OK" if abs(got - expected) < 1e-6 else "FAIL"
            print(f"{name:<22} expected {expected:.4f}  got {got:.4f}  {flag}")
        """),
        section("3. Non-maximum suppression", """
        A detector emits many overlapping boxes for a single object. NMS keeps the
        highest-scoring one and deletes anything that overlaps it too much.
        """),
        code("""
        def nms(boxes, scores, iou_threshold=0.5):
            \"\"\"Greedy non-maximum suppression. Returns kept indices, best score first.\"\"\"
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
            order = scores.argsort(descending=True)

            keep = []
            while order.numel() > 0:
                i = order[0].item()
                keep.append(i)
                if order.numel() == 1:
                    break
                ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
                order = order[1:][ious <= iou_threshold]
            return torch.tensor(keep, dtype=torch.long)


        def batched_nms(boxes, scores, labels, iou_threshold=0.5):
            \"\"\"Class-wise NMS: two genuinely overlapping objects of different classes
            must not suppress each other.\"\"\"
            keep_all = []
            labels = torch.as_tensor(labels)
            for c in labels.unique():
                idx = (labels == c).nonzero(as_tuple=True)[0]
                kept = nms(boxes[idx], scores[idx], iou_threshold)
                keep_all.append(idx[kept])
            return torch.cat(keep_all) if keep_all else torch.tensor([], dtype=torch.long)
        """),
        code("""
        # A worked example you can verify by eye.
        demo_boxes = torch.tensor([
            [10., 10., 50., 50.],    # score 0.95 — the winner
            [12., 12., 52., 52.],    # heavy overlap with the winner -> suppressed
            [11., 9.,  49., 51.],    # heavy overlap -> suppressed
            [60., 60., 90., 90.],    # separate object -> kept
            [62., 58., 92., 88.],    # overlaps the fourth -> suppressed
        ])
        demo_scores = torch.tensor([0.95, 0.88, 0.71, 0.90, 0.62])

        kept = nms(demo_boxes, demo_scores, 0.5)
        print("kept indices:", kept.tolist(), "(expected [0, 3])")

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.5))
        blank = np.full((96, 96, 3), 240, np.uint8)
        show_boxes(blank, demo_boxes, [0]*5, None, ax=a1, title=f"Before NMS — {len(demo_boxes)} boxes")
        show_boxes(blank, demo_boxes[kept], [0]*len(kept), None, ax=a2,
                   title=f"After NMS — {len(kept)} boxes")
        plt.tight_layout(); plt.show()
        """),
        section("4. An anchor-free detector", """
        Anchors involve a lot of hyper-parameter tuning that obscures the idea. This
        detector, in the spirit of CenterNet and FCOS, predicts:

        - a **centre heatmap**: one channel per class, peaking at object centres
        - a **size map**: two channels giving width and height at each location

        Decoding is then: find peaks, read the size at each peak, convert to a box.
        """),
        code("""
        GRID = 24                  # 96 / 4 — the detector works at stride 4
        STRIDE = 96 // GRID
        N_CLASSES = len(DET_CLASSES)

        def make_targets(annotations, grid=GRID, stride=STRIDE, sigma=1.2):
            \"\"\"Build heatmap, size and mask targets from box annotations.

            The heatmap target is a Gaussian rather than a single 1: a prediction one
            pixel off is nearly right, and the loss should say so.
            \"\"\"
            n = len(annotations)
            heat = torch.zeros(n, N_CLASSES, grid, grid)
            size = torch.zeros(n, 2, grid, grid)
            mask = torch.zeros(n, 1, grid, grid)

            yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
            for i, a in enumerate(annotations):
                for box, label in zip(a["boxes"], a["labels"]):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
                    gx, gy = int(min(cx, grid - 1)), int(min(cy, grid - 1))

                    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                    heat[i, label] = torch.maximum(heat[i, label], g)

                    # The centre cell must be EXACTLY 1.0. The Gaussian is evaluated at
                    # integer grid coordinates but the true centre is continuous, so its
                    # peak lands at 0.9999-something. The focal loss below selects
                    # positives with `target == 1`, and with no exact 1.0 anywhere it
                    # finds none, trains on background only, and predicts nothing --
                    # while reporting a perfectly healthy-looking loss curve.
                    heat[i, label, gy, gx] = 1.0

                    size[i, 0, gy, gx] = (x2 - x1) / stride
                    size[i, 1, gy, gx] = (y2 - y1) / stride
                    mask[i, 0, gy, gx] = 1.0
            return heat, size, mask

        heat_demo, size_demo, mask_demo = make_targets(ann_train[:4])
        print("heatmap:", tuple(heat_demo.shape), " size:", tuple(size_demo.shape))
        print("objects in these 4 scenes:", int(mask_demo.sum()))
        print("exact-1.0 peaks in the target:", int(heat_demo.eq(1).sum()),
              " <- must equal the object count")
        """),
        md("""
        > **A bug worth remembering.** The line setting the centre cell to exactly `1.0`
        > looks like a rounding detail. Remove it and this detector trains to completion,
        > reports a falling loss, and predicts **nothing at all** — because the focal loss
        > below picks its positive examples with `target == 1`, and a Gaussian evaluated at
        > integer coordinates never quite reaches 1.
        >
        > No exception, no warning, a plausible loss curve, and a completely dead model.
        > This is exactly the class of bug the Lecture 8 debugging protocol is for.
        """),
        code("""
        fig, axes = plt.subplots(2, 4, figsize=(13, 6))
        for j in range(4):
            axes[0, j].imshow(X_train_np[j]); axes[0, j].axis("off")
            axes[0, j].set_title(f"scene {j}", fontsize=10)
            axes[1, j].imshow(heat_demo[j].max(dim=0).values, cmap="hot")
            axes[1, j].axis("off"); axes[1, j].set_title("centre heatmap target", fontsize=10)
        plt.tight_layout(); plt.show()
        """),
        code("""
        class CentreNet(nn.Module):
            \"\"\"Backbone at stride 4, with a classification head and a size head.\"\"\"

            def __init__(self, n_classes=N_CLASSES, width=24):
                super().__init__()
                def block(cin, cout, stride=1):
                    return nn.Sequential(
                        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                        nn.BatchNorm2d(cout), nn.ReLU())

                # Downsample to the output stride first, then do the expensive work
                # at 24x24. Same output resolution as convolving at 96x96 throughout,
                # but ~3x faster -- worth knowing as a general design habit.
                self.backbone = nn.Sequential(
                    block(3, width, stride=2),               # 96 -> 48
                    block(width, width * 2, stride=2),       # 48 -> 24
                    block(width * 2, width * 2),
                    block(width * 2, width * 2),
                )
                self.heat_head = nn.Conv2d(width * 2, n_classes, 1)
                self.size_head = nn.Conv2d(width * 2, 2, 1)

                # Bias the heatmap toward "no object". Without this the loss starts
                # enormous, because almost every one of the 576 locations is background.
                nn.init.constant_(self.heat_head.bias, -4.0)

            def forward(self, x):
                f = self.backbone(x)
                return torch.sigmoid(self.heat_head(f)).clamp(1e-4, 1 - 1e-4), self.size_head(f)

        m = CentreNet()
        h, s = m(torch.randn(2, 3, 96, 96))
        print("heatmap out:", tuple(h.shape), " size out:", tuple(s.shape))
        print(f"parameters: {count_parameters(m):,}")
        """),
        md("""
        ### Focal loss

        Of the `24 x 24 x 4 = 2304` heatmap locations, typically two or three contain an
        object. Plain cross-entropy is swamped by the easy background and the model learns
        to predict "nothing" everywhere.

        Focal loss down-weights examples that are already well classified:

        $$\\text{FL}(p_t) = -(1-p_t)^\\gamma \\log(p_t)$$

        With `γ = 2`, a background location predicted at 0.99 contributes
        `(0.01)² = 0.0001` of its original weight. The loss is dominated by the cases that
        are still wrong — which is the point.
        """),
        code("""
        def focal_loss(pred, target, alpha=2.0, beta=4.0):
            \"\"\"CornerNet/CenterNet variant of focal loss for Gaussian heatmap targets.\"\"\"
            pos = target.eq(1).float()
            neg = 1 - pos
            neg_weight = torch.pow(1 - target, beta)      # locations near a centre count more

            pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos
            neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg

            n_pos = pos.sum().clamp(min=1)
            return (pos_loss.sum() + neg_loss.sum()) / n_pos


        def size_loss(pred, target, mask):
            \"\"\"L1 on width/height, only where an object centre actually is.\"\"\"
            return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1) / 2
        """),
        code("""
        # Precompute the targets once. ~30 s.
        from dlcourse.data import to_nchw

        X_tr = to_nchw(X_train_np)
        X_va = to_nchw(X_val_np)
        heat_tr, size_tr, mask_tr = make_targets(ann_train)
        heat_va, size_va, mask_va = make_targets(ann_val)

        n_objects = sum(len(a['boxes']) for a in ann_train)
        n_peaks = int(heat_tr.eq(1).sum())
        assert n_peaks == n_objects, (
            f"{n_peaks} exact peaks but {n_objects} objects -- the focal loss would "
            "see the wrong number of positives")
        print("targets built:", tuple(heat_tr.shape))
        print(f"{n_peaks} exact-1.0 peaks for {n_objects} objects -- matches")
        """),
        code("""
        # ~3 minutes. This is the longest cell in the lecture.
        from torch.utils.data import TensorDataset, DataLoader

        set_seed(0)
        detector = CentreNet()
        opt = torch.optim.Adam(detector.parameters(), lr=2e-3)
        loader = DataLoader(TensorDataset(X_tr, heat_tr, size_tr, mask_tr),
                            batch_size=32, shuffle=True)

        EPOCHS = 8
        history = []
        for epoch in range(EPOCHS):
            detector.train()
            running = 0.0
            for xb, hb, sb, mb in loader:
                opt.zero_grad()
                ph, ps = detector(xb)
                loss = focal_loss(ph, hb) + 0.1 * size_loss(ps, sb, mb)
                loss.backward(); opt.step()
                running += loss.item() * len(xb)

            detector.eval()
            with torch.no_grad():
                vh, vs = detector(X_va[:200])
                vloss = (focal_loss(vh, heat_va[:200])
                         + 0.1 * size_loss(vs, size_va[:200], mask_va[:200])).item()
            history.append((running / len(X_tr), vloss))
            print(f"epoch {epoch+1:2d}/{EPOCHS}  train {history[-1][0]:.4f}  val {vloss:.4f}")
        """),
        code("""
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot([h[0] for h in history], label="train", marker="o", ms=3)
        ax.plot([h[1] for h in history], label="val", marker="s", ms=3)
        ax.set_xlabel("epoch"); ax.set_ylabel("focal + size loss")
        ax.set_title("Detector training"); ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        section("5. Decoding predictions", """
        The network outputs dense maps. Turning them into a list of boxes takes three
        steps: find local peaks, read the size at each peak, then apply NMS.

        The 3x3 max-pool trick is how CenterNet does peak-finding without an explicit NMS
        on the heatmap: a location survives only if it equals the maximum of its
        neighbourhood.
        """),
        code("""
        @torch.no_grad()
        def decode(heat, size, threshold=0.3, stride=STRIDE, topk=20):
            \"\"\"Dense maps -> (boxes, scores, labels) for one image.\"\"\"
            # Peak detection: keep only local maxima.
            pooled = F.max_pool2d(heat, 3, stride=1, padding=1)
            peaks = heat * (pooled == heat).float()

            c, gy, gx = torch.where(peaks[0] > threshold)
            if len(c) == 0:
                return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)

            scores = peaks[0, c, gy, gx]
            order = scores.argsort(descending=True)[:topk]
            c, gy, gx, scores = c[order], gy[order], gx[order], scores[order]

            w = size[0, 0, gy, gx] * stride
            h = size[0, 1, gy, gx] * stride
            cx = (gx.float() + 0.5) * stride
            cy = (gy.float() + 0.5) * stride

            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
            keep = batched_nms(boxes, scores, c, iou_threshold=0.4)
            return boxes[keep], scores[keep], c[keep]
        """),
        code("""
        detector.eval()
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for j in range(4):
            with torch.no_grad():
                ph, ps = detector(X_va[j:j+1])
            boxes, scores, labels = decode(ph, ps, threshold=0.3)

            show_boxes(X_val_np[j], ann_val[j]["boxes"], ann_val[j]["labels"],
                       DET_CLASSES, ax=axes[0, j], title="ground truth")
            show_boxes(X_val_np[j], boxes, labels, DET_CLASSES, scores=scores,
                       ax=axes[1, j], title=f"predicted ({len(boxes)} boxes)")
        plt.tight_layout(); plt.show()
        """),
        section("6. Mean Average Precision", """
        mAP is the standard detection metric and it is easy to get subtly wrong.

        For each class: sort every prediction across the whole dataset by score, walk down
        the list marking each as a true or false positive (a ground-truth box can only be
        matched once), accumulate precision and recall, then integrate the
        precision-recall curve.
        """),
        code("""
        def average_precision(preds, gts, class_id, iou_threshold=0.5):
            \"\"\"preds: list of (image_idx, box, score). gts: {image_idx: [boxes]}.\"\"\"
            preds = sorted([p for p in preds if p[3] == class_id],
                           key=lambda p: -p[2])
            n_gt = sum(len(v) for v in gts.values())
            if n_gt == 0:
                return float("nan"), [], []

            matched = {k: torch.zeros(len(v), dtype=torch.bool) for k, v in gts.items()}
            tp = torch.zeros(len(preds))
            fp = torch.zeros(len(preds))

            for i, (img_idx, box, score, _) in enumerate(preds):
                boxes_gt = gts.get(img_idx, [])
                if len(boxes_gt) == 0:
                    fp[i] = 1
                    continue
                ious = box_iou(box.unsqueeze(0), torch.stack(boxes_gt))[0]
                best = ious.argmax()
                if ious[best] >= iou_threshold and not matched[img_idx][best]:
                    tp[i] = 1
                    matched[img_idx][best] = True      # each ground truth matched once
                else:
                    fp[i] = 1

            tp_c, fp_c = tp.cumsum(0), fp.cumsum(0)
            recall = tp_c / n_gt
            precision = tp_c / (tp_c + fp_c).clamp(min=1e-9)

            # All-point interpolation: precision is made monotonically decreasing.
            mrec = torch.cat([torch.tensor([0.]), recall, torch.tensor([1.])])
            mpre = torch.cat([torch.tensor([0.]), precision, torch.tensor([0.])])
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=True)[0]
            ap = ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum().item()
            return ap, recall.tolist(), precision.tolist()
        """),
        code("""
        # Run the detector over the whole validation set.
        N_EVAL = 300
        all_preds = []
        gts_by_class = {c: {} for c in range(N_CLASSES)}

        detector.eval()
        with torch.no_grad():
            for i in range(N_EVAL):
                ph, ps = detector(X_va[i:i+1])
                boxes, scores, labels = decode(ph, ps, threshold=0.05, topk=20)
                for b, s, l in zip(boxes, scores, labels):
                    all_preds.append((i, b, s.item(), int(l)))
                for b, l in zip(ann_val[i]["boxes"], ann_val[i]["labels"]):
                    gts_by_class[l].setdefault(i, []).append(torch.tensor(b))

        print(f"{len(all_preds)} predictions over {N_EVAL} images")

        aps, curves = {}, {}
        for c, name in enumerate(DET_CLASSES):
            ap, rec, prec = average_precision(all_preds, gts_by_class[c], c, 0.5)
            aps[name] = ap
            curves[name] = (rec, prec)
            print(f"  AP@0.5  {name:<10} {ap:.4f}")

        mAP = float(np.nanmean(list(aps.values())))
        print(f"\\nmAP@0.5: {mAP:.4f}")
        """),
        code("""
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, (rec, prec) in curves.items():
            if rec:
                ax.plot(rec, prec, lw=2, label=f"{name} (AP={aps[name]:.3f})")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title(f"Precision-recall by class — mAP@0.5 = {mAP:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()

        print("Read the curve, not just the summary number. A class whose precision")
        print("collapses at low recall is failing differently from one that never")
        print("reaches high recall at all.")
        """),
        section("7. Choosing an operating point", """
        The confidence threshold is a deployment decision, not a modelling one. Sweep it
        and pick deliberately.
        """),
        code("""
        thresholds = np.arange(0.05, 0.95, 0.05)
        prec_at, rec_at = [], []

        with torch.no_grad():
            cache = [detector(X_va[i:i+1]) for i in range(150)]

        for t in thresholds:
            tp = fp = fn = 0
            for i, (ph, ps) in enumerate(cache):
                boxes, scores, labels = decode(ph, ps, threshold=float(t))
                gt_boxes = torch.tensor(ann_val[i]["boxes"], dtype=torch.float32)
                used = torch.zeros(len(gt_boxes), dtype=torch.bool)
                for b in boxes:
                    if len(gt_boxes) == 0:
                        fp += 1; continue
                    ious = box_iou(b.unsqueeze(0), gt_boxes)[0]
                    j = ious.argmax()
                    if ious[j] >= 0.5 and not used[j]:
                        tp += 1; used[j] = True
                    else:
                        fp += 1
                fn += int((~used).sum())
            prec_at.append(tp / max(tp + fp, 1))
            rec_at.append(tp / max(tp + fn, 1))

        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(thresholds, prec_at, marker="o", ms=3, label="precision")
        ax.plot(thresholds, rec_at,  marker="s", ms=3, label="recall")
        f1 = [2*p*r/max(p+r, 1e-9) for p, r in zip(prec_at, rec_at)]
        ax.plot(thresholds, f1, marker="^", ms=3, label="F1", ls="--")
        best = int(np.argmax(f1))
        ax.axvline(thresholds[best], color="k", ls=":", label=f"best F1 @ {thresholds[best]:.2f}")
        ax.set_xlabel("confidence threshold"); ax.set_ylabel("score")
        ax.set_title("Operating point selection"); ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()

        print(f"best F1 = {f1[best]:.3f} at threshold {thresholds[best]:.2f}")
        print("\\nBut F1 is only the right objective if false positives and false negatives")
        print("cost the same. For a medical screen they do not — favour recall. For an")
        print("automated action taken without review, favour precision.")
        """),
        todo("1", "IoU, vectorised", """
        Implement `box_iou(boxes_a, boxes_b)` returning the full pairwise matrix. Include
        tests for identical boxes (1.0), disjoint boxes (0.0), and one box contained in
        another.
        """),
        todo_cell(),
        todo("2", "Non-maximum suppression", """
        Implement NMS from scratch and verify it against `torchvision.ops.nms` if you have
        torchvision, otherwise against the hand-worked five-box example above.
        """),
        todo_cell(),
        todo("3", "Build the detector", """
        Implement a CNN backbone with a centre heatmap head over a 24x24 grid and a
        2-channel width/height regression head. Train with focal loss on the heatmap and
        L1 on the sizes.
        """),
        todo_cell(),
        todo("4", "Decode predictions", """
        Write `decode(heatmap, sizes, threshold)` extracting peaks, converting grid
        coordinates to pixel boxes, and applying NMS. Visualise predictions against ground
        truth for eight validation scenes.
        """),
        todo_cell(),
        todo("5", "mean Average Precision", """
        Implement AP at IoU 0.5 using the all-point interpolation rule, then mAP over the
        four classes. Plot the precision-recall curve for each class.
        """),
        todo_cell(),
        todo("6", "Confidence threshold sweep", """
        Sweep the score threshold from 0.05 to 0.95. Plot precision and recall against
        threshold, pick an operating point, and justify it for a named use case.
        """),
        todo_cell(),
        todo("7", "Focal loss ablation *(stretch)*", """
        Train with plain binary cross-entropy instead of focal loss. Report the mAP
        difference and explain it via the foreground/background ratio in your data.
        """),
        todo_cell(),
        todo("8", "Multi-scale predictions *(stretch)*", """
        Add a second prediction head at a coarser stride. Report mAP separately for small
        and large objects.
        """),
        todo_cell(),
    ]
