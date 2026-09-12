"""Lecture 13 notebook — attention and the Transformer, from scratch."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. Scaled dot-product attention", """
        $$\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V$$

        Read it as a soft dictionary lookup. Each **query** asks a question; each **key**
        advertises what its position offers; the dot product scores the match; softmax
        turns scores into weights; the output is a weighted average of the **values**.
        """),
        code("""
        def attention(Q, K, V, mask=None):
            \"\"\"Q, K, V: (..., seq_len, d). mask: True where attention is FORBIDDEN.\"\"\"
            d_k = Q.size(-1)
            scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            return weights @ V, weights


        torch.manual_seed(0)
        Q, K, V = torch.randn(2, 5, 16), torch.randn(2, 5, 16), torch.randn(2, 5, 16)
        ours, w = attention(Q, K, V)
        theirs = F.scaled_dot_product_attention(Q, K, V)

        print("output shape          :", tuple(ours.shape))
        print("attention weights     :", tuple(w.shape), " rows sum to", w.sum(-1)[0, 0].item())
        print("max |Δ| vs PyTorch    :", (ours - theirs).abs().max().item())
        """),
        section("2. Why divide by sqrt(d_k)", """
        Without the scaling, dot products grow with dimension. Softmax of large values
        saturates to one-hot, and a saturated softmax has almost no gradient — the layer
        stops learning.

        If `q` and `k` have independent unit-variance entries, `q·k` has variance `d_k`.
        Dividing by `sqrt(d_k)` restores unit variance regardless of dimension.
        """),
        code("""
        fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
        for col, d_k in enumerate([8, 64, 512]):
            torch.manual_seed(0)
            q = torch.randn(1000, d_k)
            k = torch.randn(1000, d_k)
            raw = (q * k).sum(-1)
            scaled = raw / (d_k ** 0.5)

            axes[0, col].hist(raw.numpy(), bins=50, color="#c0392b", alpha=.8)
            axes[0, col].set_title(f"d_k = {d_k}, unscaled\\nstd = {raw.std():.2f}", fontsize=10)
            axes[1, col].hist(scaled.numpy(), bins=50, color="#1b866b", alpha=.8)
            axes[1, col].set_title(f"d_k = {d_k}, scaled\\nstd = {scaled.std():.2f}", fontsize=10)
            for r in (0, 1):
                axes[r, col].set_xlabel("dot product value")
        fig.suptitle("Dot-product magnitude grows with dimension; the scaling removes it")
        plt.tight_layout(); plt.show()
        """),
        code("""
        # The consequence: a saturated softmax, and a vanished gradient.
        print(f"{'d_k':>6}{'max attn (unscaled)':>22}{'entropy':>10}"
              f"{'max attn (scaled)':>20}{'entropy':>10}")
        print("-" * 70)
        for d_k in (8, 64, 512, 4096):
            torch.manual_seed(0)
            q = torch.randn(1, 1, d_k)
            k = torch.randn(1, 30, d_k)
            raw = (q @ k.transpose(-2, -1))[0, 0]
            for_scaled = raw / (d_k ** 0.5)

            p_raw = F.softmax(raw, -1)
            p_sc = F.softmax(for_scaled, -1)
            ent = lambda p: -(p * (p + 1e-12).log()).sum().item()
            print(f"{d_k:>6}{p_raw.max().item():>22.4f}{ent(p_raw):>10.3f}"
                  f"{p_sc.max().item():>20.4f}{ent(p_sc):>10.3f}")
        print("\\nAt d_k=4096 the unscaled softmax puts nearly all mass on one position.")
        print("Maximum entropy for 30 positions is log(30) = %.3f." % np.log(30))
        """),
        section("3. Multi-head attention", """
        Rather than one attention over `d_model` dimensions, project into `h` subspaces,
        attend in each independently, concatenate and project out.

        The cost is unchanged — each head works in `d_model / h` dimensions — but heads
        specialise, without anyone telling them to.
        """),
        code("""
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads, dropout=0.0):
                super().__init__()
                assert d_model % n_heads == 0, "d_model must divide evenly among heads"
                self.d_model, self.n_heads = d_model, n_heads
                self.d_head = d_model // n_heads

                self.w_q = nn.Linear(d_model, d_model)
                self.w_k = nn.Linear(d_model, d_model)
                self.w_v = nn.Linear(d_model, d_model)
                self.w_o = nn.Linear(d_model, d_model)
                self.dropout = nn.Dropout(dropout)

            def _split(self, x):
                B, L, _ = x.shape
                return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

            def forward(self, x, mask=None, return_weights=False):
                B, L, _ = x.shape
                q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))

                scores = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)
                if mask is not None:
                    scores = scores.masked_fill(mask, float("-inf"))
                weights = self.dropout(F.softmax(scores, dim=-1))

                out = (weights @ v).transpose(1, 2).reshape(B, L, self.d_model)
                out = self.w_o(out)
                return (out, weights) if return_weights else out

        mha = MultiHeadAttention(64, 8)
        x = torch.randn(2, 10, 64)
        out, w = mha(x, return_weights=True)
        print("input :", tuple(x.shape))
        print("output:", tuple(out.shape))
        print("weights (B, heads, L, L):", tuple(w.shape))
        """),
        code("""
        # Verify against nn.MultiheadAttention. The tricky part is its packed
        # in_proj_weight: Q, K and V projections are stored in one stacked matrix.
        torch.manual_seed(0)
        d_model, n_heads, L = 32, 4, 6
        mine = MultiHeadAttention(d_model, n_heads)
        ref = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=True)

        with torch.no_grad():
            ref.in_proj_weight.copy_(torch.cat([mine.w_q.weight, mine.w_k.weight,
                                                mine.w_v.weight], dim=0))
            ref.in_proj_bias.copy_(torch.cat([mine.w_q.bias, mine.w_k.bias, mine.w_v.bias]))
            ref.out_proj.weight.copy_(mine.w_o.weight)
            ref.out_proj.bias.copy_(mine.w_o.bias)

        x = torch.randn(2, L, d_model)
        mine.eval(); ref.eval()
        with torch.no_grad():
            a = mine(x)
            b, _ = ref(x, x, x, need_weights=False)
        print("max |Δ| vs nn.MultiheadAttention:", (a - b).abs().max().item())
        """),
        section("4. Position", """
        Attention is permutation-equivariant: shuffle the input and the output shuffles
        identically. It has no concept of order whatsoever.

        Position has to be injected. Sinusoidal encodings are fixed and extrapolate to
        lengths never seen in training.
        """),
        code("""
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=512):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                pos = torch.arange(max_len).unsqueeze(1).float()
                div = torch.exp(torch.arange(0, d_model, 2).float()
                                * (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        pe = PositionalEncoding(64, 100)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
        im = a1.imshow(pe.pe[0].numpy().T, cmap="RdBu_r", aspect="auto")
        a1.set_xlabel("position"); a1.set_ylabel("embedding dimension")
        a1.set_title("Sinusoidal positional encoding"); plt.colorbar(im, ax=a1)

        for d in (0, 4, 16, 32):
            a2.plot(pe.pe[0, :60, d].numpy(), label=f"dim {d}")
        a2.set_xlabel("position"); a2.set_ylabel("value")
        a2.set_title("Different dimensions oscillate at different frequencies")
        a2.legend(fontsize=9); a2.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        code("""
        # Prove attention is order-blind without it.
        torch.manual_seed(0)
        mha_test = MultiHeadAttention(16, 2).eval()
        seq = torch.randn(1, 5, 16)
        perm = torch.tensor([3, 1, 4, 0, 2])

        with torch.no_grad():
            out_original = mha_test(seq)
            out_permuted = mha_test(seq[:, perm])

        print("attention(shuffled input) == shuffle(attention(input))? ",
              torch.allclose(out_permuted, out_original[:, perm], atol=1e-6))
        print("\\nThat is permutation equivariance. Without positional information,")
        print("'dog bites man' and 'man bites dog' are the same input.")
        """),
        section("5. The Transformer encoder block", """
        Two sub-layers — multi-head self-attention and a position-wise feed-forward
        network — each wrapped in a residual connection and a LayerNorm.

        This uses **pre-norm** (normalise before the sub-layer). The original paper used
        post-norm, which needs careful learning-rate warmup to train at all. Pre-norm is
        now standard because it just works.
        """),
        code("""
        class EncoderBlock(nn.Module):
            def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
                super().__init__()
                d_ff = d_ff or 4 * d_model            # 4x is the convention
                self.norm1 = nn.LayerNorm(d_model)
                self.attn = MultiHeadAttention(d_model, n_heads, dropout)
                self.norm2 = nn.LayerNorm(d_model)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, d_ff), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_ff, d_model),
                )
                self.dropout = nn.Dropout(dropout)

            def forward(self, x, mask=None, return_weights=False):
                normed = self.norm1(x)
                if return_weights:
                    a, w = self.attn(normed, mask, return_weights=True)
                else:
                    a, w = self.attn(normed, mask), None
                x = x + self.dropout(a)                          # residual 1
                x = x + self.dropout(self.ff(self.norm2(x)))     # residual 2
                return (x, w) if return_weights else x

        block = EncoderBlock(64, 8)
        print("block output:", tuple(block(torch.randn(2, 10, 64)).shape))
        print(f"parameters in one block: {count_parameters(block):,}")

        attn_params = count_parameters(block.attn)
        ff_params = count_parameters(block.ff)
        print(f"  attention   : {attn_params:,} ({attn_params/(attn_params+ff_params):.0%})")
        print(f"  feed-forward: {ff_params:,} ({ff_params/(attn_params+ff_params):.0%})")
        print("\\nMost parameters live in the feed-forward layers, not in attention.")
        """),
        section("6. A Transformer text classifier", """
        Train it on the sentiment corpus.
        """),
        code("""
        texts, labels = load_sentiment()
        print(f"{len(texts)} sentences")
        for t, l in list(zip(texts, labels))[:3]:
            print(f"  [{'pos' if l else 'neg'}] {t}")

        # Word-level vocabulary.
        from collections import Counter
        counter = Counter(w.lower().strip(".,!?") for t in texts for w in t.split())
        vocab = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common()]
        stoi = {w: i for i, w in enumerate(vocab)}
        MAX_LEN = 32
        print(f"\\nvocabulary: {len(vocab)} words")
        """),
        code("""
        def encode(text, max_len=MAX_LEN):
            ids = [stoi.get(w.lower().strip(".,!?"), 1) for w in text.split()][:max_len]
            return ids + [0] * (max_len - len(ids))

        X = torch.tensor([encode(t) for t in texts])
        y = torch.tensor(labels)
        n_train = int(0.8 * len(X))
        Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
        print("train:", tuple(Xtr.shape), " test:", tuple(Xte.shape))
        """),
        code("""
        class TransformerClassifier(nn.Module):
            def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                         n_classes=2, max_len=MAX_LEN, use_pos=True):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.pos = PositionalEncoding(d_model, max_len) if use_pos else nn.Identity()
                self.blocks = nn.ModuleList(
                    [EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
                self.norm = nn.LayerNorm(d_model)
                self.fc = nn.Linear(d_model, n_classes)

            def forward(self, ids, return_weights=False):
                pad_mask = (ids == 0)                                   # B, L
                # Broadcast to (B, 1, 1, L): forbid attending TO padding.
                mask = pad_mask[:, None, None, :]

                x = self.pos(self.embed(ids))
                weights = []
                for block in self.blocks:
                    if return_weights:
                        x, w = block(x, mask, return_weights=True)
                        weights.append(w)
                    else:
                        x = block(x, mask)
                x = self.norm(x)

                # Mean-pool over real tokens only.
                valid = (~pad_mask).unsqueeze(-1).float()
                pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
                logits = self.fc(pooled)
                return (logits, weights) if return_weights else logits

        set_seed(0)
        clf = TransformerClassifier(len(vocab))
        print(f"parameters: {count_parameters(clf):,}")
        """),
        code("""
        # ~90 s.
        set_seed(0)
        clf = TransformerClassifier(len(vocab))
        opt = torch.optim.AdamW(clf.parameters(), lr=2e-3, weight_decay=0.01)
        hist = train(clf, (Xtr, ytr), (Xte, yte), epochs=15, optimizer=opt, batch_size=64)

        _, test_acc = evaluate(clf, (Xte, yte))
        print(f"\\ntest accuracy: {test_acc:.4f}")
        """),
        code("""
        plot_history(hist, "Transformer sentiment classifier")
        plt.show()
        """),
        section("7. Looking at the attention", """
        Attention weights are not an explanation, but they are informative. Plot them.
        """),
        code("""
        clf.eval()
        example = "The film was absolutely brilliant. The story felt moving. Highly recommended."
        ids = torch.tensor([encode(example)])
        with torch.no_grad():
            logits, weights = clf(ids, return_weights=True)

        tokens = [w.lower().strip(".,!?") for w in example.split()][:MAX_LEN]
        n_tok = len(tokens)
        pred = "positive" if logits.argmax() == 1 else "negative"
        print(f'"{example}"\\n-> predicted {pred} '
              f'(confidence {F.softmax(logits, -1).max():.3f})')

        last_layer = weights[-1][0]                     # heads, L, L
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        for h, ax in enumerate(axes):
            im = ax.imshow(last_layer[h, :n_tok, :n_tok].numpy(), cmap="viridis")
            ax.set_xticks(range(n_tok), tokens, rotation=90, fontsize=7)
            ax.set_yticks(range(n_tok), tokens, fontsize=7)
            ax.set_title(f"layer 2, head {h}", fontsize=10)
        fig.suptitle("Attention weights — row = querying token, column = attended token")
        plt.tight_layout(); plt.show()
        """),
        section("8. Causal masking", """
        A decoder must not see the future. The mask is an upper-triangular block of
        `-inf` applied before the softmax.

        Get this wrong and the model "achieves" an astonishing loss by simply reading the
        answer.
        """),
        code("""
        def causal_mask(seq_len):
            \"\"\"True where attention is forbidden: strictly above the diagonal.\"\"\"
            return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

        m = causal_mask(7)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.imshow(~m, cmap="Greens"); a1.set_title("Allowed attention (causal)")
        a1.set_xlabel("key position"); a1.set_ylabel("query position")

        torch.manual_seed(0)
        Q = K = V = torch.randn(1, 7, 16)
        _, w_causal = attention(Q, K, V, mask=m)
        a2.imshow(w_causal[0].detach().numpy(), cmap="viridis")
        a2.set_title("Resulting attention weights")
        a2.set_xlabel("key position"); a2.set_ylabel("query position")
        plt.tight_layout(); plt.show()

        print("row sums (each is a distribution):", w_causal[0].sum(-1).tolist())
        print("upper-triangular mass (must be 0):",
              float(w_causal[0][m].sum()))
        """),
        code("""
        # The ablation: a next-token model with and without the mask.
        class TinyLM(nn.Module):
            def __init__(self, vocab_size, d_model=48, causal=True):
                super().__init__()
                self.causal = causal
                self.embed = nn.Embedding(vocab_size, d_model)
                self.pos = PositionalEncoding(d_model, 64)
                self.block = EncoderBlock(d_model, 4, dropout=0.0)
                self.fc = nn.Linear(d_model, vocab_size)

            def forward(self, ids):
                L = ids.size(1)
                mask = causal_mask(L).to(ids.device) if self.causal else None
                x = self.block(self.pos(self.embed(ids)), mask)
                return self.fc(x)

        seq_data = Xtr[:, :16]
        for causal in (True, False):
            set_seed(0)
            lm = TinyLM(len(vocab), causal=causal)
            opt = torch.optim.Adam(lm.parameters(), lr=3e-3)
            for _ in range(150):
                idx = torch.randint(0, len(seq_data), (64,))
                batch = seq_data[idx]
                opt.zero_grad()
                logits = lm(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, len(vocab)),
                                       batch[:, 1:].reshape(-1))
                loss.backward(); opt.step()
            label = "causal mask (correct)" if causal else "NO mask (cheating)"
            print(f"{label:<26} final loss {loss.item():.4f}")

        print("\\nThe unmasked model reaches an implausibly low loss because position t")
        print("can attend to position t+1 — the very token it is asked to predict.")
        """),
        section("9. The quadratic cost", """
        Attention is `O(n²)` in sequence length, in both time and memory. Measure the
        exponent rather than taking it on faith.
        """),
        code("""
        import time

        lengths = [32, 64, 128, 256, 512]
        times, mems = [], []
        attn_layer = MultiHeadAttention(64, 4).eval()

        for L in lengths:
            x = torch.randn(8, L, 64)
            with torch.no_grad():
                for _ in range(3):
                    attn_layer(x)
                t0 = time.time()
                for _ in range(20):
                    attn_layer(x)
                times.append((time.time() - t0) / 20 * 1000)
            mems.append(8 * 4 * L * L * 4 / 1024**2)      # B*heads*L*L floats, in MB

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
        a1.loglog(lengths, times, marker="o", lw=2, label="measured")
        ref = [times[0] * (L / lengths[0]) ** 2 for L in lengths]
        a1.loglog(lengths, ref, "--", color="gray", label="exact n² reference")
        a1.set_xlabel("sequence length"); a1.set_ylabel("forward pass (ms)")
        a1.set_title("Time"); a1.legend(); a1.grid(alpha=.3, which="both")

        a2.loglog(lengths, mems, marker="s", lw=2, color="#7a3ea8")
        a2.set_xlabel("sequence length"); a2.set_ylabel("attention matrix (MB)")
        a2.set_title("Memory held by the attention matrix alone")
        a2.grid(alpha=.3, which="both")
        plt.tight_layout(); plt.show()

        exponent = np.polyfit(np.log(lengths), np.log(times), 1)[0]
        print(f"fitted exponent: {exponent:.2f}   (theory says 2 for the attention matrix;")
        print("at short lengths the linear projections still dominate, pulling it below 2)")
        for L, m in zip(lengths, mems):
            print(f"  length {L:>4}: attention matrix {m:8.2f} MB")
        """),
        md("""
        ### What is done about it

        - **FlashAttention** — tiling and recomputation remove the memory cost. It is
          *exact*, not an approximation, and it is why long contexts became practical.
        - **Sliding-window / local attention** — each token attends to a fixed
          neighbourhood.
        - **Sparse and linear attention** — approximate the matrix with a cheaper form.

        Note the two plots disagree about urgency: time is manageable, memory is not. At
        length 8192 with a realistic batch, the attention matrix alone runs to gigabytes.
        """),
        todo("1", "Scaled dot-product attention", """
        Implement `attention(Q, K, V, mask=None)` with correct masking. Verify against
        `F.scaled_dot_product_attention` to within `1e-5`.
        """),
        todo_cell(),
        todo("2", "Why sqrt(d_k)", """
        For `d_k` in {8, 64, 512}, sample random Q and K and plot the distribution of
        attention weights with and without the scaling. Show numerically that softmax
        saturates without it.
        """),
        todo_cell(),
        todo("3", "Multi-head attention", """
        Implement `MultiHeadAttention` as a module and verify against
        `torch.nn.MultiheadAttention` with copied weights — mind its packed
        `in_proj_weight` layout.
        """),
        todo_cell(),
        todo("4", "Positional encoding", """
        Implement sinusoidal encoding and plot the matrix as a heatmap. Then show that
        removing it leaves a sentence-order task at chance accuracy.
        """),
        todo_cell(),
        todo("5", "Encoder block and classifier", """
        Assemble a pre-norm encoder block and build a two-layer Transformer classifier.
        Train on `sentiment.csv` to at least 95% test accuracy.
        """),
        todo_cell(),
        todo("6", "Attention visualisation", """
        Plot the attention matrix for each head on two example sentences. Describe in
        three sentences any pattern a head appears to have specialised in.
        """),
        todo_cell(),
        todo("7", "Causal masking", """
        Implement a causal mask and prove by ablation that removing it lets a language
        model cheat — report the implausibly low loss it achieves.
        """),
        todo_cell(),
        todo("8", "Quadratic scaling *(stretch)*", """
        Measure forward-pass time and peak memory for lengths 64, 128, 256, 512 and 1024.
        Fit the exponent of the scaling curve and compare it to the theoretical 2.
        """),
        todo_cell(),
        todo("9", "Pre-norm versus post-norm *(stretch)*", """
        Train a 6-layer Transformer in both configurations without warmup. Report which
        diverges and explain why in terms of residual-stream magnitude.
        """),
        todo_cell(),
    ]
