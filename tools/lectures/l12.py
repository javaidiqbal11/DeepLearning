"""Lecture 12 notebook — RNNs, LSTMs, and image captioning."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("1. Recurrence from scratch", """
        An RNN applies the *same* weights at every timestep:

        $$h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

        That is weight sharing across time, exactly analogous to a CNN's weight sharing
        across space.
        """),
        code("""
        class MyRNNCell(nn.Module):
            \"\"\"One step of a vanilla RNN, written out explicitly.\"\"\"

            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
                self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
                self.b = nn.Parameter(torch.zeros(hidden_size))

            def forward(self, x, h):
                return torch.tanh(x @ self.W_xh + h @ self.W_hh + self.b)


        # Verify against PyTorch with copied weights.
        torch.manual_seed(0)
        mine = MyRNNCell(4, 6)
        theirs = nn.RNNCell(4, 6, nonlinearity="tanh")
        with torch.no_grad():
            theirs.weight_ih.copy_(mine.W_xh.T)
            theirs.weight_hh.copy_(mine.W_hh.T)
            theirs.bias_ih.copy_(mine.b)
            theirs.bias_hh.zero_()

        x, h = torch.randn(3, 4), torch.randn(3, 6)
        print("max |difference| vs nn.RNNCell:",
              (mine(x, h) - theirs(x, h)).abs().max().item())
        """),
        section("2. The LSTM, gate by gate", """
        The LSTM adds a **cell state** that runs through the sequence with only additive
        interactions — an unobstructed gradient path. Three gates control it: what to
        forget, what to write, and what to expose.
        """),
        code("""
        class MyLSTMCell(nn.Module):
            \"\"\"All four gates, computed in one matrix multiply then split.\"\"\"

            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.W_x = nn.Parameter(torch.randn(input_size, 4 * hidden_size) * 0.1)
                self.W_h = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) * 0.1)
                self.b = nn.Parameter(torch.zeros(4 * hidden_size))
                # Forget-gate bias to 1: remember by default, learn to forget.
                with torch.no_grad():
                    self.b[hidden_size:2 * hidden_size] = 1.0

            def forward(self, x, state):
                h, c = state
                gates = x @ self.W_x + h @ self.W_h + self.b
                i, f, g, o = gates.chunk(4, dim=1)
                i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
                g = torch.tanh(g)
                c_new = f * c + i * g            # <- the additive path
                h_new = o * torch.tanh(c_new)
                return h_new, c_new, {"input": i, "forget": f, "output": o, "candidate": g}


        torch.manual_seed(0)
        cell = MyLSTMCell(4, 6)
        h0, c0 = torch.zeros(2, 6), torch.zeros(2, 6)
        h1, c1, gates = cell(torch.randn(2, 4), (h0, c0))

        print("gate activations for one input (mean over units):")
        for name, g in gates.items():
            print(f"  {name:<10} {g.mean().item():.3f}")
        print("\\nForget gate starts near 1 because of the bias initialisation:")
        print("  the cell keeps its state unless something teaches it not to.")
        """),
        code("""
        # Verify against nn.LSTMCell. PyTorch orders the gates i, f, g, o — same as ours.
        theirs = nn.LSTMCell(4, 6)
        with torch.no_grad():
            theirs.weight_ih.copy_(cell.W_x.T)
            theirs.weight_hh.copy_(cell.W_h.T)
            theirs.bias_ih.copy_(cell.b)
            theirs.bias_hh.zero_()

        x = torch.randn(2, 4)
        mine_h, mine_c, _ = cell(x, (h0, c0))
        their_h, their_c = theirs(x, (h0, c0))
        print("max |Δh|:", (mine_h - their_h).abs().max().item())
        print("max |Δc|:", (mine_c - their_c).abs().max().item())
        """),
        section("3. The long-range copy task", """
        This is the cleanest demonstration of why gating was invented.

        The task: show the model a token, then `T` steps of noise, then ask it to
        reproduce the original token. Nothing about it is hard except remembering across
        the gap.
        """),
        code("""
        def make_copy_task(n_samples, T, n_symbols=6, seed=0):
            \"\"\"Sequence: [symbol, noise x T, cue]. Target: the original symbol.\"\"\"
            g = torch.Generator().manual_seed(seed)
            symbols = torch.randint(0, n_symbols, (n_samples,), generator=g)
            seq = torch.zeros(n_samples, T + 2, n_symbols + 2)
            seq[torch.arange(n_samples), 0, symbols] = 1.0           # the symbol
            seq[:, 1:T + 1, n_symbols] = 1.0                          # filler
            seq[:, T + 1, n_symbols + 1] = 1.0                        # the cue
            return seq, symbols


        class SeqClassifier(nn.Module):
            def __init__(self, kind, input_size=8, hidden=32, n_out=6):
                super().__init__()
                rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
                self.rnn = rnn(input_size, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, n_out)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1])          # classify from the final state
        """),
        code("""
        # ~3 minutes. Three gap lengths x three architectures = nine small models.
        LENGTHS = [5, 15, 40]
        results = {"RNN": [], "LSTM": [], "GRU": []}

        for T in LENGTHS:
            Xtr, ytr = make_copy_task(1500, T, seed=1)
            Xte, yte = make_copy_task(400, T, seed=2)
            for kind in results:
                set_seed(0)
                model = SeqClassifier(kind)
                opt = torch.optim.Adam(model.parameters(), lr=5e-3)
                for _ in range(35):
                    perm = torch.randperm(len(Xtr))
                    for i in range(0, len(Xtr), 128):
                        idx = perm[i:i + 128]
                        opt.zero_grad()
                        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                with torch.no_grad():
                    acc = (model(Xte).argmax(1) == yte).float().mean().item()
                results[kind].append(acc)
            print(f"T={T:>3}  " + "  ".join(f"{k} {results[k][-1]:.3f}" for k in results))
        """),
        code("""
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for kind, colour in [("RNN", "#c0392b"), ("LSTM", "#1b866b"), ("GRU", "#1f6fb2")]:
            ax.plot(LENGTHS, results[kind], marker="o", lw=2, label=kind, color=colour)
        ax.axhline(1/6, color="k", ls=":", lw=1, label="chance")
        ax.set_xlabel("gap length T (timesteps between the symbol and the cue)")
        ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title("Long-range copy task — where the vanilla RNN gives up")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        code("""
        # The mechanism, measured: gradient reaching timestep 0 across a long sequence.
        def gradient_at_start(kind, T=60, hidden=32):
            set_seed(0)
            model = SeqClassifier(kind, hidden=hidden)
            X, y = make_copy_task(32, T, seed=5)
            X.requires_grad_(True)
            F.cross_entropy(model(X), y).backward()
            per_step = X.grad.abs().sum(dim=(0, 2))
            return per_step

        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for kind, colour in [("RNN", "#c0392b"), ("LSTM", "#1b866b"), ("GRU", "#1f6fb2")]:
            g = gradient_at_start(kind)
            ax.semilogy(g.numpy() + 1e-20, label=kind, color=colour, lw=2)
            print(f"{kind:<5} gradient at t=0: {g[0]:.3e}   at t=-1: {g[-1]:.3e}   "
                  f"ratio {g[0]/max(g[-1], 1e-20):.2e}")
        ax.set_xlabel("timestep"); ax.set_ylabel("|gradient| w.r.t. input (log scale)")
        ax.set_title("Gradient reaching each timestep from a loss at the end")
        ax.legend(); ax.grid(alpha=.3, which="both")
        plt.tight_layout(); plt.show()
        """),
        section("4. A character-level language model", """
        Train an LSTM to predict the next character, then sample from it. Temperature
        controls how much the sampling trusts the model's own distribution.
        """),
        code("""
        text = load_corpus()
        vocab = sorted(set(text))
        stoi = {c: i for i, c in enumerate(vocab)}
        itos = {i: c for c, i in stoi.items()}
        encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

        print(f"corpus: {len(text):,} characters, {len(vocab)} unique")
        print("vocabulary:", "".join(vocab).replace("\\n", "\\\\n"))
        print("\\nsample:\\n", text[:180])
        """),
        code("""
        SEQ_LEN = 64

        def get_batch(data, batch_size=64, seq_len=SEQ_LEN):
            ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
            x = torch.stack([data[i:i + seq_len] for i in ix])
            y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
            return x, y


        class CharLSTM(nn.Module):
            def __init__(self, vocab_size, embed=48, hidden=192, layers=2):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed)
                self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=0.1)
                self.fc = nn.Linear(hidden, vocab_size)

            def forward(self, x, state=None):
                out, state = self.lstm(self.embed(x), state)
                return self.fc(out), state

        set_seed(0)
        lm = CharLSTM(len(vocab))
        print(f"parameters: {count_parameters(lm):,}")
        """),
        code("""
        # ~3 minutes.
        split = int(0.9 * len(encoded))
        train_data, val_data = encoded[:split], encoded[split:]

        opt = torch.optim.Adam(lm.parameters(), lr=3e-3)
        STEPS = 800
        losses = []

        lm.train()
        for step in range(STEPS):
            x, y = get_batch(train_data)
            opt.zero_grad()
            logits, _ = lm(x)
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if (step + 1) % 200 == 0:
                lm.eval()
                with torch.no_grad():
                    vx, vy = get_batch(val_data, 128)
                    vlogits, _ = lm(vx)
                    vloss = F.cross_entropy(vlogits.reshape(-1, len(vocab)), vy.reshape(-1))
                lm.train()
                print(f"step {step+1:5d}  train {np.mean(losses[-100:]):.4f}  "
                      f"val {vloss:.4f}  perplexity {torch.exp(vloss):.2f}")
        """),
        code("""
        @torch.no_grad()
        def sample(model, prompt="The ", length=300, temperature=1.0):
            \"\"\"Temperature < 1 sharpens the distribution, > 1 flattens it.\"\"\"
            model.eval()
            idx = torch.tensor([[stoi.get(c, 0) for c in prompt]])
            state = None
            out = list(prompt)
            logits, state = model(idx, state)
            for _ in range(length):
                probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
                out.append(itos[int(nxt)])
                logits, state = model(nxt.view(1, 1), state)
            return "".join(out)

        for t in (0.5, 1.0, 1.5):
            print(f"\\n{'='*70}\\ntemperature {t}\\n{'='*70}")
            print(sample(lm, length=240, temperature=t))
        """),
        md("""
        ### Reading the temperature effect

        - **0.5** — conservative. Repetitive, but almost every word is real.
        - **1.0** — the model's actual distribution.
        - **1.5** — flattened. More varied, more misspellings and broken structure.

        Temperature is the same knob exposed by every text-generation API you will use.
        """),
        section("5. Image captioning: CNN encoder, LSTM decoder", """
        A one-to-many setup. The CNN compresses the image to a vector; the LSTM generates
        tokens conditioned on it.
        """),
        code("""
        shapes = load_shapes(normalize=True)
        X_img, y_img = shapes["train"]
        caption_rows = load_captions()

        # Build a word-level vocabulary from the captions.
        PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
        words = sorted({w for _, c in caption_rows for w in c.split()})
        cap_vocab = [PAD, BOS, EOS] + words
        cstoi = {w: i for i, w in enumerate(cap_vocab)}
        citos = {i: w for w, i in cstoi.items()}
        MAX_LEN = max(len(c.split()) for _, c in caption_rows) + 2

        print(f"{len(caption_rows)} captions, vocabulary {len(cap_vocab)}, max length {MAX_LEN}")
        print("examples:")
        for idx, c in caption_rows[:4]:
            print(f"  image {idx:>4}: {c}")
        """),
        code("""
        def encode_caption(caption):
            ids = [cstoi[BOS]] + [cstoi[w] for w in caption.split()] + [cstoi[EOS]]
            return ids + [cstoi[PAD]] * (MAX_LEN - len(ids))

        cap_img_idx = torch.tensor([r[0] for r in caption_rows])
        cap_tokens = torch.tensor([encode_caption(r[1]) for r in caption_rows])
        cap_images = X_img[cap_img_idx]

        print("images:", tuple(cap_images.shape), " tokens:", tuple(cap_tokens.shape))
        print("first caption encoded:", cap_tokens[0].tolist())
        print("decoded back:", " ".join(citos[int(t)] for t in cap_tokens[0] if t != cstoi[PAD]))
        """),
        code("""
        class CaptionModel(nn.Module):
            def __init__(self, vocab_size, embed=64, hidden=128):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, hidden),
                )
                self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
                self.lstm = nn.LSTM(embed, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, vocab_size)

            def forward(self, images, tokens):
                \"\"\"Teacher forcing: the ground-truth prefix is fed in during training.\"\"\"
                feat = self.encoder(images)                       # B, hidden
                h0 = feat.unsqueeze(0)                            # initial hidden state
                c0 = torch.zeros_like(h0)
                out, _ = self.lstm(self.embed(tokens), (h0, c0))
                return self.fc(out)

            @torch.no_grad()
            def generate(self, image, max_len=None, greedy=True):
                max_len = max_len or MAX_LEN
                self.eval()
                feat = self.encoder(image.unsqueeze(0))
                state = (feat.unsqueeze(0), torch.zeros_like(feat.unsqueeze(0)))
                token = torch.tensor([[cstoi[BOS]]])
                words_out = []
                for _ in range(max_len):
                    out, state = self.lstm(self.embed(token), state)
                    logits = self.fc(out[:, -1])
                    token = (logits.argmax(-1, keepdim=True) if greedy
                             else torch.multinomial(F.softmax(logits, -1), 1))
                    w = citos[int(token)]
                    if w == EOS:
                        break
                    words_out.append(w)
                return " ".join(words_out)
        """),
        code("""
        # ~3 minutes.
        from torch.utils.data import TensorDataset, DataLoader

        set_seed(0)
        captioner = CaptionModel(len(cap_vocab))
        opt = torch.optim.Adam(captioner.parameters(), lr=2e-3)
        loader = DataLoader(TensorDataset(cap_images, cap_tokens), batch_size=64, shuffle=True)

        for epoch in range(10):
            captioner.train()
            total = 0.0
            for imgs, toks in loader:
                opt.zero_grad()
                # Predict token t+1 from tokens up to t.
                logits = captioner(imgs, toks[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, len(cap_vocab)),
                                       toks[:, 1:].reshape(-1),
                                       ignore_index=cstoi[PAD])   # never learn from padding
                loss.backward(); opt.step()
                total += loss.item() * len(imgs)
            if (epoch + 1) % 2 == 0:
                print(f"epoch {epoch+1:2d}  loss {total/len(cap_images):.4f}")
        """),
        code("""
        raw_imgs = load_shapes()["train"][0]
        test_indices = [1900, 1905, 1912, 1923, 1934, 1947, 1955, 1968]

        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        correct = 0
        for ax, i in zip(axes.ravel(), test_indices):
            caption = captioner.generate(X_img[i])
            true_class = shapes["classes"][y_img[i]]
            hit = true_class in caption
            correct += hit
            ax.imshow(raw_imgs[i].permute(1, 2, 0).numpy()); ax.axis("off")
            ax.set_title(f'"{caption}"\\n(true: {true_class}) {"OK" if hit else "WRONG"}',
                         fontsize=9)
        fig.suptitle("Generated captions — greedy decoding")
        plt.tight_layout(); plt.show()
        print(f"{correct}/{len(test_indices)} captions name the correct shape")
        """),
        md("""
        ### Teacher forcing and exposure bias

        During training the decoder receives the **ground-truth** previous token. At
        inference it receives its **own** previous prediction.

        That mismatch is *exposure bias*: the model has never practised recovering from
        its own mistakes, so one bad token early can derail the whole sequence. Scheduled
        sampling — occasionally feeding the model's own output during training — is one
        partial remedy.
        """),
        section("6. Where this is going", """
        Everything above is sequential: timestep `t` cannot be computed before `t-1`. On
        modern hardware built for parallelism, that is the binding constraint.

        And a single fixed-size hidden vector has to carry the entire history.

        Attention (Lecture 13) removes both limits: every position sees every other
        position directly, and all positions compute at once.
        """),
        code("""
        # Measure the sequential bottleneck: time scales linearly with length and
        # cannot be parallelised away.
        import time

        lengths = [16, 32, 64, 128, 256]
        rnn_times = []
        rnn = nn.LSTM(64, 128, batch_first=True)
        for L in lengths:
            x = torch.randn(32, L, 64)
            for _ in range(3):
                rnn(x)
            t0 = time.time()
            for _ in range(10):
                rnn(x)
            rnn_times.append((time.time() - t0) / 10 * 1000)

        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(lengths, rnn_times, marker="o", lw=2, color="#7a3ea8")
        ax.set_xlabel("sequence length"); ax.set_ylabel("forward pass (ms)")
        ax.set_title("LSTM cost is linear in length — and strictly sequential")
        ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()

        for L, t in zip(lengths, rnn_times):
            print(f"  length {L:>4}: {t:6.2f} ms")
        """),
        todo("1", "RNN cell from scratch", """
        Implement a vanilla RNN cell with explicit weight matrices. Verify one timestep
        against `torch.nn.RNNCell` with copied weights.
        """),
        todo_cell(),
        todo("2", "LSTM cell from scratch", """
        Implement all four gates explicitly. Verify against `torch.nn.LSTMCell` with copied
        weights, and print the gate activations for one input to show what each gate is
        doing.
        """),
        todo_cell(),
        todo("3", "Character language model", """
        Train a character-level LSTM on `corpus.txt`. Report perplexity and sample 300
        characters at temperatures 0.5, 1.0 and 1.5. Comment on the difference.
        """),
        todo_cell(),
        todo("4", "Long-range copy task", """
        Build a task requiring the model to reproduce a token seen `T` steps earlier. Plot
        accuracy against `T` for a vanilla RNN and an LSTM, for `T` in {5, 10, 25, 50, 100}.
        """),
        todo_cell(),
        todo("5", "Gradient flow comparison", """
        Measure the gradient norm at timestep 0 for both architectures on a 100-step
        sequence. Report the ratio and connect it to your copy-task result.
        """),
        todo_cell(),
        todo("6", "Image captioning", """
        Train a CNN encoder with an LSTM decoder on the shapes captions. Generate captions
        for eight test images with greedy decoding and report how many are factually
        correct.
        """),
        todo_cell(),
        todo("7", "Beam search *(stretch)*", """
        Implement beam search with width 3 and length normalisation. Compare its captions
        against greedy decoding on the same images.
        """),
        todo_cell(),
        todo("8", "Gradient clipping study *(stretch)*", """
        Train the vanilla RNN without clipping until the loss becomes `NaN`. Add clipping
        at norms 0.5, 1 and 5 and report which values keep training stable.
        """),
        todo_cell(),
    ]
