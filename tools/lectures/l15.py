"""Lecture 15 notebook — generative models."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        section("0. Discriminative versus generative", """
        Everything so far modelled `p(y | x)` — given an image, which class. A generative
        model learns `p(x)` — what images look like at all — and lets you sample new ones.
        """),
        code("""
        data = load_shapes()
        X_train, y_train = data["train"]
        X_test,  y_test  = data["test"]
        CLASSES = data["classes"]
        print("training images:", tuple(X_train.shape), " range [0, 1]")
        """),
        section("1. Autoencoders", """
        Compress to a bottleneck, reconstruct, and train on reconstruction error.

        A plain autoencoder is **not** generative: nothing constrains the shape of the
        latent space, so there is no distribution to sample from. It is still useful for
        compression, denoising and anomaly detection.
        """),
        code("""
        class Autoencoder(nn.Module):
            def __init__(self, latent=2):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),      # 16
                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),     # 8
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),     # 4
                    nn.Flatten(), nn.Linear(64 * 16, latent))
                self.decoder_in = nn.Linear(latent, 64 * 16)
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2), nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())

            def decode(self, z):
                return self.decoder(self.decoder_in(z).view(-1, 64, 4, 4))

            def forward(self, x):
                z = self.encoder(x)
                return self.decode(z), z
        """),
        code("""
        # ~1 minute.
        from torch.utils.data import TensorDataset, DataLoader

        set_seed(0)
        ae = Autoencoder(latent=2)
        opt = torch.optim.Adam(ae.parameters(), lr=2e-3)
        # Generative models here learn p(x), and 3000 images sample that distribution
        # perfectly well. Halving the data halves every training cell in the lecture.
        GEN_N = 3000
        loader = DataLoader(TensorDataset(X_train[:GEN_N]), batch_size=128, shuffle=True)

        for epoch in range(12):
            ae.train(); total = 0.0
            for (xb,) in loader:
                opt.zero_grad()
                recon, _ = ae(xb)
                loss = F.mse_loss(recon, xb)
                loss.backward(); opt.step()
                total += loss.item() * len(xb)
            if (epoch + 1) % 4 == 0:
                print(f"epoch {epoch+1:2d}  MSE {total/GEN_N:.5f}")
        """),
        code("""
        ae.eval()
        with torch.no_grad():
            recon, z_all = ae(X_test[:8])
            _, z_full = ae(X_test[:800])

        fig, axes = plt.subplots(2, 8, figsize=(14, 3.8))
        for j in range(8):
            axes[0, j].imshow(X_test[j].permute(1, 2, 0).numpy()); axes[0, j].axis("off")
            axes[1, j].imshow(recon[j].permute(1, 2, 0).numpy()); axes[1, j].axis("off")
        axes[0, 0].set_ylabel("original"); axes[1, 0].set_ylabel("reconstructed")
        fig.suptitle("Autoencoder reconstructions through a 2-dimensional bottleneck")
        plt.tight_layout(); plt.show()
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        for c, name in enumerate(CLASSES):
            m = (y_test[:800] == c).numpy()
            a1.scatter(z_full[m, 0], z_full[m, 1], s=10, label=name, alpha=.7)
        a1.set_title("Autoencoder latent space\\n(no structure imposed — note the gaps)")
        a1.set_xlabel("z1"); a1.set_ylabel("z2"); a1.legend(); a1.grid(alpha=.3)

        # Sampling from a plain autoencoder: mostly nonsense, because most of the
        # latent space was never used during training.
        with torch.no_grad():
            random_z = torch.randn(8, 2) * z_full.std(0)
            junk = ae.decode(random_z)
        for j in range(8):
            ax = a2.inset_axes([(j % 4) * 0.25, 0.5 - (j // 4) * 0.45, 0.22, 0.4])
            ax.imshow(junk[j].permute(1, 2, 0).numpy()); ax.axis("off")
        a2.axis("off"); a2.set_title("Decoding random latent points\\n"
                                     "— this is why an autoencoder is not generative")
        plt.tight_layout(); plt.show()
        """),
        section("2. Variational autoencoders", """
        A VAE encodes to a **distribution** — a mean and a variance — and adds a KL term
        pulling that distribution toward `N(0, I)`. That regularisation is what makes the
        latent space samplable.

        $$\\mathcal{L} = \\underbrace{\\mathbb{E}[\\log p(x|z)]}_{\\text{reconstruction}}
          - \\underbrace{D_{KL}(q(z|x) \\,\\|\\, p(z))}_{\\text{regulariser}}$$

        The reparameterisation trick — `z = μ + σ·ε` with `ε ~ N(0, I)` — is what makes
        this differentiable. Sampling is not differentiable; sampling `ε` and then doing
        arithmetic is.
        """),
        code("""
        class VAE(nn.Module):
            def __init__(self, latent=8):
                super().__init__()
                self.latent = latent
                self.enc = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
                self.fc_mu = nn.Linear(128 * 16, latent)
                self.fc_logvar = nn.Linear(128 * 16, latent)
                self.fc_dec = nn.Linear(latent, 128 * 16)
                self.dec = nn.Sequential(
                    nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                    nn.Upsample(scale_factor=2), nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid())

            def encode(self, x):
                h = self.enc(x)
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterise(self, mu, logvar):
                \"\"\"z = mu + sigma * eps. The randomness is in eps, which carries no
                gradient — so the gradient flows cleanly through mu and sigma.\"\"\"
                std = torch.exp(0.5 * logvar)
                return mu + std * torch.randn_like(std)

            def decode(self, z):
                return self.dec(self.fc_dec(z).view(-1, 128, 4, 4))

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterise(mu, logvar)
                return self.decode(z), mu, logvar


        def vae_loss(recon, x, mu, logvar, beta=1.0):
            \"\"\"Returns (total, reconstruction, KL). KL has a closed form for Gaussians.\"\"\"
            rec = F.binary_cross_entropy(recon, x, reduction="sum") / len(x)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
            return rec + beta * kl, rec, kl
        """),
        code("""
        # ~2 minutes.
        set_seed(0)
        vae = VAE(latent=8)
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
        rec_curve, kl_curve = [], []

        for epoch in range(15):
            vae.train(); r_tot = k_tot = 0.0
            for (xb,) in loader:
                opt.zero_grad()
                recon, mu, logvar = vae(xb)
                loss, rec, kl = vae_loss(recon, xb, mu, logvar)
                loss.backward(); opt.step()
                r_tot += rec.item() * len(xb); k_tot += kl.item() * len(xb)
            rec_curve.append(r_tot / GEN_N); kl_curve.append(k_tot / GEN_N)
            if (epoch + 1) % 5 == 0:
                print(f"epoch {epoch+1:2d}  recon {rec_curve[-1]:8.2f}  KL {kl_curve[-1]:7.3f}")
        """),
        code("""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
        a1.plot(rec_curve, color="#1f6fb2"); a1.set_title("Reconstruction term")
        a2.plot(kl_curve, color="#c0392b"); a2.set_title("KL term")
        for a in (a1, a2):
            a.set_xlabel("epoch"); a.grid(alpha=.3)
        fig.suptitle("The two halves of the ELBO pull in opposite directions")
        plt.tight_layout(); plt.show()
        """),
        code("""
        # Sampling from the prior — this is what the VAE bought us.
        vae.eval()
        with torch.no_grad():
            samples = vae.decode(torch.randn(32, vae.latent))

        show_grid(samples, n=32, ncols=8, title="VAE samples from the N(0, I) prior")
        plt.show()
        print("Blurry, and that is inherent: a pixel-wise Gaussian likelihood averages")
        print("over every plausible output rather than committing to one.")
        """),
        code("""
        # Latent interpolation — a real test of whether the space is meaningful.
        with torch.no_grad():
            i, j = 0, int((y_test != y_test[0]).nonzero()[0])
            mu_a, _ = vae.encode(X_test[i:i+1])
            mu_b, _ = vae.encode(X_test[j:j+1])
            alphas = torch.linspace(0, 1, 10).view(-1, 1)
            zs = (1 - alphas) * mu_a + alphas * mu_b
            interp = vae.decode(zs)

        fig, axes = plt.subplots(1, 10, figsize=(15, 1.9))
        for k, ax in enumerate(axes):
            ax.imshow(interp[k].permute(1, 2, 0).numpy()); ax.axis("off")
            ax.set_title(f"{alphas[k].item():.1f}", fontsize=8)
        fig.suptitle(f"Interpolating latents: {CLASSES[y_test[i]]} -> {CLASSES[y_test[j]]}")
        plt.tight_layout(); plt.show()
        print("Smooth interpolation means the latent space is continuous rather than a")
        print("lookup table of memorised training images.")
        """),
        code("""
        # beta-VAE: the reconstruction/KL trade-off, and where posterior collapse begins.
        # ~2 minutes.
        beta_results = {}
        for beta in (0.5, 1.0, 4.0, 10.0):
            set_seed(0)
            v = VAE(latent=8)
            o = torch.optim.Adam(v.parameters(), lr=1e-3)
            for _ in range(5):
                for (xb,) in loader:
                    o.zero_grad()
                    recon, mu, logvar = v(xb)
                    loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta=beta)
                    loss.backward(); o.step()
            # An "active" latent dimension is one with non-trivial variance in mu.
            with torch.no_grad():
                mu_all, _ = v.encode(X_test[:500])
            active = int((mu_all.var(0) > 0.02).sum())
            beta_results[beta] = (rec.item(), kl.item(), active)
            print(f"beta={beta:>5}  recon {rec.item():8.2f}  KL {kl.item():7.3f}  "
                  f"active latent dims {active}/8")

        print("\\nAs beta rises the KL term wins, the latent carries less, and eventually")
        print("dimensions go silent — posterior collapse.")
        """),
        section("3. Generative adversarial networks", """
        Two networks in a minimax game: a generator trying to produce convincing images
        and a discriminator trying to tell real from fake.

        Sharp samples, and genuinely unstable training.
        """),
        code("""
        class Generator(nn.Module):
            def __init__(self, z_dim=64):
                super().__init__()
                self.z_dim = z_dim
                self.fc = nn.Linear(z_dim, 128 * 16)
                self.net = nn.Sequential(
                    nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, padding=1),
                    nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Upsample(scale_factor=2), nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid())

            def forward(self, z):
                return self.net(self.fc(z).view(-1, 128, 4, 4))


        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
                    nn.Conv2d(32, 64, 4, stride=2, padding=1),
                    nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                    nn.Flatten(), nn.Linear(128 * 16, 1))

            def forward(self, x):
                return self.net(x)
        """),
        code("""
        # ~4 minutes. The longest cell in this lecture.
        set_seed(0)
        G, D = Generator(), Discriminator()
        opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

        EPOCHS = 10
        fixed_z = torch.randn(32, G.z_dim)
        g_losses, d_losses, snapshots = [], [], {}

        for epoch in range(EPOCHS):
            g_tot = d_tot = 0.0; n = 0
            for (xb,) in loader:
                bs = len(xb)
                real_label = torch.full((bs, 1), 0.9)    # label smoothing
                fake_label = torch.zeros(bs, 1)

                # --- discriminator ---
                opt_d.zero_grad()
                fake = G(torch.randn(bs, G.z_dim))
                d_loss = bce(D(xb), real_label) + bce(D(fake.detach()), fake_label)
                d_loss.backward(); opt_d.step()

                # --- generator: non-saturating loss ---
                # Maximising log D(G(z)) rather than minimising log(1 - D(G(z))),
                # because the latter has almost no gradient early on.
                opt_g.zero_grad()
                g_loss = bce(D(fake), torch.ones(bs, 1))
                g_loss.backward(); opt_g.step()

                g_tot += g_loss.item(); d_tot += d_loss.item(); n += 1

            g_losses.append(g_tot / n); d_losses.append(d_tot / n)
            if (epoch + 1) % 5 == 0:
                G.eval()
                with torch.no_grad():
                    snapshots[epoch + 1] = G(fixed_z).clone()
                G.train()
                print(f"epoch {epoch+1:2d}  G {g_losses[-1]:.4f}  D {d_losses[-1]:.4f}")
        """),
        code("""
        fig, axes = plt.subplots(len(snapshots), 8, figsize=(14, 2.0 * len(snapshots)))
        axes = np.atleast_2d(axes)
        for r, (ep, imgs) in enumerate(snapshots.items()):
            for c in range(8):
                axes[r, c].imshow(imgs[c].permute(1, 2, 0).numpy()); axes[r, c].axis("off")
            axes[r, 0].set_title(f"epoch {ep}", fontsize=9, loc="left")
        fig.suptitle("GAN samples over training — the same fixed z each time")
        plt.tight_layout(); plt.show()
        """),
        code("""
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(g_losses, label="generator", color="#7a3ea8")
        ax.plot(d_losses, label="discriminator", color="#c16a1c")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title("GAN losses — note these do NOT decrease like a normal loss")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()

        print("A GAN loss curve tells you almost nothing about sample quality. The two")
        print("networks are adversaries; equilibrium, not minimisation, is the goal.")
        print("You have to look at the samples.")
        """),
        section("4. Mode collapse", """
        The generator's failure mode: find a handful of outputs that fool the
        discriminator and stop exploring.

        Detect it by classifying many samples with the Lecture 6 classifier and looking at
        the histogram. A healthy generator produces all four classes.
        """),
        code("""
        # A quick classifier to judge the samples.
        set_seed(0)
        judge = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 16, 4))
        train(judge, (X_train, y_train), (X_test, y_test), epochs=5, lr=2e-3, verbose=False)
        _, judge_acc = evaluate(judge, (X_test, y_test))
        print(f"judge classifier accuracy: {judge_acc:.3f}")


        @torch.no_grad()
        def class_histogram(generator, n=1000):
            generator.eval(); judge.eval()
            out = []
            for i in range(0, n, 250):
                out.append(judge(generator(torch.randn(250, generator.z_dim))).argmax(1))
            generator.train()
            return torch.bincount(torch.cat(out), minlength=4).float() / n

        healthy = class_histogram(G)
        print("\\nclass distribution of GAN samples:")
        for name, f in zip(CLASSES, healthy):
            print(f"  {name:<10} {f:.3f}")
        """),
        code("""
        # Now induce collapse deliberately: over-train the generator relative to D.
        # ~2 minutes.
        set_seed(0)
        G_bad, D_bad = Generator(), Discriminator()
        opt_gb = torch.optim.Adam(G_bad.parameters(), lr=2e-3, betas=(0.5, 0.999))  # 10x too high
        opt_db = torch.optim.Adam(D_bad.parameters(), lr=1e-5, betas=(0.5, 0.999))  # crippled

        for epoch in range(6):
            for (xb,) in loader:
                bs = len(xb)
                opt_db.zero_grad()
                fake = G_bad(torch.randn(bs, G_bad.z_dim))
                (bce(D_bad(xb), torch.ones(bs, 1))
                 + bce(D_bad(fake.detach()), torch.zeros(bs, 1))).backward()
                opt_db.step()
                # Three generator steps per discriminator step.
                for _ in range(3):
                    opt_gb.zero_grad()
                    f = G_bad(torch.randn(bs, G_bad.z_dim))
                    bce(D_bad(f), torch.ones(bs, 1)).backward()
                    opt_gb.step()

        collapsed = class_histogram(G_bad)
        """),
        code("""
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4),
                                         gridspec_kw={"width_ratios": [1, 1, 1.3]})
        x = np.arange(4)
        a1.bar(x, healthy.numpy(), color="#1b866b")
        a1.set_xticks(x, CLASSES, rotation=20); a1.set_ylim(0, 1)
        a1.set_title(f"Balanced training\\nentropy {-(healthy*(healthy+1e-9).log()).sum():.3f}")
        a1.set_ylabel("fraction of 1000 samples")

        a2.bar(x, collapsed.numpy(), color="#c0392b")
        a2.set_xticks(x, CLASSES, rotation=20); a2.set_ylim(0, 1)
        a2.set_title(f"Over-trained generator\\nentropy {-(collapsed*(collapsed+1e-9).log()).sum():.3f}")

        with torch.no_grad():
            bad_samples = G_bad(torch.randn(8, G_bad.z_dim))
        for j in range(8):
            ax = a3.inset_axes([(j % 4) * 0.25, 0.5 - (j // 4) * 0.45, 0.22, 0.4])
            ax.imshow(bad_samples[j].permute(1, 2, 0).numpy()); ax.axis("off")
        a3.axis("off"); a3.set_title("Collapsed generator's samples\\n(note how similar they are)")
        plt.tight_layout(); plt.show()

        print(f"maximum possible entropy for 4 classes: {np.log(4):.3f}")
        print("Lower entropy means the generator has stopped covering the distribution.")
        """),
        section("5. Diffusion", """
        The idea that took over the field.

        **Forward**: add Gaussian noise over `T` steps until the image is pure noise. This
        is fixed — nothing is learned.

        **Reverse**: train a network to predict the noise that was added. Then start from
        noise and subtract, step by step.

        Training is a plain regression on the noise, which is why it is so much more
        stable than a GAN.
        """),
        code("""
        T = 200

        def cosine_schedule(timesteps, s=0.008):
            \"\"\"Cosine noise schedule — degrades information more gradually than linear.\"\"\"
            steps = torch.arange(timesteps + 1, dtype=torch.float32) / timesteps
            alphas_cumprod = torch.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(1e-4, 0.999)

        betas = cosine_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_acp = alphas_cumprod.sqrt()
        sqrt_1macp = (1 - alphas_cumprod).sqrt()

        def q_sample(x0, t, noise=None):
            \"\"\"Jump straight to timestep t in closed form — no loop needed.\"\"\"
            noise = torch.randn_like(x0) if noise is None else noise
            return (sqrt_acp[t].view(-1, 1, 1, 1) * x0
                    + sqrt_1macp[t].view(-1, 1, 1, 1) * noise), noise

        fig, axes = plt.subplots(1, 8, figsize=(15, 2.1))
        img0 = X_train[1:2]
        for ax, t in zip(axes, [0, 20, 50, 80, 110, 140, 170, 199]):
            noisy, _ = q_sample(img0, torch.tensor([t]))
            ax.imshow(noisy[0].permute(1, 2, 0).clamp(0, 1).numpy()); ax.axis("off")
            ax.set_title(f"t={t}", fontsize=9)
        fig.suptitle("Forward diffusion — fixed, requires no training")
        plt.tight_layout(); plt.show()
        """),
        code("""
        class TimeEmbedding(nn.Module):
            \"\"\"Sinusoidal embedding of the timestep — the same trick as positional
            encoding, because the network must know how noisy its input is.\"\"\"

            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, t):
                half = self.dim // 2
                freqs = torch.exp(-np.log(10000) * torch.arange(half).float() / half)
                args = t[:, None].float() * freqs[None]
                return torch.cat([args.sin(), args.cos()], dim=-1)


        class TinyUNet(nn.Module):
            \"\"\"Small noise-prediction network. Input: noisy image + t. Output: the noise.\"\"\"

            def __init__(self, base=32, tdim=64):
                super().__init__()
                self.time = nn.Sequential(TimeEmbedding(tdim), nn.Linear(tdim, tdim), nn.SiLU())
                self.t1 = nn.Linear(tdim, base)
                self.t2 = nn.Linear(tdim, base * 2)
                self.t3 = nn.Linear(tdim, base * 2)

                self.down1 = nn.Sequential(nn.Conv2d(3, base, 3, padding=1),
                                           nn.GroupNorm(8, base), nn.SiLU())
                self.down2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
                                           nn.GroupNorm(8, base * 2), nn.SiLU())
                self.mid = nn.Sequential(nn.Conv2d(base * 2, base * 2, 3, padding=1),
                                         nn.GroupNorm(8, base * 2), nn.SiLU())
                self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                        nn.Conv2d(base * 2, base, 3, padding=1),
                                        nn.GroupNorm(8, base), nn.SiLU())
                self.out = nn.Conv2d(base * 2, 3, 3, padding=1)

            def forward(self, x, t):
                temb = self.time(t)
                h1 = self.down1(x) + self.t1(temb)[:, :, None, None]
                h2 = self.down2(h1) + self.t2(temb)[:, :, None, None]
                h2 = self.mid(h2) + self.t3(temb)[:, :, None, None]
                u = self.up(h2)
                return self.out(torch.cat([u, h1], dim=1))     # skip connection

        unet = TinyUNet()
        print("noise prediction:", tuple(unet(torch.randn(2, 3, 32, 32), torch.tensor([5, 100])).shape))
        print(f"parameters: {count_parameters(unet):,}")
        """),
        code("""
        # ~4 minutes. The training objective is simply MSE on the noise.
        set_seed(0)
        unet = TinyUNet()
        opt = torch.optim.Adam(unet.parameters(), lr=2e-3)
        diff_losses = []

        for epoch in range(15):
            unet.train(); total = 0.0
            for (xb,) in loader:
                t = torch.randint(0, T, (len(xb),))
                noisy, noise = q_sample(xb, t)
                opt.zero_grad()
                loss = F.mse_loss(unet(noisy, t), noise)
                loss.backward(); opt.step()
                total += loss.item() * len(xb)
            diff_losses.append(total / GEN_N)
            if (epoch + 1) % 5 == 0:
                print(f"epoch {epoch+1:2d}  noise-prediction MSE {diff_losses[-1]:.5f}")

        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(diff_losses, color="#7a3ea8")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE on predicted noise")
        ax.set_title("Diffusion training — a plain, stable regression")
        ax.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        """),
        code("""
        @torch.no_grad()
        def sample_ddpm(model, n=8, capture=()):
            \"\"\"Start from pure noise and denoise step by step.\"\"\"
            model.eval()
            x = torch.randn(n, 3, 32, 32)
            frames = {}
            for t in reversed(range(T)):
                tt = torch.full((n,), t, dtype=torch.long)
                eps = model(x, tt)
                a, ac, b = alphas[t], alphas_cumprod[t], betas[t]
                mean = (x - (b / (1 - ac).sqrt()) * eps) / a.sqrt()
                x = mean if t == 0 else mean + b.sqrt() * torch.randn_like(x)
                if t in capture:
                    frames[t] = x.clone()
            frames[0] = x
            return x, frames

        set_seed(0)
        samples, frames = sample_ddpm(unet, n=8, capture=(199, 150, 100, 50, 10))

        keys = sorted(frames, reverse=True)
        fig, axes = plt.subplots(len(keys), 8, figsize=(14, 1.9 * len(keys)))
        for r, t in enumerate(keys):
            for c in range(8):
                axes[r, c].imshow(frames[t][c].permute(1, 2, 0).clamp(0, 1).numpy())
                axes[r, c].axis("off")
            axes[r, 0].set_title(f"t = {t}", fontsize=9, loc="left")
        fig.suptitle("Reverse diffusion — noise becomes an image, one step at a time")
        plt.tight_layout(); plt.show()
        """),
        code("""
        # Judge the diffusion samples the same way we judged the GAN's.
        with torch.no_grad():
            big, _ = sample_ddpm(unet, n=32)
            judged = judge(big.clamp(0, 1)).argmax(1)
        dist = torch.bincount(judged, minlength=4).float() / len(judged)

        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.bar(np.arange(4), dist.numpy(), color="#7a3ea8")
        ax.set_xticks(np.arange(4), CLASSES, rotation=20)
        ax.set_ylabel("fraction of 64 samples")
        ax.set_title(f"Diffusion sample distribution — entropy "
                     f"{-(dist*(dist+1e-9).log()).sum():.3f} (max {np.log(4):.3f})")
        plt.tight_layout(); plt.show()
        """),
        md("""
        ### Choosing a family

        | Family | Sample quality | Sampling speed | Likelihood | Training stability |
        |---|---|---|---|---|
        | Autoregressive | High | Very slow | Exact | Stable |
        | VAE | Blurry | Fast | Lower bound | Stable |
        | GAN | Sharp | Fast | None | Difficult |
        | Diffusion | Highest | Slow (iterative) | Bound | Stable |
        | Normalising flow | Moderate | Fast | Exact | Stable |

        Diffusion won on the two axes that mattered most for images — quality and training
        stability — and DDIM plus distillation cut the sampling cost enough to make it
        practical.

        ### A note you should not skip

        These models raise real questions: consent for training data, memorisation of
        individual training examples, provenance of generated media. They are engineering
        problems as much as policy ones, and "I only trained it" is not a position that
        holds up. Know what your model was trained on.
        """),
        todo("1", "Autoencoder and latent space", """
        Train an autoencoder with a 2-dimensional bottleneck on shapes. Scatter-plot the
        latent codes coloured by class and state whether the classes separate.
        """),
        todo_cell(),
        todo("2", "VAE with reparameterisation", """
        Implement the encoder producing `mu` and `logvar`, the reparameterisation trick,
        and the ELBO with a closed-form Gaussian KL. Plot reconstruction loss and KL
        separately across training.
        """),
        todo_cell(),
        todo("3", "Sample and interpolate", """
        Sample 64 images from the VAE prior. Then interpolate linearly between two latent
        codes in 10 steps and show the decoded sequence.
        """),
        todo_cell(),
        todo("4", "beta-VAE", """
        Train with `beta` in {0.5, 1, 4, 10}. Show the reconstruction/KL trade-off in a
        table and identify where posterior collapse begins.
        """),
        todo_cell(),
        todo("5", "DCGAN", """
        Implement a DCGAN generator and discriminator. Train on shapes and produce a
        sample grid every five epochs to show the progression.
        """),
        todo_cell(),
        todo("6", "Mode collapse", """
        Induce mode collapse by over-training the generator relative to the discriminator.
        Demonstrate it quantitatively with a class histogram over 1000 samples, then fix it
        and show the histogram recover.
        """),
        todo_cell(),
        todo("7", "Minimal DDPM", """
        Implement the cosine noise schedule, the closed-form forward process
        `q(x_t | x_0)`, and a small noise-prediction U-Net. Train it and visualise the
        reverse process at `t = T, 3T/4, T/2, T/4, 0`.
        """),
        todo_cell(),
        todo("8", "DDIM sampling *(stretch)*", """
        Implement deterministic DDIM sampling. Compare sample quality and wall-clock time
        at 200, 100, 50 and 10 steps.
        """),
        todo_cell(),
        todo("9", "Conditional generation *(stretch)*", """
        Add class conditioning via an embedding added to the time embedding. Generate each
        class on demand and verify with the judge classifier.
        """),
        todo_cell(),
    ]
