# Lecture 15 — Tasks
## Generative Models: Autoencoders, VAEs, GANs and Diffusion

> Learning the distribution, not just the decision boundary.

**Course:** DL-601 Deep Learning · **Part:** Part IV - Attention, Generation and Production  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 16

---

## What you should be able to do after this

- Contrast discriminative and generative modelling and the trade-offs between the major families.
- Derive the VAE evidence lower bound and implement the reparameterisation trick.
- Train a GAN, recognise mode collapse, and apply the standard stabilisation techniques.
- Explain the forward and reverse diffusion processes and implement a minimal DDPM.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_15/notebooks/L15_Generative_Models_Autoencoders_VAEs_GANs_and.ipynb
```

**Data used:** `shapes_32.npz, digits_8x8.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (7 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Autoencoder and latent space

Train an autoencoder with a 2-dimensional bottleneck on shapes. Scatter-plot the latent codes coloured by class and state whether the classes separate.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — VAE with reparameterisation

Implement the encoder producing mu and logvar, the reparameterisation trick, and the ELBO with a closed-form Gaussian KL. Plot reconstruction loss and KL separately across training.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Sample and interpolate

Sample 64 images from the VAE prior. Then linearly interpolate between two latent codes in 10 steps and show the decoded sequence.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — beta-VAE

Train with beta in {0.5, 1, 4, 10}. Show the reconstruction/KL trade-off in a table and identify where posterior collapse begins.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — DCGAN

Implement a DCGAN generator and discriminator. Train on shapes and produce a sample grid every five epochs to show the progression.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Mode collapse

Induce mode collapse by over-training the generator relative to the discriminator. Demonstrate it quantitatively with a class histogram over 1000 samples, then fix it and show the histogram recover.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 7 — Minimal DDPM

Implement the cosine noise schedule, the forward diffusion q(x_t | x_0) in closed form, and a small noise-prediction U-Net. Train it and visualise the reverse process at t = T, 3T/4, T/2, T/4, 0.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — DDIM sampling

Implement deterministic DDIM sampling. Compare sample quality and wall-clock time at 1000, 100, 50 and 10 steps.

### Stretch 2 — Conditional generation

Add class conditioning to your diffusion model via an embedding added to the time embedding. Generate each class on demand and verify with your Lecture 6 classifier.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L15.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*.
- Goodfellow et al. (2014), *Generative Adversarial Networks*.
- Ho et al. (2020), *Denoising Diffusion Probabilistic Models*.
- Radford et al. (2016), *DCGAN*.
- Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*.

---

**Next lecture —** 16: Multimodal Learning and Production ML Systems. Joining vision to language, and shipping the result responsibly.
