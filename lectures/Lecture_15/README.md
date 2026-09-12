# Lecture 15 — Generative Models: Autoencoders, VAEs, GANs and Diffusion

*Learning the distribution, not just the decision boundary.*

Part IV - Attention, Generation and Production · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_15_Generative_Models_Autoencoders_VAEs_GANs_and_Diffusion.pptx` | Lecture deck, ready to present |
| `notebooks/L15_Generative_Models_Autoencoders_VAEs_GANs_and.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_15_Tasks.md` | 7 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Contrast discriminative and generative modelling and the trade-offs between the major families.
- Derive the VAE evidence lower bound and implement the reparameterisation trick.
- Train a GAN, recognise mode collapse, and apply the standard stabilisation techniques.
- Explain the forward and reverse diffusion processes and implement a minimal DDPM.

## Lecture outline

1. **The generative families** — Autoregressive (PixelCNN, GPT): exact likelihood, slow sequential sampling.
2. **Autoencoders** — Encoder compresses to a bottleneck; decoder reconstructs. Trained on reconstruction error.
3. **Variational autoencoders** — Encode to a distribution - a mean and a variance - rather than a point.
4. **Generative adversarial networks** — Generator versus discriminator, a minimax game with a Nash equilibrium as the goal.
5. **Diffusion models** — Forward process: add Gaussian noise over T steps until the image is pure noise. Fixed, no learning.
6. **Choosing, and the ethics** — Need likelihoods? Autoregressive or flows. Need speed? VAE or GAN. Need quality? Diffusion.

## The lab

Train an autoencoder and visualise its latent space; implement a VAE with the reparameterisation trick and interpolate between latent codes; train a DCGAN on shapes and deliberately induce and then fix mode collapse; implement a minimal DDPM with a small U-Net and visualise the reverse process step by step.

**Data:** `shapes_32.npz, digits_8x8.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*.
- Goodfellow et al. (2014), *Generative Adversarial Networks*.
- Ho et al. (2020), *Denoising Diffusion Probabilistic Models*.
- Radford et al. (2016), *DCGAN*.
- Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*.
