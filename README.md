# The VAE Story: From Pixels to Robot Policies

A narrative introduction to Variational Autoencoders and their applications in robotics reinforcement learning — written for students.

## What's Here

| File | What it is |
|------|-----------|
| `vae_introduction.md` | 12-part narrative tutorial: from "what is a VAE?" to real robot deployments in 2026 |
| `vae_demo.py` | Train a VAE on MNIST, watch it learn to dream new digits from noise |
| `CLAUDE.md` | Project guidance for Claude Code |

## Quick Start

```bash
# Read the tutorial (rendered online):
#   https://tonyinvent.github.io/vae_learning/

# Run the demo:
pip install torch torchvision matplotlib
python vae_demo.py
# Check samples/ for generated digits
```

## Table of Contents (from the tutorial)

1. **The Robot's Problem** — Why 12,288 pixels describing 6 numbers is the fundamental challenge
2. **First Attempt** — The standard autoencoder and why it fails
3. **The VAE Insight** — Learn a distribution, not a point
4. **Making It Work** — Reparameterization trick and closed-form KL divergence
5. **The Complete VAE** — Code and training loop with line-by-line walkthrough
6. **When Things Go Wrong** — Posterior collapse: diagnosis and fixes
7. **Better Latent Spaces** — β-VAE, VQ-VAE, q-VAE
8. **The Leap to RL** — Why structured latent spaces matter for decision-making
9. **World Models** — The 867-parameter controller that beat humans at CarRacing
10. **Dreamer** — How the VAE principle evolved into a universal world model
11. **Real Robots** — Navigation at 30 lux, terrain locomotion, cross-embodiment transfer
12. **What We Haven't Solved** — Open problems and where to go next

## License

MIT
