"""
VAE Demo: Train on MNIST, sample new digits from noise.
========================================================

Quick start:
    pip install torch torchvision matplotlib
    python vae_demo.py

What this program does:
  1. Downloads the MNIST dataset (28×28 grayscale digits, 60,000 training images).
  2. Trains the VAE from Part 5 of the introduction for 30 epochs.
  3. Every 5 epochs, generates a grid of digits sampled from pure noise (N(0,I)).
  4. After training, produces two final comparison images.

What the program outputs:
  samples/
  ├── epoch_001.png           Gray blobs — the VAE hasn't learned anything yet
  ├── epoch_005.png           Rough digit shapes emerging
  ├── epoch_010–030.png       Progressive refinement of generated digits
  ├── final_reconstruction.png 10 real digits (top) next to their VAE
  │                           reconstructions (bottom). Shows how much detail
  │                           the compression preserves (and loses).
  └── final_sampling.png      64 brand-new digits generated from
                              torch.randn(64, 20) — pure imagination.

How to read the training log:
  The console prints one line per epoch with three columns:

      Epoch      Recon         KL   KL/Recon
          1    18993.2     1951.5     0.1027
          5    10903.6     3135.7     0.2876
         10    10369.5     3195.8     0.3082
         20    10056.4     3229.0     0.3211
         30     9916.6     3235.3     0.3262

  Recon (reconstruction loss):
    How many "bits" of error between the input digit and the VAE's
    reconstruction. Drops rapidly as the VAE learns to compress and
    reconstruct digits faithfully. Lower is better.

  KL (KL divergence):
    How far the encoder's distribution is from N(0,1). If this number
    is RISING and stabilizing, training is healthy — the encoder is
    learning to balance information preservation against the prior.
    If it's DROPPING toward zero, you have posterior collapse — the
    decoder is ignoring the latent code. See Part 6 of the introduction.

  KL/Recon ratio:
    The balance of the two forces. Values around 0.1–0.3 are typical
    for a well-trained VAE on MNIST. Much higher (>1.0) means samples
    will be good but reconstructions blurry. Near zero means collapse.

  The sample grids at epochs 1, 5, 10, 15, 20, 25, 30 let you watch
  the VAE's imagination improve. Early epochs: noise. Mid-training:
  shapes coalescing. Late training: sharp, diverse digits.

What the final images tell you:
  final_reconstruction.png — Top row are real MNIST digits the VAE
  has never seen (test set). Bottom row are the VAE's reconstructions.
  Notice the reconstructions are slightly blurrier than the originals
  (the Gaussian likelihood averages over plausible pixel values). This
  is the VAE's key weakness — but for RL, blurry is fine because the
  policy only needs the latent code z, not the reconstruction.

  final_sampling.png — 64 digits generated from torch.randn(64, 20).
  These digits never existed in the training set. Each one was "dreamed
  up" by the decoder from a random 20-number vector. If most look like
  recognizable digits, the latent space is well-structured. If they're
  all gray blobs, the VAE may have collapsed. If they're all the same
  digit, the latent space lacks diversity.

Experiments to try:
  - Change latent_dim to 2 and re-run. The samples will be worse
    (not enough capacity), but you can plot the 2D latent space
    directly with a scatter plot — every MNIST digit becomes a point,
    colored by its label. This is the best way to *see* the latent
    structure.
  - Change latent_dim to 100. Better reconstructions, but samples
    may be slightly less coherent (the prior is spread thinner).
  - Add KL annealing: multiply the KL term by min(1.0, epoch/10).
    The first 10 epochs focus on reconstruction, then the latent
    space is gradually shaped. This often produces sharper digits.
  - Train on FashionMNIST (change datasets.MNIST to datasets.FashionMNIST)
    and see what fashion items the VAE dreams up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# 1. The VAE (from Part 5 of the introduction)
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Encoder: x -> mu, logvar
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder: z -> x_recon
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x_recon, x, mu, logvar):
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl

# ---------------------------------------------------------------------------
# 2. Load MNIST
# ---------------------------------------------------------------------------

def get_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),                    # [0,255] -> [0.0, 1.0]
    ])
    train_data = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------------------------
# 3. Sample helper: decode random noise into a grid of images
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_and_plot(model, epoch, latent_dim=20, n_rows=8, save_dir='samples'):
    """Sample z ~ N(0,I), decode, and save a grid."""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    z = torch.randn(n_rows * n_rows, latent_dim)
    images = model.decode(z).view(-1, 28, 28).cpu()

    fig, axes = plt.subplots(n_rows, n_rows, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')

    plt.suptitle(f'Epoch {epoch}: digits sampled from N(0,I)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(path)
    plt.close()
    print(f'  Saved {path}')

# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train(epochs=30, latent_dim=20, batch_size=128, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')

    train_loader = get_mnist(batch_size)
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f'Training VAE: {epochs} epochs, latent_dim={latent_dim}')
    print(f'{"Epoch":>5} {"Recon":>10} {"KL":>10} {"KL/Recon":>10}')
    print('-' * 39)

    for epoch in range(1, epochs + 1):
        model.train()
        total_recon, total_kl, n_batches = 0, 0, 0

        for data, _ in train_loader:
            x = data.view(-1, 784).to(device)

            x_recon, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_recon, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon.item()
            total_kl += kl.item()
            n_batches += 1

        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        print(f'{epoch:5d} {avg_recon:10.1f} {avg_kl:10.1f} '
              f'{avg_kl/avg_recon:10.4f}')

        # Sample every 5 epochs + first and last
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            sample_and_plot(model, epoch, latent_dim)

    return model

# ---------------------------------------------------------------------------
# 5. Final demo: side-by-side comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def final_demo(model, latent_dim=20):
    """Show original MNIST digits next to their VAE reconstructions,
    plus a pure-sampling grid."""
    model.eval()

    # Load a few test images
    test_data = datasets.MNIST(
        root='./data', train=False, download=True,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(test_data, batch_size=10, shuffle=True)
    x_real, _ = next(iter(loader))
    x_real = x_real.view(-1, 784)

    # Reconstruct
    x_recon, _, _ = model(x_real)
    x_recon = x_recon.view(-1, 28, 28)
    x_real_img = x_real.view(-1, 28, 28)

    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for i in range(10):
        axes[0, i].imshow(x_real_img[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(x_recon[i], cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
    plt.suptitle('VAE Reconstruction', fontsize=14)
    plt.tight_layout()
    plt.savefig('samples/final_reconstruction.png')
    print('Saved samples/final_reconstruction.png')
    plt.close()

    # Pure sampling: random noise -> digits
    z = torch.randn(64, latent_dim)
    images = model.decode(z).view(-1, 28, 28).cpu()

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.suptitle('New Digits Sampled from N(0,I) — Pure Imagination', fontsize=14)
    plt.tight_layout()
    plt.savefig('samples/final_sampling.png')
    print('Saved samples/final_sampling.png')
    plt.close()


if __name__ == '__main__':
    model = train(epochs=30, latent_dim=20)
    final_demo(model, latent_dim=20)
    print('\nDone. Check the samples/ folder for generated images.')
