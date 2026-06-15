import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuración ──────────────────────────────────────────
LATENT_DIM   = 64
BATCH_SIZE   = 128
EPOCHS       = 50
LR           = 1e-3
BETA         = 1.0      # Peso del término KL (β-VAE)
DATA_PATH    = 'data/data/frames.npy'
CKPT_DIR     = 'data/data/checkpoints'
CKPT_EVERY   = 10       # Guardar checkpoint cada N epochs
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ───────────────────────────────────────────────────────────

os.makedirs(CKPT_DIR, exist_ok=True)
print(f'Device: {DEVICE}')

# ── Arquitectura VAE ───────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16→8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 8→4
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 32→64
            nn.Sigmoid(),  # Output en [0, 1]
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        """Solo encoder — usado por DDPG"""
        mu, logvar = self.encoder(x)
        return mu  # Usamos la media como representación determinística


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ── Dataset ────────────────────────────────────────────────
print('Cargando dataset...')
data = np.load(DATA_PATH)
print(f'  Shape  : {data.shape}')
print(f'  Min/Max: {data.min():.4f} / {data.max():.4f}')

tensor_data = torch.from_numpy(data).float()
dataset = TensorDataset(tensor_data)

# Split 90% train, 10% validación
n_val   = int(len(dataset) * 0.1)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f'  Train  : {n_train} frames')
print(f'  Val    : {n_val} frames')

# ── Entrenamiento ──────────────────────────────────────────
model     = VAE(LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_losses, val_losses = [], []
best_val_loss = float('inf')

print(f'\nEntrenando VAE — {EPOCHS} epochs en {DEVICE}')
print('─' * 60)

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    # Train
    model.train()
    train_loss = 0
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss, recon_l, kl_l = vae_loss(recon, batch, mu, logvar, BETA)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validación
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(DEVICE)
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, BETA)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    elapsed = time.time() - start
    print(f'Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.2f} | val={val_loss:.2f} | {elapsed:.1f}s')

    # Checkpoint
    if epoch % CKPT_EVERY == 0:
        ckpt_path = f'{CKPT_DIR}/vae_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'latent_dim': LATENT_DIM,
        }, ckpt_path)
        print(f'  → Checkpoint guardado: {ckpt_path}')

    # Mejor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{CKPT_DIR}/vae_best.pt')

# ── Guardar modelo final ───────────────────────────────────
torch.save(model.state_dict(), f'{CKPT_DIR}/vae_final.pt')
print(f'\n✓ Modelo final guardado: {CKPT_DIR}/vae_final.pt')
print(f'✓ Mejor val loss: {best_val_loss:.2f}')

# ── Curva de loss ──────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train')
plt.plot(val_losses,   label='Validación')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Training Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f'{CKPT_DIR}/vae_loss_curve.png', dpi=100)
plt.close()
print(f'✓ Curva de loss guardada: {CKPT_DIR}/vae_loss_curve.png')

# ── Verificación visual: reconstrucciones ─────────────────
print('\nGenerando visualización de reconstrucciones...')
model.eval()
sample = tensor_data[:8].to(DEVICE)
with torch.no_grad():
    recon, _, _ = model(sample)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    # Original
    img_orig = sample[i].cpu().numpy().transpose(1, 2, 0)
    axes[0, i].imshow(np.clip(img_orig, 0, 1))
    axes[0, i].set_title('Original', fontsize=7)
    axes[0, i].axis('off')
    # Reconstruida
    img_recon = recon[i].cpu().numpy().transpose(1, 2, 0)
    axes[1, i].imshow(np.clip(img_recon, 0, 1))
    axes[1, i].set_title('Reconstruida', fontsize=7)
    axes[1, i].axis('off')

plt.suptitle('VAE — Originales vs Reconstruidas')
plt.tight_layout()
plt.savefig(f'{CKPT_DIR}/vae_reconstructions.png', dpi=100)
plt.close()
print(f'✓ Reconstrucciones guardadas: {CKPT_DIR}/vae_reconstructions.png')