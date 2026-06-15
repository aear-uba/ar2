import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

# ── Configuración ──────────────────────────────────────────
LATENT_DIM      = 64
ACTION_DIM      = 2        # [steering, throttle]
HIDDEN_DIM      = 256
BUFFER_SIZE     = 10_000
BATCH_SIZE      = 64
GAMMA           = 0.99     # Descuento de reward futura
TAU             = 0.005    # Soft update del target network
LR_ACTOR        = 1e-4
LR_CRITIC       = 1e-3
NOISE_STD       = 0.1      # Exploración — ruido gaussiano sobre acciones
NOISE_DECAY     = 0.995    # El ruido decrece con el tiempo
NOISE_MIN       = 0.01
MAX_EPISODES    = 200
MAX_STEPS       = 500      # Pasos máximos por episodio
WARMUP_STEPS    = 1_000    # Pasos con acción aleatoria antes de entrenar
CKPT_DIR        = 'data/data/checkpoints'
CKPT_EVERY      = 25       # Guardar checkpoint cada N episodios
VAE_PATH        = 'data/data/checkpoints/vae_best.pt'
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ───────────────────────────────────────────────────────────

os.makedirs(CKPT_DIR, exist_ok=True)
print(f'Device: {DEVICE}')

# ── VAE Encoder (congelado) ────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu, _ = self.fc_mu(h), self.fc_logvar(h)
        return mu  # Representación determinística


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)

    def encode(self, x):
        return self.encoder(x)


def load_vae_encoder(path, latent_dim, device):
    """Carga solo el encoder del VAE y lo congela"""
    # Reconstruir VAE completo para cargar los pesos
    class FullDecoder(nn.Module):
        def __init__(self, latent_dim):
            super().__init__()
            self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        def forward(self, z):
            return self.deconv(self.fc(z).view(-1, 256, 4, 4))

    class FullVAE(nn.Module):
        def __init__(self, latent_dim):
            super().__init__()
            self.encoder = Encoder(latent_dim)
            self.decoder = FullDecoder(latent_dim)
        def encode(self, x):
            return self.encoder(x)

    vae = FullVAE(latent_dim)
    vae.load_state_dict(torch.load(path, map_location=device))
    encoder = vae.encoder.to(device)

    # Congelar — el encoder NO se actualiza durante DDPG
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    print(f'✓ Encoder VAE cargado y congelado desde: {path}')
    return encoder


# ── DDPG: Actor ───────────────────────────────────────────
class Actor(nn.Module):
    """
    Input : z (LATENT_DIM,)
    Output: acción (ACTION_DIM,) en [-1, 1]
    """
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Salida en [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


# ── DDPG: Critic ──────────────────────────────────────────
class Critic(nn.Module):
    """
    Input : z (LATENT_DIM,) + acción (ACTION_DIM,)
    Output: Q-value escalar
    """
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, action):
        return self.net(torch.cat([z, action], dim=1))


# ── Replay Buffer ─────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, z, action, reward, z_next, done):
        self.buffer.append((z, action, reward, z_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        z, a, r, z_next, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(z)),
            torch.FloatTensor(np.array(a)),
            torch.FloatTensor(np.array(r)).unsqueeze(1),
            torch.FloatTensor(np.array(z_next)),
            torch.FloatTensor(np.array(done)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ── Soft update de target networks ────────────────────────
def soft_update(source, target, tau):
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


# ── Extracción de frame ───────────────────────────────────
def extract_frame(obs):
    img = obs['image'][:, :, :, 0].astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, 64, 64)


# ── Setup entorno ─────────────────────────────────────────
metadrive_config = dict(
    num_scenarios=1, start_seed=42, map='SSS',
    traffic_density=0.0, image_observation=True,
    sensors=dict(rgb_camera=(RGBCamera, 64, 64)),
    vehicle_config=dict(image_source='rgb_camera'),
    physics_world_step_size=1e-1, decision_repeat=5,
    out_of_road_penalty=5.0, crash_vehicle_penalty=10.0,
    crash_object_penalty=5.0, out_of_route_done=False,
    on_continuous_line_done=False, crash_vehicle_done=False,
    crash_object_done=False, use_lateral_reward=True,
    use_render=False, show_logo=False, show_fps=False,
)

env = MetaDriveEnv(config=metadrive_config)
print('✓ Entorno MetaDrive listo')

# ── Inicializar modelos ───────────────────────────────────
encoder = load_vae_encoder(VAE_PATH, LATENT_DIM, DEVICE)

actor        = Actor(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
actor_target = Actor(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
actor_target.load_state_dict(actor.state_dict())

critic        = Critic(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
critic_target = Critic(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
critic_target.load_state_dict(critic.state_dict())

actor_opt  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)

buffer = ReplayBuffer(BUFFER_SIZE)
print('✓ Modelos DDPG inicializados')

# ── Entrenamiento DDPG ────────────────────────────────────
episode_rewards = []
episode_steps_log = []
noise_std = NOISE_STD
total_steps = 0

print(f'\nEntrenando DDPG — {MAX_EPISODES} episodios')
print(f'Warmup: {WARMUP_STEPS} pasos con acción aleatoria')
print('─' * 65)

for episode in range(1, MAX_EPISODES + 1):
    obs, info = env.reset()
    frame = extract_frame(obs).to(DEVICE)

    with torch.no_grad():
        z = encoder(frame)  # (1, LATENT_DIM)

    episode_reward = 0
    start = time.time()

    for step in range(MAX_STEPS):
        total_steps += 1

        # Acción: aleatoria en warmup, luego actor + ruido
        if total_steps < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor(z).cpu().numpy().flatten()
            # Ruido gaussiano para exploración
            noise = np.random.normal(0, noise_std, size=ACTION_DIM)
            action = np.clip(action + noise, -1.0, 1.0)

        obs_next, reward, terminated, truncated, info = env.step(action)

        frame_next = extract_frame(obs_next).to(DEVICE)
        with torch.no_grad():
            z_next = encoder(frame_next)

        done = terminated or truncated
        buffer.push(
            z.cpu().numpy().flatten(),
            action,
            reward,
            z_next.cpu().numpy().flatten(),
            float(done)
        )

        z = z_next
        episode_reward += reward

        # Actualizar redes si hay suficientes muestras
        if len(buffer) >= BATCH_SIZE and total_steps >= WARMUP_STEPS:
            z_b, a_b, r_b, z_next_b, done_b = buffer.sample(BATCH_SIZE)
            z_b      = z_b.to(DEVICE)
            a_b      = a_b.to(DEVICE)
            r_b      = r_b.to(DEVICE)
            z_next_b = z_next_b.to(DEVICE)
            done_b   = done_b.to(DEVICE)

            # Critic update
            with torch.no_grad():
                a_next   = actor_target(z_next_b)
                q_target = r_b + GAMMA * (1 - done_b) * critic_target(z_next_b, a_next)
            q_current = critic(z_b, a_b)
            critic_loss = nn.functional.mse_loss(q_current, q_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # Actor update
            actor_loss = -critic(z_b, actor(z_b)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # Soft update target networks
            soft_update(actor,  actor_target,  TAU)
            soft_update(critic, critic_target, TAU)

        if done:
            break

    # Decaimiento del ruido
    noise_std = max(NOISE_MIN, noise_std * NOISE_DECAY)
    episode_rewards.append(episode_reward)
    episode_steps_log.append(step + 1)
    elapsed = time.time() - start

    print(f'Ep {episode:4d}/{MAX_EPISODES} | '
          f'reward={episode_reward:7.2f} | '
          f'steps={step+1:4d} | '
          f'noise={noise_std:.3f} | '
          f'buf={len(buffer):5d} | '
          f'{elapsed:.1f}s')

    # Checkpoint
    if episode % CKPT_EVERY == 0:
        ckpt = {
            'episode': episode,
            'actor':   actor.state_dict(),
            'critic':  critic.state_dict(),
            'rewards': episode_rewards,
        }
        torch.save(ckpt, f'{CKPT_DIR}/ddpg_ep_{episode}.pt')
        print(f'  → Checkpoint guardado: ddpg_ep_{episode}.pt')

# ── Guardar modelo final ───────────────────────────────────
torch.save({
    'actor':   actor.state_dict(),
    'critic':  critic.state_dict(),
    'rewards': episode_rewards,
}, f'{CKPT_DIR}/ddpg_final.pt')
print(f'\n✓ DDPG final guardado')

env.close()

# ── Curva de rewards ───────────────────────────────────────
window = 10
smoothed = np.convolve(episode_rewards,
                       np.ones(window)/window, mode='valid')
plt.figure(figsize=(12, 4))
plt.plot(episode_rewards, alpha=0.3, label='Raw')
plt.plot(range(window-1, len(episode_rewards)), smoothed, label=f'Media {window} ep')
plt.xlabel('Episodio')
plt.ylabel('Reward total')
plt.title('DDPG — Reward por episodio')
plt.legend()
plt.tight_layout()
plt.savefig(f'{CKPT_DIR}/ddpg_rewards.png', dpi=100)
plt.close()
print(f'✓ Curva de rewards guardada: {CKPT_DIR}/ddpg_rewards.png')