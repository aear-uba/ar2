import torch
import torch.nn as nn
import numpy as np
import imageio
import os
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

# ── Configuración ──────────────────────────────────────────
LATENT_DIM   = 64
ACTION_DIM   = 2
HIDDEN_DIM   = 256
VIDEO_SECONDS = 40       # Cambiar para video mas largo
FPS          = 20
CKPT_DIR     = 'data/data/checkpoints'
VIDEO_PATH   = 'data/data/driving_policy.mp4'
DEVICE       = torch.device('cpu')
# ───────────────────────────────────────────────────────────

# ── Modelos ────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)
    def forward(self, x):
        return self.fc_mu(self.conv(x).flatten(1))

class FullDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )
    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 256, 4, 4))

class FullVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = FullDecoder(latent_dim)
    def encode(self, x): return self.encoder(x)

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
    def forward(self, z): return self.net(z)

# ── Cargar pesos ───────────────────────────────────────────
vae = FullVAE(LATENT_DIM)
vae.load_state_dict(torch.load(f'{CKPT_DIR}/vae_best.pt', map_location=DEVICE))
vae.eval()
encoder = vae.encoder
for p in encoder.parameters():
    p.requires_grad = False

actor = Actor(LATENT_DIM, ACTION_DIM, HIDDEN_DIM)
ckpt = torch.load(f'{CKPT_DIR}/ddpg_final.pt', map_location=DEVICE)
actor.load_state_dict(ckpt['actor'])
actor.eval()
print('✓ Modelos cargados')

# ── Entorno con render habilitado ─────────────────────────
config = dict(
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

env = MetaDriveEnv(config=config)
obs, info = env.reset()
print('✓ Entorno listo')

# ── Grabar video ───────────────────────────────────────────
total_frames = VIDEO_SECONDS * FPS
frames = []

print(f'Grabando {VIDEO_SECONDS} segundos ({total_frames} frames)...')

step = 0
total_reward = 0

while len(frames) < total_frames:
    # Extraer frame para el VAE
    img = obs['image'][:, :, :, 0].astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    tensor = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)

    # Obtener acción del actor (sin ruido — política pura)
    with torch.no_grad():
        z      = encoder(tensor)
        action = actor(z).numpy().flatten()

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1

    # Capturar frame para el video — imagen de la cámara del auto
    img_frame = obs['image'][:, :, :, 0]
    if img_frame.max() <= 1.0:
        img_frame = (img_frame * 255).astype(np.uint8)
    else:
        img_frame = img_frame.astype(np.uint8)

    # Escalar a 256x256 para que el video sea visible
    from PIL import Image
    pil_img = Image.fromarray(img_frame).resize((256, 256), Image.NEAREST)
    frames.append(np.array(pil_img))

    if terminated or truncated:
        obs, info = env.reset()
        print(f'  Reset en step {step} (reward acumulada: {total_reward:.2f})')

env.close()

# ── Guardar MP4 ────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
imageio.mimwrite(VIDEO_PATH, frames, fps=FPS, codec='libx264')
print(f'\n✓ Video guardado: {VIDEO_PATH}')
print(f'  Duración : {VIDEO_SECONDS} segundos')
print(f'  Frames   : {len(frames)}')
print(f'  Reward   : {total_reward:.2f}')
print(f'\nPara video de 60 segundos: cambiar VIDEO_SECONDS = 60')