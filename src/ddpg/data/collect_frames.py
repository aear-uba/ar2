import numpy as np
import os
import time
from tqdm import tqdm
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

# ── Configuración ──────────────────────────────────────────
TARGET_FRAMES   = 15_000
SAVE_PATH       = 'data/frames.npy'
CHECKPOINT_PATH = 'data/frames_partial.npy'
SAVE_EVERY      = 1_000
# ───────────────────────────────────────────────────────────

config = dict(
    num_scenarios=1,
    start_seed=42,
    map='SSS',
    traffic_density=0.0,
    image_observation=True,
    sensors=dict(rgb_camera=(RGBCamera, 64, 64)),
    vehicle_config=dict(image_source='rgb_camera'),
    physics_world_step_size=1e-1,
    decision_repeat=5,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=10.0,
    crash_object_penalty=5.0,
    out_of_route_done=False,
    on_continuous_line_done=False,
    crash_vehicle_done=False,
    crash_object_done=False,
    use_lateral_reward=True,
    use_render=False,
    show_logo=False,
    show_fps=False,
)

def extract_frame(obs):
    """obs['image'] (64,64,3,3) → float32 (3,64,64) normalizado [0,1]"""
    img = obs['image'][:, :, :, 0]          # (64, 64, 3)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img.transpose(2, 0, 1)           # (3, 64, 64)

def biased_action(action_space):
    """Sesgo hacia adelante — el auto avanza en lugar de girar en círculos"""
    steering = np.random.uniform(-0.3, 0.3)
    throttle = np.random.uniform( 0.3, 0.8)
    return np.array([steering, throttle], dtype=np.float32)

os.makedirs('data', exist_ok=True)

# Reanudar desde checkpoint si existe
if os.path.exists(CHECKPOINT_PATH):
    frames = list(np.load(CHECKPOINT_PATH))
    print(f'Reanudando desde checkpoint: {len(frames)} frames existentes')
else:
    frames = []
    print('Iniciando recolección desde cero')

print(f'Objetivo: {TARGET_FRAMES} frames — faltan {TARGET_FRAMES - len(frames)}')

env = MetaDriveEnv(config=config)
obs, info = env.reset()
episode_steps = 0
episodes = 0

start = time.time()
pbar = tqdm(total=TARGET_FRAMES - len(frames), unit='frame')

try:
    while len(frames) < TARGET_FRAMES:
        action = biased_action(env.action_space)
        obs, reward, terminated, truncated, info = env.step(action)

        frames.append(extract_frame(obs))
        episode_steps += 1
        pbar.update(1)

        if terminated or truncated or episode_steps > 500:
            obs, info = env.reset()
            episode_steps = 0
            episodes += 1

        if len(frames) % SAVE_EVERY == 0:
            np.save(CHECKPOINT_PATH, np.array(frames, dtype=np.float32))
            pbar.set_postfix({'episodios': episodes})

finally:
    pbar.close()
    env.close()

# Guardar dataset final
dataset = np.array(frames[:TARGET_FRAMES], dtype=np.float32)
np.save(SAVE_PATH, dataset)
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

elapsed = time.time() - start
print(f'\n✓ Dataset guardado: {SAVE_PATH}')
print(f'  Shape  : {dataset.shape}')
print(f'  Min/Max: {dataset.min():.4f} / {dataset.max():.4f}')
print(f'  Tiempo : {elapsed/60:.1f} minutos')
print(f'  Episodios: {episodes}')