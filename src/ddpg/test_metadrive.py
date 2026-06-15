import metadrive
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import numpy as np
import time
from PIL import Image

# print(f'MetaDrive version: {metadrive.__version__}')

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
    use_render=True,
    show_fps=True,
)

env = MetaDriveEnv(config=config)
obs, info = env.reset()

print(f'obs["image"] shape : {obs["image"].shape}')
print(f'Min/Max            : {obs["image"].min():.4f} / {obs["image"].max():.4f}')

# Test velocidad — 100 pasos
start = time.time()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
elapsed = time.time() - start
print(f'100 pasos en {elapsed:.2f}s → {elapsed * 150:.0f}s estimado para 15000 frames')

# Guardar frame de muestra
img = obs['image'][:, :, :, 0]
print(f'Frame shape : {img.shape}')
print(f'Min/Max     : {img.min():.4f} / {img.max():.4f}')
if img.max() > 1.0:
    img = (img / 255.0)
Image.fromarray((img * 255).astype(np.uint8)).save('test_frame.png')
print('✓ Frame guardado como test_frame.png — abrir para verificar')

env.close()