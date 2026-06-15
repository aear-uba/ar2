import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dataset = np.load('data/frames.npy')
print(f'Shape  : {dataset.shape}')
print(f'Min/Max: {dataset.min():.4f} / {dataset.max():.4f}')
print(f'Mean   : {dataset.mean():.4f}')

indices = np.linspace(0, len(dataset)-1, 16, dtype=int)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle(f'Muestra de frames ({len(dataset)} totales)')

for i, ax in enumerate(axes.flat):
    img = dataset[indices[i]].transpose(1, 2, 0)  # (3,64,64) → (64,64,3)
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(f'Frame {indices[i]}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('data/sample_grid.png', dpi=100)
print('✓ Grilla guardada en data/sample_grid.png')