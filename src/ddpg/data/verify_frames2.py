import numpy as np
import matplotlib.pyplot as plt

dataset = np.load('data/frames.npy')

# Ver frame 1 (el segundo)
img = dataset[1].transpose(1, 2, 0)  # (3,64,64) → (64,64,3)
img = np.clip(img, 0, 1)

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title('Frame 1')
plt.axis('off')
plt.savefig('data/frame_1.png', dpi=100)
plt.close()
print('✓ Guardado en data/frame_1.png')