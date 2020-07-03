import numpy as np
import matplotlib
import matplotlib.pyplot as plt


depth = np.load('mug_depth.npz')['depth']
print(depth)
plt.imshow(depth, vmin=0.0, vmax=0.15)
plt.show()
