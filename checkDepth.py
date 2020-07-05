import numpy as np
import matplotlib
import matplotlib.pyplot as plt


depth = np.load('push_initial_depth.npz')['depth']
print(depth)
plt.imshow(depth, vmin=0.0, vmax=1.0)
plt.show()
