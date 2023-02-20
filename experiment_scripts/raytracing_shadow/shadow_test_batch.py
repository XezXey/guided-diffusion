import numpy as np
import cv2
import torch as pt
from torchvision.utils import save_image, make_grid

data = np.load('./depth_grid.npy', allow_pickle=True)
grid = pt.tensor(data.item().get('depth_grid'))
ray = pt.tensor(data.item().get('light_direction')).view(3)
vis = pt.clone(grid[:, :, 2])
orig = pt.clone(grid[:, :, 2])
save_image(vis, "vis.png")

# manipulate depth / ray
grid[:, :, 2] *= 100

n = 224
gs = grid.permute((2, 0, 1)).view(1, 3, n, n)

for b in range(100):
  vis = pt.clone(orig)
  print(b)
  ray[0] = np.sin(b / 100 * np.pi * 2)
  ray[1] = np.cos(b / 100 * np.pi * 2)
  ray[2] = 0.5

  ray = ray / pt.norm(ray)
  mxaxis = max(abs(ray[0]), abs(ray[1]))
  shift = ray / mxaxis * pt.arange(n).view(n, 1)
  for y in range(n):
    for x in range(n):
      if grid[y, x, 2] == 0: continue
      coords = grid[y, x] + shift
      output = pt.nn.functional.grid_sample(
        gs,
        coords[:, :2].view(1, n, 1, 2) / (n - 1) * 2 - 1,
        align_corners=True
      )
      if pt.min(coords[:, 2] - output[0, 2, :, 0]) < -0.1:
        vis[y, x] = 0.5

  save_image(vis, "output_%03d.png" % b)

