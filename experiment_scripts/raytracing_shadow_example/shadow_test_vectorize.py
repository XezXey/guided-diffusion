import numpy as np
import torch as pt
from torchvision.utils import save_image
import time


data = np.load('./depth_grid.npy', allow_pickle=True)
grid = pt.tensor(data.item().get('depth_grid'))
ray = pt.tensor(data.item().get('light_direction')).view(3)
vis = pt.clone(grid[:, :, 2])
save_image(vis, "vis.png")

# manipulate depth / ray
grid[:, :, 2] *= 100
ray[2] *= 0.5
# ray[0] = 1
# ray[1] = -1
# ray[2] = 1
# print(ray)
# exit()


n = 224
ray = ray / pt.norm(ray)
mxaxis = max(abs(ray[0]), abs(ray[1]))
shift = ray / mxaxis * pt.arange(n).view(n, 1)
coords = grid.view(1, n, n, 3) + shift.view(n, 1, 1, 3)

output = pt.nn.functional.grid_sample(
  # pt.tile(grid[:, :, 2].view(1, 1, n, n), [n, 1, 1, 1]),
  pt.tensor(np.tile(grid[:, :, 2].view(1, 1, n, n), [n, 1, 1, 1])),
  coords[..., :2] / (n - 1) * 2 - 1,
  align_corners=True)
diff = coords[..., 2] - output[:, 0] 
vis *= (pt.min(diff, dim=0)[0] > -0.1) * 0.5 + 0.5
save_image(vis, "output_vec3.png")
