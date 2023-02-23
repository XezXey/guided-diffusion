import numpy as np
import cv2
import torch as pt
from torchvision.utils import save_image, make_grid

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
print(ray/mxaxis)
# Walk by 1 pixel; 
# ray/mxaxis for ensuring that we move follow desired direction by 1 pixel
shift = ray / mxaxis * pt.arange(n).view(n, 1)  
print(shift.shape)
for y in range(n):
  for x in range(n):
    if grid[y, x, 2] == 0: continue
    # grid[y, x] = 3d pts (x, y, depth)
    # get depth of ray 
    coords = grid[y, x] + shift
    # coords = grid[y, x]
    # print(coords.shape)
    # coords[:, :2] = coords[:, :2] + shift[:, :2]
    print(coords.shape)
    exit()
    # exit()
    # print(pt.max(coords[:, :2].view(1, n, 1, 2)))
    # print(pt.min(coords[:, :2].view(1, n, 1, 2)))
    # print(pt.max(coords[:, :2].view(1, n, 1, 2) / (n - 1) * 2 - 1))
    # print(pt.min(coords[:, :2].view(1, n, 1, 2) / (n - 1) * 2 - 1))
    # Get depth at coords[x, y]
    output = pt.nn.functional.grid_sample(
      grid[:, :, 2].view(1, 1, n, n),   # Depth
      coords[:, :2].view(1, n, 1, 2) / (n - 1) * 2 - 1, # Coordinates to eval which norm to [-1, 1]
      align_corners=True  # Use for 
    )
    # if exists any points along the (ray - depth) < -0.1 => under shadow
    if pt.min(coords[:, 2] - output[0, 0, :, 0]) < -0.1:
      # Edit only pixel under shadow to have value = 0.5 
      vis[y, x] = 0.5
  
print(pt.unique(vis))

save_image(vis, "output.png")
exit()

