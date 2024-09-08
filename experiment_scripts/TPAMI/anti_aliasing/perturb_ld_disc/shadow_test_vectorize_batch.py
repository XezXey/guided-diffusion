import numpy as np
import torch as pt
from torchvision.utils import save_image
import time, os
from torchvision.transforms import GaussianBlur

data = np.load('./depth_grid.npy', allow_pickle=True)
grid = pt.tensor(data.item().get('depth_grid'))
ray = pt.tensor(data.item().get('light_direction')).view(3)
orig = pt.clone(grid[:, :, 2])

blurred_orig = GaussianBlur(kernel_size=13, sigma=2.0)(orig.unsqueeze(0).unsqueeze(0))

mask = orig != 0
mask = mask.float().unsqueeze(0).unsqueeze(0)
blurred_mask = GaussianBlur(kernel_size=13, sigma=2.0)(mask)
blurred_mask[blurred_mask == 0] = 1e-10 # prevent division by zero
blurred_orig = blurred_orig / blurred_mask
# Remove batch and channel dimensions
blurred_orig = blurred_orig.squeeze(0).squeeze(0)

blur_mask = (orig == 0).float()
blurred_orig = blurred_orig * (1 - blur_mask)

grid[:, :, 2] = blurred_orig

# apply gaussian blur to orig
# save_image(blurred_orig, "orig_blur.png")
# exit()

grid[:, :, 2] *= 100
# print(grid.shape, pt.std(grid, dim=(0, 1)))
# exit()

for b in range(100):
  # b = 20
  vis = pt.clone(orig)
  print(b)
  ray[0] = np.sin(b / 100 * np.pi * 2)
  ray[1] = np.cos(b / 100 * np.pi * 2)
  ray[2] = 0.5

  ray = ray / pt.norm(ray)

  orth = pt.cross(ray, pt.tensor([0, 0, 1.0], dtype=pt.double))
  orth2 = pt.cross(ray, orth)
  orth = orth / pt.norm(orth)
  orth2 = orth2 / pt.norm(orth2)
#
  max_radius = 0.2

  big_coords = []
  round = 30
  
  if round > 1:
    for ti in range(round):
      tt = ti / (round - 1) * 2 * np.pi * 3

      n = 224
      pray = (orth * np.cos(tt) + orth2 * np.sin(tt)) * (ti / (round - 1)) * max_radius + ray

      mxaxis = max(abs(pray[0]), abs(pray[1]))
      shift = pray / mxaxis * pt.arange(n).view(n, 1)
      # exit()
      coords = grid.view(1, n, n, 3) + shift.view(n, 1, 1, 3)
      big_coords.append(coords)
    big_coords = pt.cat(big_coords, dim=0)
  else:
    n = 224
    mxaxis = max(abs(ray[0]), abs(ray[1]))
    shift = ray / mxaxis * pt.arange(n).view(n, 1)
    big_coords = grid.view(1, n, n, 3) + shift.view(n, 1, 1, 3)

  # print(big_coords.shape)
  # exit()
#
  # print("tt", tt.shape)
  # print("orth", orth.shape)
  # ss = (orth * pt.cos(tt) + orth2 * pt.sin(tt)) * (tt / round) * max_radius + ray
  # ss = ss / pt.norm(ss, dim=1, keepdim=True)
  # print("ss", ss.shape)
  # print(ss)
  # exit()

  # manipulate depth / ray
  # ray[2] *= 0.5
  # ray[0] = 1
  # ray[1] = -1
  # ray[2] = 1
  # print(ray)
  # exit()

  # n = 224
  # ray = ray / pt.norm(ray)
  # mxaxis = max(abs(ray[0]), abs(ray[1]))
  # shift = ray / mxaxis * pt.arange(n).view(n, 1)
  # print(grid.shape, shift.shape)
  # exit()

  output = pt.nn.functional.grid_sample(
    pt.tensor(np.tile(grid[:, :, 2].view(1, 1, n, n), [n * round, 1, 1, 1])),
    big_coords[..., :2] / (n - 1) * 2 - 1,
    align_corners=True)
  diff = big_coords[..., 2] - output[:, 0] 
  # print(diff.shape)
  # print(diff.view(round, -1, n, n).shape)
  # print(pt.min(diff.view(round, -1, n, n), dim=1, keepdim=True)[0].shape)
  kk = pt.mean((pt.min(diff.view(round, -1, n, n), dim=1, keepdim=True)[0] > -0.1) * 1.0, dim=(0, 1)) * 0.5 + 0.5
  vis *= kk 
  # print(kk.shape)
  # exit()
  # vis *= (pt.min(diff, dim=0)[0] > -0.1) * 0.5 + 0.5
  os.makedirs("./r30", exist_ok=True)
  save_image(vis, "./r30/output_%03d.png" % b)
  # exit()
  # save_image(vis, "output_vec.png")
