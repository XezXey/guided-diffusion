# Rule: tmux send -t 1 "python rotate2.py" Enter
import numpy as np
import tqdm
import torch as pt
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import itertools
import os
from scipy.spatial.transform import Rotation as R

import pyshtools as pysh

def applySHlight(normal_images, sh_coeff):
  N = normal_images
  sh = pt.stack(
    [
      N[0] * 0.0 + 1.0,
      N[0],
      N[1],
      N[2],
      N[0] * N[1],
      N[0] * N[2],
      N[1] * N[2],
      N[0] ** 2 - N[1] ** 2,
      3 * (N[2] ** 2) - 1,
    ],
    0,
  )  # [9, h, w]
  pi = np.pi
  constant_factor = pt.tensor(
    [
      1 / np.sqrt(4 * pi),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
    ]
  ).float()
  sh = sh * constant_factor[:, None, None]

  shading = pt.sum(
    sh_coeff[:, :, None, None] * sh[:, None, :, :], 0
  )  # [9, 3, h, w]

  return shading

def applySHlightXYZ(xyz, sh):
  out = applySHlight(xyz, sh)
  # out /= pt.max(out)
  out *= 0.7
  return pt.clip(out, 0, 1)

def genSurfaceNormals(n):
  x = pt.linspace(-1, 1, n)
  y = pt.linspace(1, -1, n)
  y, x = pt.meshgrid(y, x)

  z = (1 - x ** 2 - y ** 2)
  z[z < 0] = 0
  z = pt.sqrt(z)
  return pt.stack([x, y, z], 0)

def drawSphere(sh, img_size=256):
  n = img_size
  xyz = genSurfaceNormals(n)
  out = applySHlightXYZ(xyz, sh)
  out[:, xyz[2] == 0] = 0
  return out

def drawMap(sh, img_size=256):

  n = img_size
  lr = pt.linspace(0, 2 * np.pi, 2 * n)
  ud = pt.linspace(0, np.pi, n)
  ud, lr = pt.meshgrid(ud, lr)

  # we want to make the sphere unwrap at the center of this map,
  # so the left-most column is the furthest-away point on the sphere
  # lr going counter-clockwise = increasing in value.
  # ud starting from 0 (top) to pi (bottom).
  x = -pt.sin(ud) * pt.sin(lr)
  y = pt.cos(ud)
  z = -pt.sin(ud) * pt.cos(lr)

  lm = n // 2
  rm = n + (n // 2)

  out = applySHlightXYZ(pt.stack([x, y, z], 0), sh)
  out_centered = out[:, :, lm:rm].clone()
  out_clean = out.clone()
  out[:, :, lm] = pt.tensor((1, 0, 0))[:, None]
  out[:, :, rm] = pt.tensor((1, 0, 0))[:, None]
  return out, out_centered, out_clean


def drawSH(sh_np, img_size=256):
  sh = pt.tensor(sh_np).view(9, 3) 
  ball = drawSphere(sh, img_size)
  map, map_centered, map_clean = drawMap(sh, img_size)
  combined = pt.cat([ball, map], 2)
  # save_image(combined, output)
  # save_image(map_centered, output.replace(".png", "_centered.png"))
  return ball, map, map_centered, map_clean, combined

def readImage(fn):
  img = Image.open(fn)
  return transforms.ToTensor()(img)

def interpolateSH(inp_sh, tgt_sh, n_step, img_size=256):
  print(inp_sh.shape, tgt_sh.shape)
  intp = np.linspace(0, 1, n_step)
  out = inp_sh * (1 - intp[:, None]) + tgt_sh * intp[:, None]
  print(out.shape)
  for i in tqdm.tqdm(range(n_step)):
    sh = out[i]
    ball, map, map_centered, map_clean, combined = drawSH(sh, img_size)
    for j in zip([ball, map, map_centered, map_clean, combined], ['ball', 'map', 'map_centered', 'map_clean', 'combined']):
      save_image(j[0], f"video_out/{j[1]}/m_{i:03d}.png")
    
    
  for j in zip([ball, map, map_centered, map_clean, combined], ['ball', 'map', 'map_centered', 'map_clean', 'combined']):
    os.system(f"ffmpeg -y -i video_out/{j[1]}/m_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 video_out/{j[1]}.mp4")
  

# sh_text = "3.7467763 3.7607439 3.7748303 -0.17061733 -0.18169762 -0.18631022 0.07023415 0.07821477 0.07901665 -0.41905978 -0.40052396 -0.3687197 0.1470597 0.14538132 0.14575471 -0.28285792 -0.294153 -0.29541978 0.6458615 0.65194905 0.65252924 0.63551205 0.64745045 0.6551203 0.049085654 0.04447888 0.051973432"
# sh_text = "4.1208167 4.113482 4.119587 -0.35264063 -0.38354635 -0.3695604 0.22570051 0.23566791 0.2297648 -0.46458092 -0.48861718 -0.47885132 -0.31964195 -0.3250893 -0.32510865 -0.03330686 -0.05441852 -0.047533773 0.5244178 0.5305497 0.52580774 -0.10181245 -0.107420176 -0.10384175 0.097432286 0.091219686 0.09322698"

inp_sh_text = "3.4063814 3.4182117 3.4240203 0.2447223 0.2731144 0.27873707 0.36326286 0.37438118 0.37373984 -0.53145957 -0.511477 -0.49685538 -0.02802198 -0.02605348 -0.025497597 0.15530688 0.17207308 0.17424926 0.5655835 0.57417613 0.5728966 0.25129333 0.25584137 0.25848323 0.7325827 0.7334002 0.73769367"
inp_sh_np = np.array([float(x) for x in inp_sh_text.split(" ")])
tgt_sh_text = "3.4756186 3.4662082 3.4709003 -0.32444367 -0.3257522 -0.3201226 0.22026052 0.23849344 0.23579775 -0.53786767 -0.5714127 -0.5618794 -0.054980133 -0.056462407 -0.056166932 -0.14991328 -0.15211679 -0.14935327 0.32371297 0.33627337 0.33428767 0.22112036 0.22257182 0.22352639 0.7944299 0.7839997 0.7864212"
tgt_sh_np = np.array([float(x) for x in tgt_sh_text.split(" ")])

os.makedirs("video_out/", exist_ok=True)
for j in ['ball', 'map', 'map_centered', 'map_clean', 'combined']:
  os.makedirs(f"video_out/{j}", exist_ok=True)


# drawSH(inp_sh_np)
# drawSH(tgt_sh_np)

interpolateSH(inp_sh_np, tgt_sh_np, 60, 256)
