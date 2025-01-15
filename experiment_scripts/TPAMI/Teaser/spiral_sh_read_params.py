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

def drawSphere(sh, ld=None):
  xyz = genSurfaceNormals(256)
  out = applySHlightXYZ(xyz, sh)
  out[:, xyz[2] == 0] = 0
  return out

def drawMap(sh):
  n = 256

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

  out = applySHlightXYZ(pt.stack([x, y, z], 0), sh)
  out[:, :, 128] = pt.tensor((1, 0, 0))[:, None]
  out[:, :, 256+128] = pt.tensor((1, 0, 0))[:, None]
  return out


def drawSH(sh_np, output, ld=None):
  sh = pt.tensor(sh_np).view(9, 3) 
  ball = drawSphere(sh)
  if ld is not None:
    # Normalized direction (lx, ly, lz)
    lx, ly, lz = ld / np.linalg.norm(ld)
    ly = -ly
    lx = int((lx + 1) * 128)
    ly = int((ly + 1) * 128)
    
    offset = 10
    ball[0, ly-offset:ly+offset, lx-offset:lx+offset] = 0.0
    # then color that pixel
    # ball[0, v, u] = 1.0
    # ball[1, v, u] = 0.0
    # ball[2, v, u] = 0.0
    
  map = drawMap(sh)
  combined = pt.cat([ball, map], 2)
  # print(pt.max(combined))
  # print(pt.min(combined))
  # print("save to " + output)
  save_image(combined, output)

def toCoeff(c):
  t = pysh.SHCoeffs.from_zeros(2)
  t.set_coeffs(c[0], 0, 0)
  t.set_coeffs(c[1], 1, 1)
  t.set_coeffs(c[2], 1, -1)
  t.set_coeffs(c[3], 1, 0)
  t.set_coeffs(c[4], 2, -2)
  t.set_coeffs(c[5], 2, 1)
  t.set_coeffs(c[6], 2, -1)
  t.set_coeffs(c[7], 2, 2)
  t.set_coeffs(c[8], 2, 0)
  return t

def toRGBCoeff(c):
  return [toCoeff(c[::3]), toCoeff(c[1::3]), toCoeff(c[2::3])]

def toDeca(c):
  a = c.coeffs
  lst = [a[0, 0, 0],
         a[0, 1, 1],
         a[1, 1, 1],
         a[0, 1, 0],
         a[1, 2, 2],
         a[0, 2, 1],
         a[1, 2, 1],
         a[0, 2, 2],
         a[0, 2, 0]]
  return np.array(lst)

def toRGBDeca(cc):
  return list(itertools.chain(*zip(toDeca(cc[0]), toDeca(cc[1]), toDeca(cc[2]))))

def axisAngleToEuler(x, y, z, degree):
  xyz = np.array([x, y, z])
  xyz = xyz / np.linalg.norm(xyz)

  rot = R.from_mrp(xyz * np.tan(degree * np.pi / 180 / 4))
  return rot.as_euler('zyz', degrees=True)

def readImage(fn):
  img = Image.open(fn)
  return transforms.ToTensor()(img)


def rotateSH(sh_np, x, y, z, degree):
  cc = toRGBCoeff(sh_np)
  euler = axisAngleToEuler(x, y, z, degree)
  cc[0] = cc[0].rotate(*euler)
  cc[1] = cc[1].rotate(*euler)
  cc[2] = cc[2].rotate(*euler)
  return toRGBDeca(cc)

def stackResult():
  count = 0
  while True:
    if not os.path.exists(f"video_out/0_{count:02d}.png"): break
    im = [readImage(f"video_out/{x}_{count:02d}.png") for x in range(3)]
    
    save_image(pt.cat(im, 1), f"video_out/c_{count:02d}.png")
    count += 1

  os.system(f"ffmpeg -y -i video_out/c_%02d.png -c:v libx264 -pix_fmt yuv420p -crf 18 video_combined.mp4")


# stackResult()
# exit()

def sh_to_ld(sh):
    #NOTE: Roughly Convert the SH to light direction
    sh = sh.reshape(-1, 9, 3)
    ld = np.mean(sh[0:1, 1:4, :], axis=2)
    return ld

def spiralLight(sh_np, cx, cy):
  xyz  = genSurfaceNormals(256)
  save_image(xyz, 'normals.png')
  save_image(xyz[0:1, ...], 'normals_x.png')
  save_image(xyz[1:2, ...], 'normals_y.png')
  save_image(xyz[2:3, ...], 'normals_z.png')
  v = xyz[:, cy, cx]
  print("V : ", v)
  drawSH(sh_np, f"original.png")
  centered = sh_np
  # centered = rotateSH(sh_np,    0, 0, 1, np.arcsin(float(v[0])) * 180 / np.pi)
  # centered = rotateSH(centered, 1, 0, 0, np.arcsin(float(v[1])) * 180 / np.pi)
  drawSH(centered, f"centered.png")
  
  r_end = 1.0
  r_start = 0
  rounds = 6
  n = 120
  # n = 10


  init_direction = sh_to_ld(np.array(centered)[None, ...]).reshape(-1)
  init_direction = init_direction / np.linalg.norm(init_direction)
  at = np.arctan2(init_direction[1], init_direction[0])
  at2 = np.arcsin(init_direction[2])

  centered = rotateSH(centered, 0, 0, 1, at * 180 / np.pi)
  centered = rotateSH(centered, 0, 1, 0, -at2 * 180 / np.pi)
  for i in tqdm.tqdm(range(n)):
    # Original
    # t = i / n 
    # tt = 2 * np.pi * t * rounds
    # rad = t * 0.9
    
    # x = np.cos(tt) * rad
    # y = np.sin(tt) * rad
    # moved = rotateSH(centered, 0, 0, 1, -np.arcsin(y) * 180 / np.pi)
    # moved = rotateSH(moved   , 1, 0, 0, -np.arcsin(x) * 180 / np.pi)
    
    # Edit
    t = i / n 
    tt = 2 * np.pi * t * rounds
    rad = t * 0.9
    
    moved = rotateSH(centered.copy(), 0, 1, 0, 90 * rad)
    moved = rotateSH(moved, 0, 0, 1, tt * 180 / np.pi)
    ld = sh_to_ld(np.array(moved)[None, ...]).reshape(-1)
    # print(moved)
    # print(ld)
    # assert False


    drawSH(moved, f"./video_out/m_{i:03d}.png", ld=ld)

  os.system(f"ffmpeg -y -i video_out/m_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 video_spiral4.mp4")
  exit()

def spiralLight_readPath(sh_np, cx, cy):
  xyz  = genSurfaceNormals(256)
  save_image(xyz, 'normals.png')
  save_image(xyz[0:1, ...], 'normals_x.png')
  save_image(xyz[1:2, ...], 'normals_y.png')
  save_image(xyz[2:3, ...], 'normals_z.png')
  v = xyz[:, cy, cx]
  print("V : ", v)
  drawSH(sh_np, f"original.png")
  
  centered = sh_np
  # init_direction = sh_to_ld(np.array(centered)[None, ...]).reshape(-1)
  # init_direction = init_direction / np.linalg.norm(init_direction)
  # at = np.arctan2(init_direction[1], init_direction[0])
  # at2 = np.arcsin(init_direction[2])

  # centered = rotateSH(centered, 0, 0, 1, at * 180 / np.pi)
  # centered = rotateSH(centered, 0, 1, 0, -at2 * 180 / np.pi)
  drawSH(centered, f"centered.png")
  
  light_path = np.load("./light_params.npy", allow_pickle=True)
  n = len(light_path)
  print(n)
  for i in tqdm.tqdm(range(n)):
    t = light_path[i]["t"]
    rad = light_path[i]["radius"]
    tt = light_path[i]["angle"]
    
    moved = rotateSH(centered.copy(), 0, 1, 0, 90 * rad)
    moved = rotateSH(moved, 0, 0, 1, tt * 180 / np.pi)
    ld = sh_to_ld(np.array(moved)[None, ...]).reshape(-1)
    drawSH(moved, f"./video_out/m_{i:03d}.png", ld=ld)
    
  os.system(f"ffmpeg -y -i video_out/m_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 video_spiral_read.mp4")
  exit()

  for i in tqdm.tqdm(range(n)):
    # Original
    # t = i / n 
    # tt = 2 * np.pi * t * rounds
    # rad = t * 0.9
    
    # x = np.cos(tt) * rad
    # y = np.sin(tt) * rad
    # moved = rotateSH(centered, 0, 0, 1, -np.arcsin(y) * 180 / np.pi)
    # moved = rotateSH(moved   , 1, 0, 0, -np.arcsin(x) * 180 / np.pi)
    
    # Edit
    t = i / n 
    tt = 2 * np.pi * t * rounds
    rad = t * 0.9
    
    moved = rotateSH(centered.copy(), 0, 1, 0, 90 * rad)
    moved = rotateSH(moved, 0, 0, 1, tt * 180 / np.pi)
    ld = sh_to_ld(np.array(moved)[None, ...]).reshape(-1)
    # print(moved)
    # print(ld)
    # assert False


    drawSH(moved, f"./video_out/m_{i:03d}.png", ld=ld)

  os.system(f"ffmpeg -y -i video_out/m_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 video_spiral4.mp4")
  exit()

# 65797.jpg
sh_text = "3.7764273 3.7647202 3.7740586 -0.45223573 -0.48492554 -0.48608136 0.3177414 0.34008643 0.33421847 -0.44365892 -0.47285086 -0.45525044 -0.27055222 -0.26994315 -0.2692122 -0.033267528 -0.047869906 -0.050032064 0.16282524 0.17702723 0.172417 0.14684218 0.14223212 0.14653295 0.2784819 0.27089873 0.27471355"

sh_np = np.array([float(x) for x in sh_text.split(" ")])

os.makedirs("video_out/", exist_ok=True)
spiralLight_readPath(sh_np, 0, 128)
# spiralLight(sh_np, 161, 212)
# spiralLight(sh_np, 5, 120)
# spiralLight(sh_np, 0, 128)
# spiralLight(sh_np, 128, 0)
# spiralLight(sh_np, 128, 128)


# cc = toRGBCoeff(sh_np)
# for axis in range(3):
  # count = 0
  # for i in range(0, 360, 10):
    # mod_sh_np = rotateSH(sh_np, axis==0, axis==1, axis==2, i)
    # drawSH(mod_sh_np, f"video_out/{axis}_{count:02d}.png")
    # count += 1
  # os.system(f"ffmpeg -y -i video_out/{axis}_%02d.png -c:v libx264 -pix_fmt yuv420p -crf 18 video_axis{axis}.mp4")
#

