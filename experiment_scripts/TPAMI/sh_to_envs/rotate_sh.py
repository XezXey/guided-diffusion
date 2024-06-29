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
import matplotlib.pyplot as plt

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

def drawSphere(sh):
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


def drawSH(sh_np, output):
  sh = pt.tensor(sh_np).view(9, 3) 
  ball = drawSphere(sh)
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
    if not os.path.exists(f"rotate_out/0_{count:02d}.png"): break
    im = [readImage(f"rotate_out/{x}_{count:02d}.png") for x in range(3)]
    
    save_image(pt.cat(im, 1), f"rotate_out/c_{count:02d}.png")
    count += 1

  os.system(f"ffmpeg -y -i rotate_out/c_%02d.png -c:v libx264 -pix_fmt yuv420p -crf 18 video_combined.mp4")


# stackResult()
# exit()

def drawLD(ld, output):
  # Draw xy, xz, yz from ld (1 x 3)
  ld = ld.flatten()
  ld = ld / np.linalg.norm(ld)
  x = ld[0]
  y = ld[1]
  z = ld[2]
  # 3D plot
  fig = plt.figure(figsize=(10, 5))
  ax1 = fig.add_subplot(121, projection='3d')
  ax1.scatter(x, y, z, c='r', marker='o')
  ax1.set_xlabel('X axis')
  ax1.set_ylabel('Y axis')
  ax1.set_zlabel('Z axis')
  ax1.set_title('3D Plot')

  # XY plane projection
  ax2 = fig.add_subplot(122)
  ax2.scatter(x, y, c='r', marker='o')
  ax2.set_xlabel('X axis')
  ax2.set_ylabel('Y axis')
  ax2.set_title('Projection in XY Plane')

  # Add grid lines
  ax1.grid(True)
  ax2.grid(True)
  # Add dot markers to the axes
  for ax in [ax1, ax2]:
    ax.scatter(0, 1, c='k', marker='.')
    ax.scatter(1, 0, c='k', marker='.')
    
    ax.scatter(0, -1, c='k', marker='.')
    ax.scatter(-1, 0, c='k', marker='.')

  # Adjust layout
  plt.tight_layout()

  # Save the figure
  plt.savefig(output)
  plt.close()
  

def spiralLight(sh_np, cx, cy):
  xyz  = genSurfaceNormals(256)
  save_image(xyz, 'normals.png')
  save_image(xyz[0:1, ...], 'normals_x.png')
  save_image(xyz[1:2, ...], 'normals_y.png')
  save_image(xyz[2:3, ...], 'normals_z.png')
  v = xyz[:, cy, cx]
  print("V : ", v)
  drawSH(sh_np, f"original.png")
  centered = rotateSH(sh_np,    0, 1, 0, np.arcsin(float(v[0])) * 180 / np.pi)
  centered = rotateSH(centered, 1, 0, 0, np.arcsin(float(v[1])) * 180 / np.pi)
  drawSH(centered, f"centered.png")
  
  rounds = 5
  n = 60
  for i in tqdm.tqdm(range(n)):
    # print(i)
    t = i / n # Fraction of rotation
    tt = t * rounds * 2 * np.pi
    rad = t * 0.9

    x = np.sin(tt) * rad
    y = np.cos(tt) * rad
    # moved = rotateSH(centered, 0, 0, 1, -np.arcsin(x) * 180 / np.pi)
    # moved = rotateSH(centered, 0, 1, 0, -np.arcsin(x) * 180 / np.pi)  # Rotate over y axis by -np.arcsin(x) * 180 / np.pi
    # moved = rotateSH(centered, 1, 0, 0, -np.arcsin(x) * 180 / np.pi)  # Rotate over y axis by -np.arcsin(x) * 180 / np.pi
    
    moved = rotateSH(centered, 0, 1, 0, -np.arcsin(x) * 180 / np.pi)  # Rotate over y axis by -np.arcsin(x) * 180 / np.pi
    moved = rotateSH(moved   , 1, 0, 0, -np.arcsin(y) * 180 / np.pi)  # Rotate over x axis by -np.arcsin(y) * 180 / np.pi
    sh_moved = np.array(moved).reshape(-1, 9, 3)
    ld = np.mean(sh_moved[0:1, 1:4, :], axis=2)
    
    
    drawLD(ld, f"rotate_out/ld_{i:03d}.png")
    drawSH(moved, f"./rotate_out/m_{i:03d}.png")

  os.system(f"ffmpeg -y -i rotate_out/m_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 ./rotate_out/spiral_out.mp4")
  os.system(f"ffmpeg -y -i rotate_out/ld_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 ./rotate_out/LD_spiral_out.mp4")
  exit()


# sh_text = "3.7467763 3.7607439 3.7748303 -0.17061733 -0.18169762 -0.18631022 0.07023415 0.07821477 0.07901665 -0.41905978 -0.40052396 -0.3687197 0.1470597 0.14538132 0.14575471 -0.28285792 -0.294153 -0.29541978 0.6458615 0.65194905 0.65252924 0.63551205 0.64745045 0.6551203 0.049085654 0.04447888 0.051973432"
sh_text = "3.4063814 3.4182117 3.4240203 0.2447223 0.2731144 0.27873707 0.36326286 0.37438118 0.37373984 -0.53145957 -0.511477 -0.49685538 -0.02802198 -0.02605348 -0.025497597 0.15530688 0.17207308 0.17424926 0.5655835 0.57417613 0.5728966 0.25129333 0.25584137 0.25848323 0.7325827 0.7334002 0.73769367"
# sh_text = "4.1208167 4.113482 4.119587 -0.35264063 -0.38354635 -0.3695604 0.22570051 0.23566791 0.2297648 -0.46458092 -0.48861718 -0.47885132 -0.31964195 -0.3250893 -0.32510865 -0.03330686 -0.05441852 -0.047533773 0.5244178 0.5305497 0.52580774 -0.10181245 -0.107420176 -0.10384175 0.097432286 0.091219686 0.09322698"
# sh_text = "3.4756186 3.4662082 3.4709003 -0.32444367 -0.3257522 -0.3201226 0.22026052 0.23849344 0.23579775 -0.53786767 -0.5714127 -0.5618794 -0.054980133 -0.056462407 -0.056166932 -0.14991328 -0.15211679 -0.14935327 0.32371297 0.33627337 0.33428767 0.22112036 0.22257182 0.22352639 0.7944299 0.7839997 0.7864212"

sh_np = np.array([float(x) for x in sh_text.split(" ")])

out_dir = 'rotate_out/'
os.makedirs("rotate_out/", exist_ok=True)
# spiralLight(sh_np, 161, 212)
# spiralLight(sh_np, 5, 120)
spiralLight(sh_np, 128, 128)
# spiralLight(sh_np, 128, 0)
# spiralLight(sh_np, 128, 128)


# cc = toRGBCoeff(sh_np)
# for axis in range(3):
  # count = 0
  # for i in range(0, 360, 10):
    # mod_sh_np = rotateSH(sh_np, axis==0, axis==1, axis==2, i)
    # drawSH(mod_sh_np, f"rotate_out/{axis}_{count:02d}.png")
    # count += 1
  # os.system(f"ffmpeg -y -i rotate_out/{axis}_%02d.png -c:v libx264 -pix_fmt yuv420p -crf 18 video_axis{axis}.mp4")
#
