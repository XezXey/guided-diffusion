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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--light_path', required=True)
args = parser.parse_args()

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

def spiralLight_readPath(sh_np, cx, cy):
  xyz  = genSurfaceNormals(256)
  save_image(xyz, 'normals.png')
  save_image(xyz[0:1, ...], 'normals_x.png')
  save_image(xyz[1:2, ...], 'normals_y.png')
  save_image(xyz[2:3, ...], 'normals_z.png')
  v = xyz[:, cy, cx]
  print("V : ", v)
  drawSH(sh_np, f"original.png")
  
  
  sh_text_ref = "3.7764273 3.7647202 3.7740586 -0.45223573 -0.48492554 -0.48608136 0.3177414 0.34008643 0.33421847 -0.44365892 -0.47285086 -0.45525044 -0.27055222 -0.26994315 -0.2692122 -0.033267528 -0.047869906 -0.050032064 0.16282524 0.17702723 0.172417 0.14684218 0.14223212 0.14653295 0.2784819 0.27089873 0.27471355"
  ld_ref = sh_to_ld(np.array([float(x) for x in sh_text_ref.split(" ")])[None, ...]).reshape(-1)
  ld_ref = ld_ref / np.linalg.norm(ld_ref)
  print("Ref Direction : ", ld_ref)
  
  inp_sh = sh_np  # [27, ]
  ld = sh_to_ld(np.array(inp_sh)[None, ...]).reshape(-1)
  ld = ld / np.linalg.norm(ld)
  print("Init Direction : ", ld)
  
  # Compute rotation angle in the xy-plane
  theta_ref = np.arctan2(ld_ref[1], ld_ref[0])  # Ref azimuth
  theta_ld = np.arctan2(ld[1], ld[0])  # Input azimuth

  # Compute the rotation angle needed
  rotation_angle = np.degrees(theta_ref - theta_ld)
  print("Rotation Angle : ", rotation_angle)
  # Adjust the input SH to align with the reference azimuth
  inp_sh = rotateSH(inp_sh, 0, 0, 1, -rotation_angle)
  
  # assert False
  at = np.arctan2(ld[1], ld[0])
  at2 = np.arcsin(ld[2])

  # centered = rotateSH(centered, 0, 0, 1, at * 180 / np.pi)
  # centered = rotateSH(centered, 0, 1, 0, -at2 * 180 / np.pi)
  # drawSH(centered, f"centered.png")
  
  # light_path = np.load("./light_params.npy", allow_pickle=True)
  light_traj = np.load(args.light_path, allow_pickle=True).item()['traj']
  light_params = np.load(args.light_path, allow_pickle=True).item()['params']
  n = len(light_traj)
  a0 = 0
  # rchanged = 0.148
  rchanged = np.degrees((np.arccos(light_params['radius_0']) - np.arccos(light_params['radius_1']))) / (light_params['n']//2)
  # print(light_params)
  # print(np.degrees((np.arccos(light_params['radius_0']) - np.arccos(light_params['radius_1']))))
  # print(np.degrees((np.arccos(light_params['radius_1']) - np.arccos(light_params['radius_0']))))
  print("[#] Degree changed per frame : ", rchanged)
  angle = 0
  for i in tqdm.tqdm(range(n)):
    t = light_traj[i]["t"]  # 0~1
    tt = light_traj[i]["rel_angle"] + a0
    rr = light_traj[i]["rel_radius"]  # Relative change in radius

    if np.isclose(rr, 0):
      angle = angle # No change
    else: 
      angle += rchanged * np.sign(-rr)  # Change in radius


    # rr = light_traj[i]["rel_radius"] + rstart
    
    # if rr < -0.001:
    #   sp_r = -((np.arccos(rr)) * 180 / np.pi)
    # else: 
    #   sp_r = ((np.arccos(rr)) * 180 / np.pi)
    # rstart += rr
    
    # Rotate original to align with x (Preventing the spiral from unawarely orbiting)
    moved = rotateSH(inp_sh.copy(), 0, 0, 1, at * 180 / np.pi)
    # Rotate spiral (Decrease radius)
    moved = rotateSH(moved, 0, 1, 0, -angle)#* rr)
    # Rotate back to original
    moved = rotateSH(moved, 0, 0, 1, -at * 180 / np.pi)
    # Rotate spiral (Orbit around z)
    moved = rotateSH(moved, 0, 0, 1, tt * 180 / np.pi)
    
    ld = sh_to_ld(np.array(moved)[None, ...]).reshape(-1)
    drawSH(moved, f"./video_out/m_{i:03d}.png", ld=ld)
    a0 += light_traj[i]["rel_angle"]
    
  os.system(f"ffmpeg -y -framerate 30 -i video_out/m_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 video_spiral_read.mp4")
  os.system(f"ffmpeg -y -i output.mp4 -i video_spiral_read.mp4  -filter_complex \"[0:v][1:v]hstack=inputs=2\" cmp.mp4")
  exit()

# 65797.jpg
sh_text = "3.7764273 3.7647202 3.7740586 -0.45223573 -0.48492554 -0.48608136 0.3177414 0.34008643 0.33421847 -0.44365892 -0.47285086 -0.45525044 -0.27055222 -0.26994315 -0.2692122 -0.033267528 -0.047869906 -0.050032064 0.16282524 0.17702723 0.172417 0.14684218 0.14223212 0.14653295 0.2784819 0.27089873 0.27471355"
# 69809.jpg
# sh_text = "3.7345207 3.7213473 3.7314432 0.7694048 0.7836141 0.7984594 0.22823927 0.23264684 0.2287304 -0.4806327 -0.5140073 -0.48469663 0.038980436 0.039153174 0.039689075 -0.31383342 -0.30161086 -0.2934053 0.24958973 0.25237265 0.24871074 0.48290193 0.4683239 0.4711157 0.7713614 0.76298386 0.7751074"
# 62011.jpg 
# sh_text = "3.8343465 3.8336594 3.8284006 0.1729947 0.17721233 0.17012812 0.13337857 0.1411026 0.14271605 -0.45823875 -0.47945493 -0.49274385 -0.14240256 -0.13979109 -0.13815635 0.60094035 0.6050799 0.60517627 0.12989694 0.13691846 0.13731007 -0.14883745 -0.13712367 -0.14319716 0.57338154 0.5598248 0.5587959"
# 61992.jpg 
# sh_text = "3.5095832 3.5131266 3.5243688 0.6004163 0.62568486 0.6229789 0.10465601 0.10563105 0.101914756 -0.46082363 -0.46018302 -0.42792398 0.03933739 0.039667428 0.039996076 -0.091053 -0.07264688 -0.07542678 0.22469139 0.22775024 0.22464316 0.40681338 0.40605047 0.40891302 0.6666809 0.6633684 0.6743531"

sh_np = np.array([float(x) for x in sh_text.split(" ")])

if os.path.exists("video_out/"):
  os.system("rm -r ./video_out")
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

