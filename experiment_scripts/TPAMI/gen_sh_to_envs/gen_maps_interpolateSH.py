# Rule: tmux send -t 1 "python rotate2.py" Enter
import numpy as np
import tqdm, json
import torch as pt
from torchvision.utils import save_image
import pandas as pd
from torchvision import transforms
from PIL import Image
import itertools
from collections import defaultdict
import os
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import pyshtools as pysh

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--sh_file', type=str, required=True)
parser.add_argument('--sample_json', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--set_', type=str, default='valid')
parser.add_argument('--n_step', type=int, default=2)
args = parser.parse_args()

if args.dataset_name in ['mp_test', 'mp_test2', 'mp_valid', 'mp_valid2']:
    if args.dataset_name == 'mp_test':
        sub_f = 'MultiPIE_testset'
    elif args.dataset_name == 'mp_test2':
        sub_f = 'MultiPIE_testset2'
    elif args.dataset_name == 'mp_valid':
        sub_f = 'MultiPIE_validset'
    elif args.dataset_name == 'mp_valid2':
        sub_f = 'MultiPIE_validset2'
    else: 
        raise ValueError(f'Unknown dataset name of {args.dataset_name}...')
     
    img_path = f'/data/mint/DPM_Dataset/MultiPIE/{sub_f}/mp_aligned/{args.set_}/'
    sh_path = f'/data/mint/DPM_Dataset/MultiPIE/{sub_f}/params/{args.set_}/ffhq-{args.set_}-light-anno.txt'

elif args.dataset_name in ['ffhq', 'ffhq_target', 'ffhq_target2', 'ffhq_rotate', 'ffhq_shadows', 'ffhq_diffuse']:
    img_path = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/{args.set_}/'
    sh_path = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/params/{args.set_}/ffhq-{args.set_}-light-anno.txt'
else:
    raise ValueError(f'Unknown dataset name of {args.dataset_name}...')
    

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

def drawUnwrappedSphere(x, y, z, output='unwrapped_sphere.png'):
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

# Draw using plotly and save as html
def drawUnwrappedSpherePlotly(x, y, z, output='unwrapped_sphere.html'):
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    
    # Create a figure
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5), name='3D Plot'))
    
    # # Add grid lines
    # fig.update_layout(scene=dict(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='black'),
    #                             yaxis=dict(showgrid=True, gridwidth=1, gridcolor='black'),
    #                             zaxis=dict(showgrid=True, gridwidth=1, gridcolor='black')))
    
    # Save the figure
    fig.write_html(output)

def drawMap(sh, img_size=256):
    n = img_size
    lr = pt.linspace(0, 2 * np.pi, 2 * n)
    ud = pt.linspace(0, np.pi, n)
    ud, lr = pt.meshgrid(ud, lr)

    # we want to make the sphere unwrap at the center of this map,
    # so the left-most column is the furthest-away point on the sphere
    # lr going counter-clockwise = increasing in value.
    # ud starting from 0 (top) to pi (bottom).
    # Lattitude = azimuth = deg from one of xz axis
    # Longtitude = elevation = deg from y-axis
    # In standard unitsphere orientation;
    # z = up (so set y = pt(cos(ud))) ref. https://www.learningaboutelectronics.com/Articles/Spherical-to-cartesian-rectangular-coordinate-converter-calculator.php
    x = -pt.sin(ud) * pt.sin(lr)  # Negative to ensure correct left-right orientation
    y = pt.cos(ud)                # No negative sign needed for up-down orientation
    z = -pt.sin(ud) * pt.cos(lr)  # Negative to ensure correct front-back orientation

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

def interpolateSH(inp_sh, tgt_sh, n_step, out_path, img_size=256):
    # print(inp_sh.shape, tgt_sh.shape)
    intp = np.linspace(0, 1, n_step)
    out = inp_sh * (1 - intp[:, None]) + tgt_sh * intp[:, None]
    # print(out.shape)
    for i in tqdm.tqdm(range(n_step), desc=f'Interpolating SH for {n_step} frames...', leave=False):
        sh = out[i]
        ball, map, map_centered, map_clean, combined = drawSH(sh, img_size)
        for j in zip([ball, map, map_centered, map_clean, combined], ['ball', 'map', 'map_centered', 'map_clean', 'combined']):
            save_image(j[0], f"{out_path}/{j[1]}/m_{i:03d}.png")
            
    for j in zip([ball, map, map_centered, map_clean, combined], ['ball', 'map', 'map_centered', 'map_clean', 'combined']):
        os.system(f"ffmpeg -loglevel warning -y -i {out_path}/{j[1]}/m_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 {out_path}/{j[1]}.mp4")
    
    
def read_params(path):
    params = pd.read_csv(path, header=None, sep=" ", index_col=False, lineterminator='\n')
    params.rename(columns={0:'img_name'}, inplace=True)
    params = params.set_index('img_name').T.to_dict('list')
    return params

def swap_key(params):
    params_s = defaultdict(dict)
    for params_name, v in params.items():
        for img_name, params_value in v.items():
            params_s[img_name][params_name] = np.array(params_value).astype(np.float64).flatten()

    return params_s
  
if __name__ == '__main__':
    
    # Load SH values
    sh = {'light':read_params(sh_path)}
    sh = swap_key(sh)	# returned foramt is {'img_name': {'light': [sh_values]}}
    # Load sample json
    with open(args.sample_json, 'r') as f:
        sample = json.load(f)['pair']
    
    out_path = f'{args.out_path}/{args.dataset_name}/{args.set_}/'
    os.makedirs(out_path, exist_ok=True)
    for pair_id, pair in tqdm.tqdm(sample.items(), desc='Processing pairs...'):
        src_name = pair['src']
        dst_name = pair['dst']
        sub_out = f'{out_path}/{pair_id}_src={src_name}_dst={dst_name}/n_step={args.n_step}/'
        os.makedirs(sub_out, exist_ok=True)
        
        # print(pair_id, pair)
        inp_sh_np = sh[src_name]['light']
        tgt_sh_np = sh[dst_name]['light']
        
        for j in ['ball', 'map', 'map_centered', 'map_clean', 'combined']:
            os.makedirs(f"{sub_out}/{j}", exist_ok=True)
        
        interpolateSH(inp_sh_np, tgt_sh_np, args.n_step, img_size=256, out_path=sub_out)
        
        os.system(f'cp {img_path}/{src_name} {sub_out}/src={src_name}')
        os.system(f'cp {img_path}/{dst_name} {sub_out}/dst={dst_name}')
        
        misc = {'src':src_name, 'dst':dst_name, 'src_sh':inp_sh_np.tolist(), 'dst_sh':tgt_sh_np.tolist(), 'n_step':args.n_step, 'img_size':256}
        with open(f'{sub_out}/misc.json', 'w') as f:
            json.dump(misc, f, indent=4)