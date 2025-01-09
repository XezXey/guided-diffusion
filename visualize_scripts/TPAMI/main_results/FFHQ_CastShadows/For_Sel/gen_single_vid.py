import json, subprocess, tqdm, os, glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ax', required=True)
parser.add_argument('--c', required=True)
args = parser.parse_args()

path = '/data/mint/sampling/TPAMI/main_result/ffhq/for_sel/'
# ax = 2
# c = 'srcC'
ax = args.ax
c = args.c

sampling_folder = f'/log=paired+difareli+cs+nodpm+trainset_256_cfg=paired+difareli+cs+nodpm+trainset_256.yaml_rot{ax}_{c}/'
pf = '/ema_300000/valid/render_face/reverse_sampling/'
out_dir = f'./Out/{sampling_folder}/'
sf = '/home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/sample_json/DiFaReli++/top50perc_shadow_for_rotate.json'

def sort_by_frame(path_list):
    frame_anno = []
    for p in path_list:
        frame_idx = os.path.splitext(p.split('/')[-1].split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_anno.append(int(frame_idx))
    sorted_idx = np.argsort(frame_anno)
    sorted_path_list = []
    for idx in sorted_idx:
      sorted_path_list.append(path_list[idx])
    return sorted_path_list

with open(sf, 'r') as f:
    data = json.load(f)
    data = data['pair']

os.makedirs(out_dir, exist_ok=True)
for pid, pair in tqdm.tqdm(data.items()):
    res_path = f"{path}/{sampling_folder}/{pf}/src={pair['src']}/dst={pair['dst']}/Lerp_1000/n_frames=60/"
    if not os.path.exists(res_path):
        continue
    else:
        out_fn = f"{out_dir}/{pid}_src={pair['src']}_dst={pair['dst']}.mp4"
        frames = glob.glob(f"{res_path}/res_f*.png")
        frames = sort_by_frame(frames)[1:]
        # Copy frames to ./tmp/ folder and rename them to 0001.png, 0002.png, ...
        os.makedirs(f'./tmp/{ax}_{c}/', exist_ok=True)
        for i, frame in enumerate(frames):
            os.system(f"cp {frame} ./tmp/{ax}_{c}/{i:04d}.png")
        # Use ffmpeg with highest quality settings (e.g, crf 17) to create a video
        # -y: overwrite output file if it exists
        # -r: frame rate = 24
        cmd = f"ffmpeg -y -r 24 -i ./tmp/{ax}_{c}/%04d.png -c:v libx264 -crf 17 -pix_fmt yuv420p {out_fn}"
        # Run subprocess without ffmpeg output
        subprocess.run(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)