import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import glob
import torchvision
import torch as th

def filter_vid(sample_vid):
    return sample_vid

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', type=str, required=True)
    args = parser.parse_args()

    sample_vid = glob.glob(f'{args.sample_dir}/*.mp4')

    sample_vid = filter_vid(sample_vid)

    video = []
    fps = []
    for vid_path in sample_vid:
        out = torchvision.io.read_video(vid_path, pts_unit='sec')
        vid = out[0]
        fps.append(out[2]['video_fps'])
        video.append(vid)
    
    assert fps.count(fps[0]) == len(fps)

    n_cols = 4
    n_rows = math.ceil(len(video) / n_cols)

    grid = [[[] for _ in range(n_cols)]] * n_rows

    # Video is in T x H x W x C
    vid_ptr = 0
    vid_shape = video[0].shape
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if vid_ptr < len(video):
                grid[i][j] = video[vid_ptr]
            else:
                grid[i][j] = th.zeros(vid_shape)
            vid_ptr+=1

    output = []
    for i in range(len(grid)):
        output.append(th.cat(grid[i], dim=2))

    combined = th.cat(output, dim=1)
    torchvision.io.write_video(filename=f'{args.sample_dir}/combined.mp4', video_array=combined, fps=fps[0])