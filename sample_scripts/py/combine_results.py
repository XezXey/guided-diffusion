import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import glob, os
import torchvision
import torch as th

def filter_vid(sample_vid):
    filtered = []
    for vid in sample_vid:
        if args.filtered is not None and args.filtered in vid:
            filtered.append(vid)
            
    return filtered 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', type=str, required=True)
    parser.add_argument('--filtered', type=str, default=None)
    args = parser.parse_args()

    sample_vid = glob.glob(f'{args.sample_dir}/*.mp4')

    sample_vid = sorted(filter_vid(sample_vid))

    video = []
    fps = []
    for vid_path in sample_vid:
        frames, audio, info = torchvision.io.read_video(str(vid_path), pts_unit='sec')
        fps.append(info['video_fps'])
        video.append(frames)
    
    assert fps.count(fps[0]) == len(fps)

    n_cols = 3
    n_rows = math.ceil(len(video) / n_cols)

    
    gridlines = []
    for i in range(n_cols):
        gridlines.append([])
    grid = []
    for i in range(n_rows):
        grid.append(list(gridlines))
        
    # ptr=0
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         print(grid[i][j])
    #         grid[i][j] = ptr
    #         ptr+=1
    # print(grid)
    # exit()

    # Video is in T x H x W x C
    vid_ptr = 0
    vid_shape = video[0].shape
            
    for i in range(n_rows):
        for j in range(n_cols):
            # print(i, j)
            # print(grid[i][j])
            if vid_ptr < len(video):
                grid[i][j] = video[vid_ptr]
                # print(video[vid_ptr].shape)
            else:
                grid[i][j] = th.zeros(vid_shape, dtype=th.uint8)
            vid_ptr+=1

    output = []
        
    for i in range(n_rows):
        output.append(th.cat(grid[i], dim=2))

    combined = th.cat(output, dim=1)
    print("Final video shape : ", combined.shape)
    
    os.makedirs(f'{args.sample_dir}/combined/', exist_ok=True)
    torchvision.io.write_video(filename=f'{args.sample_dir}/combined/{args.filtered}_combined.mp4', video_array=combined, fps=fps[0])