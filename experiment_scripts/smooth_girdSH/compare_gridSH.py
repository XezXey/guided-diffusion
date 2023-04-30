import cv2 as cv
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cx', type=float, default=0)
parser.add_argument('--cy', type=float, default=0)
parser.add_argument('--rx', type=float, default=1)
parser.add_argument('--ry', type=float, default=1)
parser.add_argument('--resize', action='store_true', default=False)
parser.add_argument('--baseline', action='store_true', default=False)
parser.add_argument('--model', )
args = parser.parse_args()

def compare_smooth_spiral(sj_name, n_frames, savepath):

    cfg = 'cx0_rx1_cy0_ry1'
    # Change path here!!!
    baseline = f'./vid_out_testpath/Masked_Face_woclip+BgNoHead+shadow_256/{cfg}/{sj_name}.mp4'
    out1 = f'./vid_out_testpath/paired+allenc_eps+ddst/{cfg}/{sj_name}.mp4'
    out2 = f'./vid_out_testpath/paired+allenc_eps+ddst+presample/{cfg}/{sj_name}.mp4'
    out3 = f'./vid_out_testpath/paired+allunet_eps+ddst/{cfg}/{sj_name}.mp4'
    out4 = f'./vid_out_testpath/paired+allunet_eps+ddst+presample/{cfg}/{sj_name}.mp4'
    if (not os.path.exists(baseline)) or (not os.path.exists(out1)) or (not os.path.exists(out2)) or (not os.path.exists(out2)):
        print(f"{sj_name} : No file...")
        return

    os.system(f"ffmpeg -y -i {baseline} -i {out1} -i {out2} -i {out3} -i {out4} -filter_complex \"[0:v][1:v][2:v][3:v][4:v]hstack=5,format=yuv420p[v]\" -map \"[v]\" ./cmp/{sj_name}.mp4")


if __name__ == '__main__':
    # all_sj = os.listdir(f'/data/mint/sampling/ICCV_rotate_light_top25shadow/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml/ema_085000/valid/render_face/reverse_sampling/')
    # all_sj = os.listdir(f'/data/mint/sampling/')
    # for i, sj in tqdm.tqdm(enumerate(all_sj)):
    for i, sj in tqdm.tqdm(enumerate(['src=60265.jpg', 'src=60268.jpg', 'src=60340.jpg', 'src=60374.jpg', 'src=60414.jpg', 'src=60865.jpg', 'src=61003.jpg', 'src=61062.jpg', 'src=61777.jpg'])):
        compare_smooth_spiral(sj_name = sj, n_frames=49, savepath=False if i==0 else False)