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
parser.add_argument('--model', nargs='+', required=True)
parser.add_argument('--ckpt', nargs='+', required=True)
parser.add_argument('--out', default='./cmp')
args = parser.parse_args()

def compare_smooth_spiral(sj_name, n_frames, savepath):

    cfg = 'cx0_rx1_cy0_ry1'
    # Change path here!!!
    baseline = f'./vid_out_testpath/Masked_Face_woclip+BgNoHead+shadow_256/{cfg}/{sj_name}.mp4'
    out = []
    for m, cp in zip(args.model, args.ckpt):
        out.append(f'./vid_out_testpath/{m}/{cfg}/{cp}/{sj_name}.mp4')
    # out1 = f'./vid_out_testpath/paired+allenc_eps+ddst_128/{cfg}/{sj_name}.mp4'
    # out2 = f'./vid_out_testpath/paired+allenc_eps+ddst+nobg_128/{cfg}/{sj_name}.mp4'
    # out3 = f'./vid_out_testpath/paired+allenc_eps+nodpm_128/{cfg}/{sj_name}.mp4'
    # out4 = f'./vid_out_testpath/paired+allunet_eps+ddst_128/{cfg}/{sj_name}.mp4'
    # out5 = f'./vid_out_testpath/paired+allunet_eps+ddst+nobg_128/{cfg}/{sj_name}.mp4'
    # out6 = f'./vid_out_testpath/paired+allunet_eps+nodpm_128/{cfg}/{sj_name}.mp4'
    if (not os.path.exists(baseline)) or np.any([not os.path.exists(o) for o in out]):
        print(f"{sj_name} : No file...")
        return

    instream = ' -i ' + ' -i '.join(out)
    vidx = ''.join([f'[{i}:v]' for i in range(1, len(out)+1)])
    # print(vidx)
    # print(instream)
    # exit()
    os.makedirs(args.out, exist_ok=True)
    os.system(f"ffmpeg -y -i {baseline} {instream} -filter_complex \"[0:v]{vidx}hstack={len(out)+1},format=yuv420p[v]\" -map \"[v]\" ./{args.out}/{sj_name}.mp4")


if __name__ == '__main__':
    # all_sj = os.listdir(f'/data/mint/sampling/ICCV_rotate_light_top25shadow/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml/ema_085000/valid/render_face/reverse_sampling/')
    # all_sj = os.listdir(f'/data/mint/sampling/')
    # for i, sj in tqdm.tqdm(enumerate(all_sj)):
    for i, sj in tqdm.tqdm(enumerate(['src=60265.jpg', 'src=60268.jpg', 'src=60340.jpg', 'src=60374.jpg', 'src=60414.jpg', 'src=60865.jpg', 'src=61003.jpg', 'src=61062.jpg', 'src=61777.jpg', 'src=62683.jpg', 'src=66386.jpg', 'src=67411.jpg', 'src=68568.jpg', 'src=68862.jpg'])):
        compare_smooth_spiral(sj_name = sj, n_frames=49, savepath=False if i==0 else False)