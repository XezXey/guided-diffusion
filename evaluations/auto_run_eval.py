import numpy as np
import os, glob, json, subprocess
import argparse

'''
Command: 
$python evaluator.py 
    --gt /data/mint/DPM_Dataset/MultiPIE/MultiPIE_validset/mp_aligned_128/valid 
    --pred /data/mint/TPAMI_evaluations/MP/pred/Ours/ours_paired+allunet+nobg+nodpm+trainset_128/ema_150000/out/    
    --mask /data/mint/DPM_Dataset/MultiPIE/MultiPIE_validset/face_segment/valid/anno/ 
    --batch_size 10 
    --face_part faceseg_faceskin 
    --postfix _ours_paired+allunet+nobg+nodpm+trainset_128_validset_n200 
    --out_score_dir /data/mint/TPAMI_evaluations/MP/score/ablation/arch 
    --ds_mask --save_for_dssim /data/mint/TPAMI_evaluations_dssim
'''


parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, required=True)
parser.add_argument('--model_pred_dir', type=str, required=True)
parser.add_argument('--mask', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--face_part', type=str, default='faceseg_faceskin')
parser.add_argument('--out_score_dir', type=str, default='')
parser.add_argument('--ds_mask', action='store_true', default=False)
parser.add_argument('--save_for_dssim', type=str, default='')
args = parser.parse_args()

ckpt = 'ema_085000'
model = glob.glob(f'{args.model_pred_dir}/{ckpt}/*')
if args.ds_mask:
    args.ds_mask = '--ds_mask'
else: args.ds_mask = ''

for m in model:
    cmd = f"""python evaluator.py \
        --gt {args.gt} \
        --pred {m} \
        --mask {args.mask} \
        --batch_size {args.batch_size} \
        --face_part {args.face_part} \
        --postfix _{m.split('/')[-1]} \
        --out_score_dir {args.out_score_dir} \
        {args.ds_mask} \
        --save_for_dssim {args.save_for_dssim} \
    """
    print("#"*100)
    print("Running: ", cmd)
    subprocess.run([x for x in cmd.split(' ') if x != ''])
    print("#"*100)