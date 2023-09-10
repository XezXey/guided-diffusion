import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--ckpt_step', nargs='+', type=str, required=True, help='checkpoint step')
parser.add_argument('--ckpt_type', type=str, default='ema', help='checkpoint type')
parser.add_argument('--out_dir', type=str, required=True, help='checkpoint name')
parser.add_argument('--batch_size', type=int, required=True, help='batch size')
parser.add_argument('--itp', type=str, required=True, default='render_face', help='interpolation step')
parser.add_argument('--itp_step', type=int, required=True, help='interpolation step')
parser.add_argument('--gpu_id', type=int, required=True, help='gpu id')
parser.add_argument('--sample_idx', nargs='+', type=int, default=[0, 999999], help='sample index to run (start, end)')
parser.add_argument('--sample_pair_json', type=str, required=True, help='sample pair json file')
parser.add_argument('--diffusion_steps', type=int, default=1000, help='diffusion step')
parser.add_argument('--time_respace', nargs='+', type=str, default=[""], help='time respace')
parser.add_argument('--force_render', action='store_true', default=False)
args = parser.parse_args()

'''
# Command
python ldm_relight.py --dataset ffhq --set valid --step 160000 --out_dir /data/mint/sampling/ldm 
--cfg_name difareli_ldm_kl_f8.yaml --log_dir difareli_ldm_kl_f8 --diffusion_steps 1000 --timestep_respacing 1000 
--seed 47 --sample_pair_json ./sample_json/targetSH/gen_data_80perc_testset_n=46.json --sample_pair_mode pair 
--itp render_face --itp_step 5 --batch_size 1 --gpu_id 0 --lerp --idx 0 1000 --save_vid --w_mm
'''

for ckpt in args.ckpt_step:
    for time_respace in args.time_respace:
        print("#"*100)
        print(f"Model: {args.model}")
        print(f'Running checkpoint: {ckpt}')
        print(f"Time_respace: {time_respace}")
        print(f"diffusion_steps: {args.diffusion_steps}")
        print("#"*100)
        cmd = (
            f"""
            python ldm_relight.py --ckpt_selector {args.ckpt_type} --dataset ffhq --set valid --step {ckpt} --out_dir {args.out_dir} \
            --cfg_name {args.model}.yaml --log_dir {args.model} --seed 47 \
            --sample_pair_json {args.sample_pair_json} --sample_pair_mode pair \
            --itp {args.itp} --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} \
            --lerp --idx {args.sample_idx[0]} {args.sample_idx[1]} --w_mm \
            """
            )
        if args.force_render: cmd += ' --force_render'
        print(cmd)
        os.system(cmd)
        print("#"*100)
