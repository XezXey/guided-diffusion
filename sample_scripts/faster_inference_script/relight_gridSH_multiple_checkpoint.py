import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--ckpt_step', nargs='+', type=str, required=True, help='checkpoint step')
parser.add_argument('--ckpt_type', type=str, default='ema', help='checkpoint type')
parser.add_argument('--out_dir', type=str, required=True, help='checkpoint name')
parser.add_argument('--batch_size', type=int, required=True, help='batch size')
parser.add_argument('--itp_step', type=int, required=True, help='interpolation step')
parser.add_argument('--gpu_id', type=int, required=True, help='gpu id')
parser.add_argument('--sample_idx', nargs='+', type=int, default=[0, 999999], help='sample index to run (start, end)')
parser.add_argument('--sample_pair_json', type=str, required=True, help='sample pair json file')
args = parser.parse_args()

'''
# Command

python relight_paired_gridSH.py --ckpt_selector model --dataset ffhq --set valid --step 020000 --out_dir /data/mint/sampling/paired_training_gridSH 
--cfg_name paired+allunet_eps+nodpm_128.yaml --log_dir paired+allunet_eps+nodpm_128 
--diffusion_steps 1000 --seed 47 
--sample_pair_json ./sample_json/gridSH/gen_data_80perc.json --sample_pair_mode pair 
--itp render_face --itp_step 49 --batch_size 5 --gpu_id 0 --lerp --save_vid --idx 0 10 --sh_grid_size 7 --sh_span_x -4 4 --sh_span_y 4 -4 --use_sh --sh_scale 0.6
'''


for ckpt in args.ckpt_step:
    print("#"*100)
    print(f'Running checkpoint {ckpt}')
    print("#"*100)
    cmd = (
        f"""
        python relight_paired_gridSH.py --ckpt_selector {args.ckpt_type} --dataset ffhq --set valid --step {ckpt} --out_dir {args.out_dir} \
        --cfg_name {args.model}.yaml --log_dir {args.model} \
        --diffusion_steps 1000 --seed 47 \
        --sample_pair_json {args.sample_pair_json} --sample_pair_mode pair \
        --itp render_face --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} --lerp --save_vid --idx {args.sample_idx[0]} {args.sample_idx[1]} \
        --sh_grid_size 7 --sh_span_x -4 4 --sh_span_y 4 -4 --use_sh --sh_scale 0.6
        """
        )
    print(cmd)
    os.system(cmd)
    print("#"*100)