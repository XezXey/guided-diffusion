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
parser.add_argument('--dataset', type=str, default='ffhq', help='dataset name')
parser.add_argument('--eval_dir', type=str, default=None, help='eval dir')
args = parser.parse_args()

'''
# Command
python relight_paired_nodpm.py --ckpt_selector ema --dataset ffhq --set valid --step 100000 
--out_dir /data/mint/sampling/paired_training_experiment/targetSH 
--cfg_name paired+allunet_eps+ddst_128.yaml --log_dir paired+allunet_eps+ddst_128 
--diffusion_steps 1000 --seed 47 
--sample_pair_json ./sample_json/targetSH/gen_data_80perc_testset_n=46.json --sample_pair_mode pair 
--itp render_face --itp_step 5 --batch_size 5 --gpu_id 0 --lerp --idx 0 1000
'''

for ckpt in args.ckpt_step:
    for time_respace in args.time_respace:
        print("#"*100)
        print(f'Running checkpoint {ckpt} w/ time_respace={time_respace} & diffusion_steps={args.diffusion_steps}')
        print("#"*100)
        if time_respace == "":
            cmd = (
                f"""
                python relight_paired_nodpm.py --ckpt_selector {args.ckpt_type} --dataset {args.dataset} --set valid --step {ckpt} --out_dir {args.out_dir} \
                --cfg_name {args.model}.yaml --log_dir {args.model} \
                --seed 47 \
                --sample_pair_json {args.sample_pair_json} --sample_pair_mode pair \
                --itp {args.itp} --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} --lerp --idx {args.sample_idx[0]} {args.sample_idx[1]}"""
                )
        else:
            cmd = (
                f"""
                python relight_paired_nodpm.py --ckpt_selector {args.ckpt_type} --dataset {args.dataset} --set valid --step {ckpt} --out_dir {args.out_dir} \
                --cfg_name {args.model}.yaml --log_dir {args.model} \
                --timestep_respacing {time_respace} --seed 47 \
                --sample_pair_json {args.sample_pair_json} --sample_pair_mode pair \
                --itp {args.itp} --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} --lerp --idx {args.sample_idx[0]} {args.sample_idx[1]}"""
                )
        if args.force_render: cmd += ' --force_render'
        if args.eval_dir is not None: cmd += f' --eval_dir {args.eval_dir}'
        print(cmd)
        os.system(cmd)
        print("#"*100)
