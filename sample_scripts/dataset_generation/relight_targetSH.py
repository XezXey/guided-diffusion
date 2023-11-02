import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--set_', type=str, required=True, help='train or valid')
parser.add_argument('--ckpt_step', nargs='+', type=str, required=True, help='checkpoint step')
parser.add_argument('--ckpt_type', type=str, default='ema', help='checkpoint type')
parser.add_argument('--out_dir', type=str, required=True, help='checkpoint name')
parser.add_argument('--batch_size', type=int, required=True, help='batch size')
parser.add_argument('--itp_step', type=int, required=True, help='interpolation step')
parser.add_argument('--gpu_id', type=int, required=True, help='gpu id')
parser.add_argument('--sample_idx', nargs='+', type=int, default=[0, 999999], help='sample index to run (start, end)')
parser.add_argument('--sample_pair_json', type=str, required=True, help='sample pair json file')
parser.add_argument('--diffusion_steps', type=int, default=1000, help='diffusion step')
parser.add_argument('--time_respace', nargs='+', type=str, default=[""], help='time respace')
parser.add_argument('--chunk_size', type=int, default=0, help='chunk size, This help to decrease the loading time of the sampling file')
args = parser.parse_args()

'''
# Command
python relight.py --ckpt_selector ema --dataset ffhq --set valid --step 100000 
--out_dir /data/mint/sampling/paired_training_experiment/targetSH 
--cfg_name paired+allunet_eps+ddst_128.yaml --log_dir paired+allunet_eps+ddst_128 
--diffusion_steps 1000 --seed 47 
--sample_pair_json ./sample_json/targetSH/gen_data_80perc_testset_n=46.json --sample_pair_mode pair 
--itp render_face --itp_step 5 --batch_size 5 --gpu_id 0 --lerp --idx 0 1000
'''

def chunking(start, end, chunk_size):
    if start < 0 or end < start or chunk_size <= 0:
        raise ValueError("Invalid input values")

    subchunks = []
    current = start
    while current < end:
        subchunks.append((current, min(current + chunk_size, end)))
        current += chunk_size

    return subchunks


for ckpt in args.ckpt_step:
    for time_respace in args.time_respace:
        for sub_chunk in chunking(args.sample_idx[0], args.sample_idx[1], args.chunk_size):
            print("#"*100)
            print(f'Running checkpoint {ckpt} w/ time_respace={time_respace} & diffusion_steps={args.diffusion_steps}')
            print(f'Running on sub_chunk {sub_chunk} from sample_idx {args.sample_idx}')
            print("#"*100)
            cmd = (
                f"""
                python relight.py --ckpt_selector {args.ckpt_type} --dataset ffhq --set {args.set_} --step {ckpt} --out_dir {args.out_dir} \
                --cfg_name {args.model}.yaml --log_dir {args.model} \
                --diffusion_steps {args.diffusion_steps} --timestep_respacing {time_respace} --seed 47 \
                --sample_pair_json {args.sample_pair_json} --sample_pair_mode pair \
                --itp render_face --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} --lerp --idx {sub_chunk[0]} {sub_chunk[1]}\
                --postfix step={time_respace}"""
                )
            print(cmd)
            os.system(cmd)
            print("#"*100)