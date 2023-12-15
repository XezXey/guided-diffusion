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
parser.add_argument('--dataset', nargs='+', required=True, help='dataset name')
parser.add_argument('--gpu_id', type=int, required=True, help='gpu id')
parser.add_argument('--sample_idx', nargs='+', type=int, default=[0, 999999], help='sample index to run (start, end)')
parser.add_argument('--sample_pair_json', nargs='+', required=True, help='sample pair json file')
parser.add_argument('--eval_dir', type=str, default=None, help='eval dir')
parser.add_argument('--force_render', action='store_true', default=False)

# Solver
parser.add_argument('--solver_alg', nargs='+', type=str, default=['dpmsolver++'])
parser.add_argument('--solver_steps', nargs='+', type=int, default=[20])
parser.add_argument('--solver_method', nargs='+', type=str, default=['multistep'])
parser.add_argument('--solver_order', nargs='+', type=int, default=[2])
parser.add_argument('--solver_correcting_x0_fn', nargs='+', default=[None])

args = parser.parse_args()

'''
# Command
 python relight_with_solver.py --dataset ffhq --set valid --step 050000 
 --out_dir /data/mint/sampling/with_solver/unipc --cfg_name Masked_Face_woclip+BgNoHead+shadow.yaml 
 --log_dir Masked_Face_woclip+BgNoHead+shadow 
 --sample_pair_json ../faster_inference_script/sample_json/targetSH/gen_data_80perc_testset_n=46.json 
 --sample_pair_mode pair --itp render_face --itp_step 5 
 --batch_size 1 --gpu_id 0 --lerp --idx 0 99999 
 --solver 
'''
assert len(args.dataset) == len(args.sample_pair_json)

for dataset_idx, dataset in enumerate(args.dataset):
    for ckpt in args.ckpt_step:
        for solver_alg in args.solver_alg:
            for solver_method in args.solver_method:
                for solver_order in args.solver_order:
                    for solver_steps in args.solver_steps:
                        for solver_correcting_x0_fn in args.solver_correcting_x0_fn:
                            if (solver_correcting_x0_fn is not None) and (solver_alg != 'dpmsolver++'):
                                raise ValueError('solver_correcting_x0_fn is only available for dpmsolver++')
                                            
                            print("#"*100)
                            print(f'Running checkpoint: {ckpt}')
                            print(f'Solver: {solver_alg}')
                            print(f'Solver method: {solver_method}')
                            print(f'Solver steps: {solver_steps}')
                            print(f'Solver order: {solver_order}')
                            print(f'Solver correcting_x0_fn: {solver_correcting_x0_fn}')
                            print("#"*100)
                
                            cmd = (
                                f"""
                                python relight_with_solver.py --ckpt_selector {args.ckpt_type} --dataset {dataset} --set valid --step {ckpt} --out_dir {args.out_dir} \
                                --cfg_name {args.model}.yaml --log_dir {args.model} \
                                --diffusion_steps 1000 --timestep_respacing 1000 --seed 47 \
                                --sample_pair_json {args.sample_pair_json[dataset_idx]} --sample_pair_mode pair \
                                --itp {args.itp} --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} --lerp --idx {args.sample_idx[0]} {args.sample_idx[1]} \
                                --solver_alg {solver_alg} --solver_steps {solver_steps} --solver_method {solver_method} --solver_order {solver_order} --solver_correcting_x0_fn {solver_correcting_x0_fn} \
                                --postfix {solver_alg}_{solver_order}_{solver_method}_{solver_steps}_{solver_correcting_x0_fn} --eval_dir {args.eval_dir}\
                                """
                            )
                            if args.force_render: cmd += ' --force_render'
                            print(cmd)
                            os.system(cmd)
                            print("#"*100)
