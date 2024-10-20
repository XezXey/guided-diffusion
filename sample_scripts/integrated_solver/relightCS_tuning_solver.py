import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='model name')
parser.add_argument('--cfg_name', type=str, required=True, help='config name')
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
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--sdiff_dir', nargs='+', required=True, help='Shadow difference directory')
parser.add_argument('--scale_depth', nargs='+', type=int, default=[100, 256])
parser.add_argument('--save_vid', action='store_true', default=False)
parser.add_argument('--rasterize_type', type=str, default='standard')

# Solver
parser.add_argument('--solver_alg', nargs='+', type=str, default=['dpmsolver++'])
parser.add_argument('--solver_steps', nargs='+', type=int, default=[20])
parser.add_argument('--solver_method', nargs='+', type=str, default=['multistep'])
parser.add_argument('--solver_order', nargs='+', type=int, default=[2])
parser.add_argument('--solver_correcting_x0_fn', nargs='+', default=[None])

args = parser.parse_args()

'''
# Command
python ./relight.py --ckpt_selector ema 
--dataset mp_test --set valid 
--step 085000 
--out_dir /data/mint/sampling/TPAMI/cast_shadows_results_mp/mp_test/ts_1000/ 
--cfg_name DiFaReli_FsBg_Sdiff_SS_256_V100_inference.yaml 
--log_dir DiFaReli_FsBg_Sdiff_SS_256_V100 
--diffusion_steps 1000 --timestep_respacing 1000 
--seed 47 
--sample_pair_json /home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/paper_multipie/multipie_testset.json 
--sample_pair_mode pair --itp render_face --itp_step 2 --batch_size 2 
--gpu_id 0 --lerp --idx 600 700 
--postproc_shadow_mask_smooth --up_rate_for_AA 1
--postfix inv_with_sd_SS_1000ts_shtold_dstC
--shadow_diff_dir /data/mint/DPM_Dataset/MultiPIE/MultiPIE_testset/shadow_diff_SS_with_c_simplified/ --eval_dir /data/mint/TPAMI_evaluations/MP/pred/Ours/ours_difareli++_256/ts_1000/DiFaReli_FsBg_Sdiff_SS_256_V100_shtold_dstC/mp_test/ 
--inverse_with_shadow_diff --relight_with_dst_c --pt_round 1
'''
assert len(args.dataset) == len(args.sample_pair_json)
postfix = args.postfix
if postfix != '':
    postfix = '_' + postfix

for dataset_idx, dataset in enumerate(args.dataset):
    for ckpt in args.ckpt_step:
        for scale_depth in args.scale_depth:
            for solver_alg in args.solver_alg:
                for solver_method in args.solver_method:
                    for solver_order in args.solver_order:
                        for solver_steps in args.solver_steps:
                            for solver_correcting_x0_fn in args.solver_correcting_x0_fn:
                                if (solver_correcting_x0_fn is not None) and (solver_alg != 'dpmsolver++'):
                                    raise ValueError('solver_correcting_x0_fn is only available for dpmsolver++')
                                                
                                print("#"*100)
                                print(f'[#] Running checkpoint: {ckpt}')
                                print(f'[#] Dataset: {dataset}')
                                print(f'[#] Solver: {solver_alg}')
                                print(f'[#] Solver method: {solver_method}')
                                print(f'[#] Solver steps: {solver_steps}')
                                print(f'[#] Solver order: {solver_order}')
                                print(f'[#] Solver correcting_x0_fn: {solver_correcting_x0_fn}')
                                print(f'[#] Scale depth: {scale_depth}')
                                print("#"*100)
                    
                                cmd = (
                                    f"""
                                    python relightCS_with_solver.py --ckpt_selector {args.ckpt_type} --dataset {dataset} --set valid --step {ckpt} --out_dir {args.out_dir} \
                                    --cfg_name {args.cfg_name} --log_dir {args.model_dir} \
                                    --diffusion_steps 1000 --timestep_respacing 1000 --seed 47 \
                                    --sample_pair_json {args.sample_pair_json[dataset_idx]} --sample_pair_mode pair \
                                    --itp {args.itp} --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} --lerp --idx {args.sample_idx[0]} {args.sample_idx[1]} \
                                    --solver_alg {solver_alg} --solver_steps {solver_steps} --solver_method {solver_method} --solver_order {solver_order} --solver_correcting_x0_fn {solver_correcting_x0_fn} \
                                    --postfix {solver_alg}_{solver_order}_{solver_method}_{solver_steps}_{solver_correcting_x0_fn}_SD{scale_depth}_{postfix} --eval_dir {args.eval_dir} \
                                    --postproc_shadow_mask_smooth --up_rate_for_AA 1 --shadow_diff_dir {args.sdiff_dir[dataset_idx]} --inverse_with_shadow_diff \
                                    --relight_with_dst_c --pt_round 1 --scale_depth {scale_depth} --rasterize_type {args.rasterize_type}\
                                    """
                                )
                                if args.force_render: cmd += ' --force_render'
                                if args.save_vid: cmd += ' --save_vid'
                                print(cmd)
                                os.system(cmd)
                                print("#"*100)
