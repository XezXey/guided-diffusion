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
parser.add_argument('--gpu_id', type=int, required=True, help='gpu id')
parser.add_argument('--sample_idx', nargs='+', type=int, default=[0, 999999], help='sample index to run (start, end)')
parser.add_argument('--sample_pair_json', type=str, required=True, help='sample pair json file')
parser.add_argument('--force_render', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='ffhq', help='dataset name')
parser.add_argument('--eval_dir', type=str, default=None, help='eval dir')
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--sdiff_dir', type=str, required=True, help='Shadow difference directory')
args = parser.parse_args()

'''
# Command
python relight_paired_nodpm.py --ckpt_selector ema --dataset mp_valid2_data2 
--set valid --step 200000 
--out_dir /data2/mint/sampling/TPAMI/main_result/difarelit++_cast_shadows/mp_model_selection/ 
--cfg_name paired+difareli+cs+nodpm+trainset_256.yaml --log_dir paired+difareli+cs+nodpm+trainset_256 
--seed 47 
--sample_pair_json /home2/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/paper_multipie/multipie_validset2.json 
--sample_pair_mode pair --itp render_face --itp_step 2 --batch_size 1 --gpu_id 0 --lerp 
--idx 0 1 --shadow_diff_dir /data2/mint/DPM_Dataset/MultiPIE/MultiPIE_validset2/shadow_diff_SS_with_c_simplified/ 
--eval_dir /data2/mint/TPAMI_evaluations/MP/pred/Ours/ours_difareli++_single_shot/ 
--rasterize_type pytorch3d --postproc_shadow_mask_smooth --relight_with_dst_c --pt_round 1 --scale_depth 256

'''

postfix = args.postfix
if postfix != '':
    postfix = '_' + postfix

for ckpt in args.ckpt_step:
    print("#"*100)
    print(f'Running checkpoint {ckpt}...')
    print("#"*100)
    cmd = (
        f"""
        python relight_paired_nodpm.py --ckpt_selector {args.ckpt_type} --dataset {args.dataset} --set valid --step {ckpt} --out_dir {args.out_dir} \
        --cfg_name {args.cfg_name} --log_dir {args.model_dir} \
        --seed 47 \
        --sample_pair_json {args.sample_pair_json} --sample_pair_mode pair \
        --itp {args.itp} --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} --lerp --idx {args.sample_idx[0]} {args.sample_idx[1]} \
        --eval_dir {args.eval_dir} \
        --postproc_shadow_mask_smooth --up_rate_for_AA 1 --shadow_diff_dir {args.sdiff_dir}  \
        --relight_with_dst_c --pt_round 1 --scale_depth 256.\
        """
        )
    if args.force_render: cmd += ' --force_render'
    if args.eval_dir is not None: cmd += f' --eval_dir {args.eval_dir}'
    if postfix != '': cmd += f' --postfix {postfix}'
    print(cmd)
    os.system(cmd)
    print("#"*100)
