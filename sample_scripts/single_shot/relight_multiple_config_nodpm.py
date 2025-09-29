import numpy as np
import argparse
import os, sys
sys.path.append('/home/mint/Dev/DiFaReli/difareli-faster')
# /home/mint/Dev/DiFaReli/difareli-faster/guided_diffusion/mint_logger.py
from guided_diffusion.mint_logger import createLogger
logger = createLogger()


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='model name')
parser.add_argument('--cfg_name', type=str, required=True, help='config name')
parser.add_argument('--ckpt_step', nargs='+', type=str, required=True, help='checkpoint step')
parser.add_argument('--ckpt_type', type=str, default='ema', help='checkpoint type')
parser.add_argument('--out_dir', type=str, required=True, help='checkpoint name')
parser.add_argument('--batch_size', type=int, required=True, help='batch size')
parser.add_argument('--itp', type=str, required=True, default='render_face', help='interpolation step')
parser.add_argument('--itp_step', type=int, required=True, help='interpolation step')
parser.add_argument('--render_batch_size', type=int, default=1, help='render batch size')
parser.add_argument('--gpu_id', type=int, required=True, help='gpu id')
parser.add_argument('--sample_idx', nargs='+', type=int, default=[0, 999999], help='sample index to run (start, end)')
parser.add_argument('--force_render', action='store_true', default=False)
parser.add_argument('--dataset', nargs='+', type=str, required=True, help='dataset name')
parser.add_argument('--eval_dir', type=str, default=None, help='eval dir')
parser.add_argument('--sample_pair_json', nargs='+', type=str, required=True, help='sample pair json file')
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--sdiff_dir', nargs='+', type=str, required=True, help='Shadow difference directory')
parser.add_argument('--rasterize_type', type=str, default='standard', help='rasterize type')
parser.add_argument('--scale_depth', nargs='+', type=int, default=[256])
parser.add_argument('--rotate_sh', action='store_true', default=False)
parser.add_argument('--rotate_sh_axis', type=int, default=2, help='axis to rotate sh, 0:x, 1:y, 2:z')
parser.add_argument('--relight_with_dst_c', action='store_true', default=False)
parser.add_argument('--relight_with_src_c', action='store_true', default=False)
parser.add_argument('--relight_with_given_c', action='store_true', default=False)
parser.add_argument('--c', nargs='+', type=float, default=[0.7])
parser.add_argument('--scale_sh', nargs='+', type=float, default=[1.0])
parser.add_argument('--adjust_contrast', nargs='+', type=float, default=None, help='adjust the contrast of rendered images, two values: down, up')
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

python ./relight_paired_nodpm.py --ckpt_selector ema --dataset ffhq --set valid --step 300000 --out_dir /data/mint/sampling/TPAMI_MajorRevision/FixPlastic --cfg_name paired+difareli+cs+nodpm+trainset_256.yaml --log_dir paired+difareli+cs+nodpm+trainset_256 --seed 47 --sample_pair_json /home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/sample_json/TPAMI_MajorRevision/rotateSH_axis2.json  --sample_pair_mode pair --itp render_face --itp_step 60 --batch_size 1 --gpu_id 0 --lerp --idx 0 2500 --shadow_diff_dir /data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/ --postproc_shadow_mask_smooth --save_vid --render_batch_size 60 --postfix rot2_0.1 --rotate_sh --rotate_sh_axis 2 --inverse_with_shadow_diff --rasterize_type pytorch3d --relight_with_given_c 0.1

'''

postfix = args.postfix
if postfix != '':
    postfix = '_' + postfix
    
assert args.relight_with_dst_c + args.relight_with_src_c + args.relight_with_given_c <= 1, "[#] Only one of --relight_with_dst_c, --relight_with_src_c, --relight_with_given_c can be set."
if args.adjust_contrast is not None:
    # Given a single list make it many-to-many e.g., [0, 1, 2, 3] -> [0, 0], [0, 1], [0, 2], [0, 3], [1, 0], ..., [3, 3]
    adjust_contrast = [[x, y] for x in args.adjust_contrast for y in args.adjust_contrast]
else:
    adjust_contrast = [None]

for ckpt in args.ckpt_step:
    for dataset in args.dataset:
        if dataset == 'ffhq_data2':
            shadow_diff_dir = "/data2/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/"
        else:
            shadow_diff_dir = "/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/"
        for sample_pair_json in args.sample_pair_json:
            for sdiff_dir in args.sdiff_dir:
                for scale_depth in args.scale_depth:
                    for scale_sh in args.scale_sh:
                        for cval in args.c:
                            for adj_contrast in adjust_contrast:
                                logger.warning("#"*100)
                                logger.info(f'[#] Running checkpoint {ckpt}...')
                                logger.info(f'[#] Dataset: {dataset}')
                                logger.info(f'[#] Sample pair json: {sample_pair_json}')
                                logger.info(f'[#] Shadow difference directory: {sdiff_dir}')
                                logger.info(f'[#] Scale depth: {scale_depth}')
                                logger.info(f'[#] c value: {cval}')
                                logger.info(f'[#] Rotate SH: {args.rotate_sh}, axis: {args.rotate_sh_axis}')
                                logger.info(f'[#] Scale SH: {scale_sh}')
                                logger.info(f'[#] Adjust contrast of rendered images: {adj_contrast is not None}')
                                
                                cmd = (
                                    f"""
                                    python relight_paired_nodpm.py --ckpt_selector {args.ckpt_type} --dataset {dataset} --set valid --step {ckpt} --out_dir {args.out_dir} \
                                    --cfg_name {args.cfg_name} --log_dir {args.model_dir} \
                                    --seed 47 --render_batch_size {args.render_batch_size} --save_vid\
                                    --sample_pair_json {sample_pair_json} --sample_pair_mode pair \
                                    --itp {args.itp} --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} --lerp --idx {args.sample_idx[0]} {args.sample_idx[1]} \
                                    --postproc_shadow_mask_smooth --inverse_with_shadow_diff --shadow_diff_dir {sdiff_dir}  \
                                    --pt_round 1 --scale_depth {scale_depth} --rasterize_type {args.rasterize_type} --scale_sh {scale_sh}\
                                    """
                                    )
                                if args.force_render: cmd += ' --force_render'
                                if args.eval_dir is not None: cmd += f' --eval_dir {args.eval_dir}'
                                
                                if args.relight_with_given_c:
                                    logger.info(f'[#] Relighting with given c: {cval}')
                                    cmd += f' --relight_with_given_c {cval}'
                                    pf = f'{cval}C'
                                elif args.relight_with_dst_c:
                                    logger.info(f'[#] Relighting with dst c.')
                                    cmd += f' --relight_with_dst_c'
                                    pf = f'dstC'
                                elif args.relight_with_src_c:
                                    logger.info(f'[#] Relighting with src c.')
                                    pf = f'srcC'
                                    # No additional argument needed, default is src c
                                else: raise ValueError('[#] One of --relight_with_dst_c, --relight_with_src_c, --relight_with_given_c must be set.')
                                
                                if args.rotate_sh:
                                    logger.info(f'[#] Rotating SH, axis: {args.rotate_sh_axis}')
                                    cmd += f' --rotate_sh --rotate_sh_axis {args.rotate_sh_axis}'
                                    pf = f'{pf}_rot{args.rotate_sh_axis}'
                                    
                                if adj_contrast is not None:
                                    logger.info(f'[#] Adjusting the contrast of rendered images with: down={adj_contrast[0]}, up={adj_contrast[1]}.')
                                    cmd += f' --adjust_contrast {adj_contrast[0]} {adj_contrast[1]}'
                                    pf = f'con{adj_contrast[0]}-{adj_contrast[1]}_{pf}'
                                
                                if postfix != '': cmd += f' --postfix SD{scale_depth}_{postfix}_{pf}'
                                else: cmd += f' --postfix {pf}_{scale_sh}sh'
                                
                                logger.info(f"cmd: {cmd}")
                                os.system(cmd)
                                logger.warning("#"*100)
