import numpy as np
import argparse
import os
from guided_diffusion.mint_logger import createLogger
logger = createLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='model name')
parser.add_argument('--cfg_name', type=str, required=True, help='config name')
parser.add_argument('--ckpt_step', nargs='+', type=str, default=['300000'], help='checkpoint step')
parser.add_argument('--ckpt_type', type=str, default='ema', help='checkpoint type')
parser.add_argument('--out_dir', type=str, required=True, help='checkpoint name')
parser.add_argument('--batch_size', type=int, required=True, help='batch size')
parser.add_argument('--itp', type=str, default='render_face_hdr', help='interpolation step')
parser.add_argument('--itp_step', type=int, required=True, help='interpolation step')
parser.add_argument('--gpu_id', type=int, required=True, help='gpu id')
parser.add_argument('--sample_idx', nargs='+', type=int, default=[0, 999999], help='sample index to run (start, end)')
parser.add_argument('--force_render', action='store_true', default=False)
parser.add_argument('--dataset', nargs='+', type=str, default=['ffhq'], help='dataset name')
parser.add_argument('--eval_dir', type=str, default=None, help='eval dir')
parser.add_argument('--sample_pair_json', nargs='+', type=str, required=True, help='sample pair json file')
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--sdiff_dir', nargs='+', type=str, default=['/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/'], help='Shadow difference directory')
parser.add_argument('--rasterize_type', type=str, default='pytorch3d', help='rasterize type')
parser.add_argument('--scale_depth', nargs='+', type=int, default=[100, 256])
parser.add_argument('--use_shading_grey', action='store_true', default=False)
parser.add_argument('--hdr_dir', nargs='+', type=str, required=True, help='hdr file directory')
parser.add_argument('--relight_with_dst_c', action='store_true', default=False)
parser.add_argument('--relight_with_src_c', action='store_true', default=False)
parser.add_argument('--relight_with_given_c', action='store_true', default=False)
parser.add_argument('--c_list', nargs='+', default=['1.0'])
parser.add_argument('--rotate_sh_axis', nargs='+', type=int, default=[2], help='rotate sh axis')
parser.add_argument('--Lmax', type=int, default=2, help='sh order')
parser.add_argument('--tonemap_percentile', nargs='+', default=[50.0], help='Tonemap percentile for HDR rendering')
parser.add_argument('--tonemap_max_mapping', nargs='+', default=[0.5], help='Tonemap max mapping for HDR rendering')
args = parser.parse_args()

'''
# Command
python ./relight_paired_nodpm_with_hdr.py --ckpt_selector ema --dataset ffhq --set valid --step 300000 
--out_dir /data/mint/TPAMI_MajorRevision/Ours/ffhq_hdr/ --cfg_name paired+difareli+cs+nodpm+trainset_256.yaml 
--log_dir paired+difareli+cs+nodpm+trainset_256 --seed 47 
--sample_pair_json /home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/sample_json/TPAMI_MajorRevision/fq.json 
--sample_pair_mode pair 
--itp render_face_hdr --itp_step 60 --batch_size 1 --gpu_id 1 --lerp --idx 0 2500 
--shadow_diff_dir /data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/ 
--scale_depth 256 --pt_round 1 --postproc_shadow_mask_smooth --save_vid --render_batch_size 60 
--postfix rot1_maxC --rotate_sh --rotate_sh_axis 1 --inverse_with_shadow_diff --rasterize_type pytorch3d --relight_with_strongest_c 
--hdr /home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/neural_gaffer_environment_map_sample/117_hdrmaps_com_free_2K.exr

'''

postfix = args.postfix
if postfix != '':
    postfix = '_' + postfix
    
assert args.relight_with_dst_c + args.relight_with_src_c + args.relight_with_given_c <= 1, "[#] Only one of --relight_with_dst_c, --relight_with_src_c, --relight_with_given_c can be set."

for ckpt in args.ckpt_step:
    for dataset in args.dataset:
        if dataset == 'ffhq_data2':
            shadow_diff_dir = "/data2/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/"
        else:
            shadow_diff_dir = "/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/"
            
        for sample_pair_json in args.sample_pair_json:
            for hdr in args.hdr_dir:
                for scale_depth in args.scale_depth:
                    for rotate_sh_axis in args.rotate_sh_axis:
                        for tonemap_percentile in args.tonemap_percentile:
                            for tonemap_max_mapping in args.tonemap_max_mapping:
                                for c in args.c_list:
                                    logger.warning("#"*100)
                                    logger.info(f'[#] Running checkpoint {ckpt}...')
                                    logger.info(f'[#] Dataset: {dataset}')
                                    logger.info(f'[#] Sample pair json: {sample_pair_json}')
                                    logger.info(f'[#] HDR file: {hdr}')
                                    logger.info(f'[#] Scale depth: {scale_depth}')
                                    logger.info(f"[#] Use shading: {'sColor' if not args.use_shading_grey else 'sGrey'}")
                                    logger.info(f"[#] Rotate sh: {rotate_sh_axis}")
                                    logger.info(f'[#] Tonemap percentile: {tonemap_percentile}')
                                    logger.info(f'[#] Tonemap max mapping: {tonemap_max_mapping}') 
                                    cmd = (
                                        f"""
                                        python relight_paired_nodpm_with_hdr.py --ckpt_selector {args.ckpt_type} --dataset {dataset} --set valid --step {ckpt} --out_dir {args.out_dir} \
                                        --cfg_name {args.cfg_name} --log_dir {args.model_dir} \
                                        --seed 47 \
                                        --sample_pair_json {sample_pair_json} --sample_pair_mode pair \
                                        --itp {args.itp} --itp_step {args.itp_step} --batch_size {args.batch_size} --gpu_id {args.gpu_id} --lerp --idx {args.sample_idx[0]} {args.sample_idx[1]} \
                                        --shadow_diff_dir {shadow_diff_dir} \
                                        --scale_depth {scale_depth} --pt_round 1 --postproc_shadow_mask_smooth --save_vid --render_batch_size 60 \
                                        --rotate_sh --rotate_sh_axis {rotate_sh_axis} --inverse_with_shadow_diff --rasterize_type {args.rasterize_type}\
                                        --hdr {hdr} --Lmax {args.Lmax} --tonemap_percentile {tonemap_percentile} --tonemap_max_mapping {tonemap_max_mapping} \
                                        """
                                        )
                                    if args.force_render: cmd += ' --force_render'
                                    if args.eval_dir is not None: cmd += f' --eval_dir {args.eval_dir}'
                                    
                                    if args.relight_with_given_c:
                                        logger.info(f'[#] Relighting with given c: {c}')
                                        cmd += f' --relight_with_given_c {c}'
                                        pf = f'{c}C'
                                    elif args.relight_with_dst_c:
                                        logger.info(f'[#] Relighting with dst c.')
                                        cmd += f' --relight_with_dst_c'
                                        c = 'dst'
                                    elif args.relight_with_src_c:
                                        logger.info(f'[#] Relighting with src c.')
                                        c = 'src'
                                    logger.warning("#"*100)
                                    
                                    if postfix != '': cmd += f" --postfix SD{scale_depth}_{postfix}_{'sColor' if args.use_shading_grey == '' else 'sGrey'}_rAxis{rotate_sh_axis}"
                                    else: cmd += f" --postfix SD{scale_depth}_{c}C_{'sColor' if not args.use_shading_grey else 'sGrey'}_Lmax{args.Lmax}_tmperc{tonemap_percentile}_tmmax{tonemap_max_mapping}_rAxis{rotate_sh_axis}"
                                    if args.use_shading_grey: cmd += f' --use_shading_grey'
                                    print(cmd)
                                    os.system(cmd)
                                    print("#"*100)
