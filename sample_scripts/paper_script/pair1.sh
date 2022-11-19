#!/bin/bash
python relight_idx_supmat.py --dataset ffhq --set valid --step 085000 --out_dir /data/mint/sampling/FFHQ_Hope_Supmat_RN_lastday_real --cfg_name Masked_Face_woclip+BgNoHead+shadow_256.yaml --log_dir Masked_Face_woclip+BgNoHead+shadow_256 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/fucking_lastday/main_rotate_normals_web.json --sample_pair_mode pair --itp render_face --itp_step 60 --batch_size 10 --gpu_id 0 --lerp --postfix with_MM_6e-1 --idx 0 1 --save_vid --rotate_normals --scale_sh 0.6  && wait &&
python relight_idx_supmat.py --dataset ffhq --set valid --step 085000 --out_dir /data/mint/sampling/FFHQ_Hope_Supmat_RN_lastday_real --cfg_name Masked_Face_woclip+BgNoHead+shadow_256.yaml --log_dir Masked_Face_woclip+BgNoHead+shadow_256 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/fucking_lastday/main_rotate_normals_web.json --sample_pair_mode pair --itp render_face --itp_step 60 --batch_size 10 --gpu_id 0 --lerp --postfix with_MM_8e-1 --idx 0 1 --save_vid --rotate_normals --scale_sh 0.8 && wait &&
# python relight_idx_supmat.py --dataset ffhq --set valid --step 085000 --out_dir /data/mint/sampling/FFHQ_Hope_Supmat_RN_lastday_real --cfg_name Masked_Face_woclip+BgNoHead+shadow_256.yaml --log_dir Masked_Face_woclip+BgNoHead+shadow_256 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/fucking_lastday/main_rotate_normals_web.json --sample_pair_mode pair --itp render_face --itp_step 60 --batch_size 10 --gpu_id 0 --lerp --postfix with_MM_1 --idx 0 1 --save_vid --rotate_normals --scale_sh 1.0 && wait &&
python relight_idx_supmat.py --dataset ffhq --set valid --step 085000 --out_dir /data/mint/sampling/FFHQ_Hope_Supmat_RN_lastday_real --cfg_name Masked_Face_woclip+BgNoHead+shadow_256.yaml --log_dir Masked_Face_woclip+BgNoHead+shadow_256 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/fucking_lastday/main_rotate_normals_web.json --sample_pair_mode pair --itp render_face --itp_step 60 --batch_size 10 --gpu_id 0 --lerp --postfix with_MM_11e-1 --idx 0 1 --save_vid --rotate_normals --scale_sh 1.1 && wait &&
python relight_idx_supmat.py --dataset ffhq --set valid --step 085000 --out_dir /data/mint/sampling/FFHQ_Hope_Supmat_RN_lastday_real --cfg_name Masked_Face_woclip+BgNoHead+shadow_256.yaml --log_dir Masked_Face_woclip+BgNoHead+shadow_256 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/fucking_lastday/main_rotate_normals_web.json --sample_pair_mode pair --itp render_face --itp_step 60 --batch_size 10 --gpu_id 0 --lerp --postfix with_MM_13e-1 --idx 0 1 --save_vid --rotate_normals --scale_sh 1.3