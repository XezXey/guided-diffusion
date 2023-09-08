#!/bin/bash
python ldm_relight.py --dataset ffhq --set valid --step 050000 --out_dir /data/mint/sampling/ldm --cfg_name difareli_ldm_kl_f4.yaml --log_dir difareli_ldm_kl_f4 --diffusion_steps 1000 --timestep_respacing 250 --seed 47 --sample_pair_json ./sample_json/targetSH/gen_data_80perc_testset_n=46.json --sample_pair_mode pair --itp render_face --itp_step 5 --batch_size 1 --gpu_id 1 --lerp --idx 0 1000 --save_vid && wait &&
python ldm_relight.py --dataset ffhq --set valid --step 050000 --out_dir /data/mint/sampling/ldm --cfg_name difareli_ldm_kl_f4.yaml --log_dir difareli_ldm_kl_f4 --diffusion_steps 1000 --timestep_respacing 250 --seed 47 --sample_pair_json ./sample_json/targetSH/gen_data_80perc_testset_n=46.json --sample_pair_mode pair --itp render_face --itp_step 5 --batch_size 1 --gpu_id 1 --lerp --idx 0 1000 --save_vid && wait &&
python ldm_relight.py --dataset ffhq --set valid --step 100000 --out_dir /data/mint/sampling/ldm --cfg_name difareli_ldm_kl_f4.yaml --log_dir difareli_ldm_kl_f4 --diffusion_steps 1000 --timestep_respacing 1000 --seed 47 --sample_pair_json ./sample_json/targetSH/gen_data_80perc_testset_n=46.json --sample_pair_mode pair --itp render_face --itp_step 5 --batch_size 1 --gpu_id 1 --lerp --idx 0 1000 --save_vid && wait &&
python ldm_relight.py --dataset ffhq --set valid --step 100000 --out_dir /data/mint/sampling/ldm --cfg_name difareli_ldm_kl_f4.yaml --log_dir difareli_ldm_kl_f4 --diffusion_steps 1000 --timestep_respacing 1000 --seed 47 --sample_pair_json ./sample_json/targetSH/gen_data_80perc_testset_n=46.json --sample_pair_mode pair --itp render_face --itp_step 5 --batch_size 1 --gpu_id 1 --lerp --idx 0 1000 --save_vid

