#!/bin/bash

python relight_idx.py --dataset ffhq --set valid --step 085000 --out_dir /data/mint/sampling/FFHQ_Hope --cfg_name Masked_Face_woclip+BgNoHead+shadow_256.yaml --log_dir Masked_Face_woclip+BgNoHead+shadow_256 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/all_ffhq.json --sample_pair_mode pair --itp render_face --itp_step 2 --gpu_id 2 --lerp --postfix mean_matching --idx 0 100 && wait &&

python relight_idx.py --dataset ffhq --set valid --step 110000 --out_dir /data/mint/sampling/FFHQ_Hope --cfg_name Masked_Face_woclip+BgNoHead+shadow_256.yaml --log_dir Masked_Face_woclip+BgNoHead+shadow_256 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/all_ffhq.json --sample_pair_mode pair --itp render_face --itp_step 2 --gpu_id 2 --lerp --postfix mean_matching --idx 0 100

