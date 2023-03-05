#!/bin/bash
read -p "Input the parameters to run (shape, pose, exp, cam, all) : " pname
read -p "Input noise level to run (1e-1, 1e-2, ...) : " nlvl
read -p "Running on : " gpu
cd ../../../

# Variables
diffusion=1000

for lvl in $nlvl
do
    echo "[!] Running : "$pname - $lvl
    echo -e "\n\n"
    python relight_idx_robust_params.py --dataset mp_valid --set valid --step 085000 --out_dir /data/mint/sampling/Robustness/MP/"$pname" --cfg_name Masked_Face_woclip+BgNoHead+shadow_256.yaml --log_dir Masked_Face_woclip+BgNoHead+shadow_256 --diffusion_steps "$diffusion" --seed 47 --sample_pair_json /home/mint/guided-diffusion/sample_scripts/paper_script/sample_json/experiment/ffhq_figure_original.json --sample_pair_mode pair --itp render_face --itp_step 2 --batch_size 10 --gpu_id "$gpu" --lerp --idx 0 30 --noisy_cond /home/mint/guided-diffusion/experiment_scripts/statistic_test/stat_ffhq_training_deca.json "$lvl" "$pname" --postfix "$pname"_"$lvl"
    echo "[!] Done : "$pname - $lvl
done

echo "[!] Finished all : "$pname - $nlvl
                                                                                                                                                                                                                                                                                                                                                                                                                