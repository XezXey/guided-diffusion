#!/bin/bash
read -p "Input the sub_dataset to run (e.g. Anakin_2, Sull_1, Jon_1) : " dname
read -p "Running on : " gpu
cd ../../

# Variables
diffusion=1000

for d in $dname
do
    echo "[!] Running : "$d
    echo -e "\n\n"
    python ./extract_noisemap/extract_noisemap.py --dataset Videos --sub_dataset "$d" --set valid --step 085000 --cfg_name Masked_Face_woclip+BgNoHead+shadow_256.yaml --log_dir Masked_Face_woclip+BgNoHead+shadow_256 --diffusion_steps 1000 --seed 47 --batch_size 5 --gpu_id "$gpu" --idx 0 1000
    echo "[!] Done : "$d
done

echo "[!] Finished all : "$dname
                                                                                                                                                                                                                                                                                                                                                                                                                