#!/bin/bash
read -p "Input the parameters to run (shape, pose, exp, cam, all) : " pname
read -p "Input noise level to run (1e-1, 1e-2, ...) : " nlvl
read -p "Running on : " gpu
cd ../

# Variables
diffusion=1000

for p in $pname
do
    for lvl in $nlvl
    do
        echo "[!] Running : "$p - $lvl
        echo -e "\n\n"
        CUDA_VISIBLE_DEVICES="$gpu" python evaluator.py --gt /data/mint/DPM_Dataset/MultiPIE/MultiPIE_validset/mp_aligned_128/valid --pred /data/mint/robustness_for_evaluation/log=Masked_Face_woclip+BgNoHead+shadow_cfg=Masked_Face_woclip+BgNoHead+shadow.yaml/"$p"/log=Masked_Face_woclip+BgNoHead+shadow_cfg=Masked_Face_woclip+BgNoHead+shadow.yaml_"$p"_"$lvl"  --mask /data/mint/DPM_Dataset/MultiPIE/MultiPIE_validset/face_segment/valid/anno/ --batch_size 50 --face_part faceseg_face_noears --postfix ours_128_"$p"_"$lvl"_validset_n200 --out_score_dir /data/mint/robustness_for_evaluation/log=Masked_Face_woclip+BgNoHead+shadow_cfg=Masked_Face_woclip+BgNoHead+shadow.yaml/scores --ds_mask --save_for_dssim /data/mint/robustness_for_evaluation/log=Masked_Face_woclip+BgNoHead+shadow_cfg=Masked_Face_woclip+BgNoHead+shadow.yaml/probust_dssim
        echo "[!] Done : "$p - $lvl
        echo "####################################################################"
    done
done

echo "[!] Finished all : "$pname - $nlvl
                                                                                                                                                                                                                                                                                                                                                                                                                