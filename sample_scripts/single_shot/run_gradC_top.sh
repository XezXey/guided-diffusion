#!/bin/bash

# User-defined variables
GPU_ID=$1  # First argument: GPU ID
START=$2   # Second argument: Start index
END=$3     # Third argument: End index

# Loop through the specified index range, incrementing by 2
for ((IDX=$START; IDX<$END; IDX+=1))
do
    IDX_NEXT=$((IDX + 1))
    
    # Ensure IDX_NEXT does not exceed END
    if [ $IDX_NEXT -gt $END ]; then
        break
    fi

    echo "Running with gpu_id=${GPU_ID} and idx=${IDX} ${IDX_NEXT}"

    python ./relight_paired_nodpm_gradC.py \
        --ckpt_selector ema \
        --dataset ffhq_png \
        --set valid \
        --step 300000 \
        --out_dir /data/mint/sampling/TPAMI/main_result/ffhq/Website/fancy_rotate_gradC/ \
        --cfg_name paired+difareli+cs+nodpm+trainset_256.yaml \
        --log_dir backup_paired+difareli+cs+nodpm+trainset_256 \
        --seed 47 \
        --sample_pair_json /home/mint/Dev/DiFaReli/difareli-faster/visualize_scripts/TPAMI/main_results/FFHQ_CastShadows/For_Sel/aj_ake_top_candidates.json \
        --sample_pair_mode pair \
        --itp render_face \
        --itp_step 5 \
        --batch_size 1 \
        --gpu_id ${GPU_ID} \
        --lerp \
        --idx ${IDX} ${IDX_NEXT} \
        --shadow_diff_dir /data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/ \
        --scale_depth 256 \
        --pt_round 1 \
        --postproc_shadow_mask_smooth \
        --save_vid \
        --render_batch_size 100 \
        --postfix fancy \
        --inverse_with_shadow_diff \
        --rasterize_type pytorch3d \
        --fancy_rotate_sh
done
