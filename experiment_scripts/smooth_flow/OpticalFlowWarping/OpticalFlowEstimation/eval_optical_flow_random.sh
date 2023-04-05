#!/bin/bash
# Reconfiguration
declare -a VID_LIST=("penn_action-2278"
                    "insta_variety-instagolf"
                    "insta_variety-javelinthrow"
                    "insta_variety-penaltykick"
                    "insta_variety-tabletennis"
                    "penn_action-0191"
                    "penn_action-0910"
                    "penn_action-2278"
                    "table_tennis_cropped"
                    "CR7_Dribbling_1"
                    "CR7_Dribbling_2"
                    "Squash")
N_SAMPLES_CONFIG=35
N_SAMPLES_EVAL_WARP=25

for EACH_VIDEO in "${VID_LIST[@]}"
  do
    echo "[*]Processing...${EACH_VIDEO}"
    echo "[*]Generating random configs/random warping"
    python generate_random_eval.py --image_dir ./Data/${EACH_VIDEO}/imagesFrame --n_samples ${N_SAMPLES_EVAL_WARP} --step_size_limit 6
    for ((i=1; i<=${N_SAMPLES_CONFIG};i++)) 
      do
        echo "[*] Round ${i} : Generating an Opticalflow config..."
        python get_config_flow.py --config_opticalflow_path ./config_opticalflow.ini --evaluation_opticalflow_path ./evaluation_opticalflow.ini --image_dir ${EACH_VIDEO} --mode random
        # Run the optical flow estimation
        echo "[*]Estimating the flow..."
        python optical_flow_estimation.py --image_dir ./Data/${EACH_VIDEO}/imagesFrame --output_savepath ./Data/${EACH_VIDEO}/ --video_name ${EACH_VIDEO} --config_opticalflow_path ./config_opticalflow.ini
        echo "[*]Params evaluation..."
        # Run the params evaluation
        python optical_flow_params_evaluation.py --evaluation_opticalflow_path ./evaluation_opticalflow.ini --mode random
      done
  done
