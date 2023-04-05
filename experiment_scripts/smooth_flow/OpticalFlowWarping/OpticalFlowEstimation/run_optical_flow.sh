#!/bin/bash
# Reconfiguration
VIDEO_NAME="penn_action-2278"
echo "[*]Processing...${VIDEO_NAME}"
echo "[*]Generating an Opticalflow config..."
python get_config_flow.py --config_opticalflow_path ./config_opticalflow.ini --evaluation_opticalflow_path ./evaluation_opticalflow.ini --image_dir_list ${VIDEO_NAME}
# Run the optical flow estimation
echo "[*]Estimating the flow..."
python optical_flow_estimation.py --image_dir ./Data/${VIDEO_NAME}/imagesFrame --output_savepath ./Data/${VIDEO_NAME}/ --video_name ${VIDEO_NAME} --config_opticalflow_path ./config_opticalflow.ini
echo "[*]Params evaluation..."
# Run the params evaluation
python optical_flow_params_evaluation.py --evaluation_opticalflow_path ./evaluation_opticalflow.ini
