import subprocess

commands = [
    # UNetCond_Spatial_Hadamart_Tanh_Shape_WD2e-1
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 050000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_WD2e-1.yaml --log_dir UNetCond_Spatial_Hadamart_Tanh_Shape_WD2e-1 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 150000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_WD2e-1.yaml --log_dir UNetCond_Spatial_Hadamart_Tanh_Shape_WD2e-1 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 200000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_WD2e-1.yaml --log_dir UNetCond_Spatial_Hadamart_Tanh_Shape_WD2e-1 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 250000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_WD2e-1.yaml --log_dir UNetCond_Spatial_Hadamart_Tanh_Shape_WD2e-1 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    # UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1                                                                                                          
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 050000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1.yaml --log_dir UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 100000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1.yaml --log_dir UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 150000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1.yaml --log_dir UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 200000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1.yaml --log_dir UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 250000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1.yaml --log_dir UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_WD2e-1 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",


]

with open('./gpu_0_status.txt', mode='w') as f:
    for cmd in commands:
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()