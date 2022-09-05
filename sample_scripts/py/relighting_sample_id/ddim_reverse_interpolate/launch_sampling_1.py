import subprocess

commands = [
    # "python auto_sampling_rev_itp_Enc_input.py --set valid --step 300000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./hard_samples --cfg_name UNetCond_Spatial_Concat_Shape.yaml --log_dir UNetCond_Spatial_Concat_Shape --n_subject 2 --slerp --diffusion_steps 1000 --seed 23 --sample_pairs ./hard_samples.json --gpu_id 0"
    # "python auto_sampling_rev_itp_Enc_input.py --set valid --step 500000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./hard_samples --cfg_name UNetCond_Spatial_Concat_Shape.yaml --log_dir UNetCond_Spatial_Concat_Shape --n_subject 2 --slerp --diffusion_steps 1000 --seed 23 --sample_pairs ./hard_samples.json --gpu_id 0"
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 100000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 150000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 200000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 250000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 300000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 350000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 400000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 450000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 500000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 550000 --interpolate spatial_latent --interpolate_step 30 --out_dir ./hard_samples --n_subject 1 --cfg_name UNetCond_Spatial_Hadamart_Tanh_Shape_128.yaml --log_dir UNetCod_Spatial_Hadamart_Tanh_Shape_128 --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 1" ,

]

with open('./gpu_1_status.txt', mode='w') as f:
    for cmd in commands:
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()