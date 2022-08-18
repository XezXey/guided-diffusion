import subprocess

commands = [
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 100000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Spatial_Concat_Shape.yaml --log_dir UNetCond_Spatial_Concat_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode shape --gpu_id 2",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 200000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Spatial_Concat_Shape.yaml --log_dir UNetCond_Spatial_Concat_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode shape --gpu_id 2",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 300000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Spatial_Concat_Shape.yaml --log_dir UNetCond_Spatial_Concat_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode shape --gpu_id 2",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 100000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Spatial_Concat_Template_Shape.yaml --log_dir UNetCond_Spatial_Concat_Template_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode template_shape --gpu_id 2",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 150000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Spatial_Concat_Template_Shape.yaml --log_dir UNetCond_Spatial_Concat_Template_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode template_shape --gpu_id 2",
]

with open('./gpu_2_status.txt', mode='w') as f:
    for cmd in commands:
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()