import subprocess

commands = [
    "python auto_sampling_rev_itp.py --set valid --step 100000 --interpolate light --out_dir ./samples --cfg_name cond_img64_by_deca_arcface.yaml --log_dir cond_img64_by_deca_arcface --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --gpu_id 0",
    "python auto_sampling_rev_itp.py --set valid --step 200000 --interpolate light --out_dir ./samples --cfg_name cond_img64_by_deca_arcface.yaml --log_dir cond_img64_by_deca_arcface --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --gpu_id 0",
    "python auto_sampling_rev_itp.py --set valid --step 300000 --interpolate light --out_dir ./samples --cfg_name cond_img64_by_deca_arcface.yaml --log_dir cond_img64_by_deca_arcface --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --gpu_id 0",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 100000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Latent_Shape.yaml --log_dir UNetCond_Latent_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode shape --gpu_id 0",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 200000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Latent_Shape.yaml --log_dir UNetCond_Latent_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode shape --gpu_id 0",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 300000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Latent_Shape.yaml --log_dir UNetCond_Latent_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode shape --gpu_id 0",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 100000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Latent_Template_Shape.yaml --log_dir UNetCond_Latent_Template_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode template_shape --gpu_id 0",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 200000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Latent_Template_Shape.yaml --log_dir UNetCond_Latent_Template_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode template_shape --gpu_id 0",
    "python auto_sampling_rev_itp_Enc_input.py --set valid --step 300000 --interpolate spatial_latent --out_dir ./samples --cfg_name UNetCond_Latent_Template_Shape.yaml --log_dir UNetCond_Latent_Template_Shape --n_subject 20 --lerp --slerp --diffusion_steps 1000 --seed 23 --render_mode template_shape --gpu_id 0",
]

with open('./gpu_0_status.txt', mode='w') as f:
    for cmd in commands:
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()