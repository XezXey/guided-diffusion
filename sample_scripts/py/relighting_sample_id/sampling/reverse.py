import subprocess

n_diffusion = 1000
n_subject = -1
sampling = "reverse_sampling"
sample_pair_json = "./hard_samples.json"
sample_pair_mode = "pair"
interpolate = "render_face"
interpolate_fn = "slerp"
interpolate_step = 5
set_ = "valid"
out_dir = "/data/mint/sampling/Add_Bg_exp/"
gpu_id = 0

commands = [
    "python relight.py --dataset ffhq --set valid --step 110000 --out_dir /data/mint/sampling/relight --cfg_name Masked_Face_woclip+UNet_Bg_share_dpm_noise_masking_shadow_256.yaml --log_dir Masked_Face_woclip+UNet_Bg_share_dpm_noise_masking_shadow_256_cont2 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/hard_samples.json --sample_pair_mode pair --itp render_face --itp_step 3 --n_subject -1 --gpu_id 2 --slerp",
    "python relight.py --dataset ffhq --set valid --step 100000 --out_dir /data/mint/sampling/relight --cfg_name Masked_Face_woclip+UNet_Bg_share_dpm_noise_masking_shadow_256.yaml --log_dir Masked_Face_woclip+UNet_Bg_share_dpm_noise_masking_shadow_256_cont2 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/hard_samples.json --sample_pair_mode pair --itp render_face --itp_step 3 --n_subject -1 --gpu_id 2 --slerp",
    "python relight.py --dataset ffhq --set valid --step 090000 --out_dir /data/mint/sampling/relight --cfg_name Masked_Face_woclip+UNet_Bg_share_dpm_noise_masking_shadow_256.yaml --log_dir Masked_Face_woclip+UNet_Bg_share_dpm_noise_masking_shadow_256_cont2 --diffusion_steps 1000 --seed 47 --sample_pair_json ./sample_json/hard_samples.json --sample_pair_mode pair --itp render_face --itp_step 3 --n_subject -1 --gpu_id 2 --slerp"

]                                                                                                  

with open(f'./gpu_{gpu_id}_status.txt', mode='w') as f:
    for cmd in commands:
        print(f"[#]Running : {cmd}\n")
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()
