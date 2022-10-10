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
gpu_id = 2

commands = [
    # # Masked_Face_woclip+Bg
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0 --set {set_} --step 060000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0 --set {set_} --step 070000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0 --set {set_} --step 080000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0 --set {set_} --step 090000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD0 --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1 --set {set_} --step 060000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1 --set {set_} --step 070000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1 --set {set_} --step 080000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1 --set {set_} --step 090000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1.yaml --log_dir Masked_Face_woclip+UNet_Bg_dpm_schedule_WD1e-1 --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
]                                                                                                  

with open(f'./gpu_{gpu_id}_status.txt', mode='w') as f:
    for cmd in commands:
        print(f"[#]Running : {cmd}\n")
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()
