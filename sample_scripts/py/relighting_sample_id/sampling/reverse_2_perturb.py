import subprocess

n_diffusion = 1000
n_subject = -1
sampling = "reverse_sampling"
sample_pair_json = "./hard_samples.json"
sample_pair_mode = "pair"
interpolate = "spatial_latent"
interpolate_fn = "slerp"
interpolate_step = 5
set_ = "valid"
out_dir = "/data/mint/sampling/neg1/"
perturb_img_cond = "perturb_img_cond"
perturb_mode = "neg1"
perturb_where = "faceseg_bg_noface&nohair"
gpu_id = 2


commands = [
    # # Face_woclip+Bg+Eyes_dilate5
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg+Eyes_dilate5.yaml --log_dir Face_woclip+Bg+Eyes_dilate5 --{perturb_img_cond} --perturb_mode {perturb_mode} --perturb_where {perturb_where}  --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg+Eyes_dilate5.yaml --log_dir Face_woclip+Bg+Eyes_dilate5 --{perturb_img_cond} --perturb_mode {perturb_mode} --perturb_where {perturb_where}  --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg+Eyes_dilate5.yaml --log_dir Face_woclip+Bg+Eyes_dilate5 --{perturb_img_cond} --perturb_mode {perturb_mode} --perturb_where {perturb_where}  --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # # Face_woclip+Bg_dilate5+Eyes_dilate5
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg_dilate5+Eyes_dilate5.yaml --log_dir Face_woclip+Bg_dilate5+Eyes_dilate5 --{perturb_img_cond} --perturb_mode {perturb_mode} --perturb_where {perturb_where}  --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg_dilate5+Eyes_dilate5.yaml --log_dir Face_woclip+Bg_dilate5+Eyes_dilate5 --{perturb_img_cond} --perturb_mode {perturb_mode} --perturb_where {perturb_where}  --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg_dilate5+Eyes_dilate5.yaml --log_dir Face_woclip+Bg_dilate5+Eyes_dilate5 --{perturb_img_cond} --perturb_mode {perturb_mode} --perturb_where {perturb_where}  --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
]                                                                                                  

with open('./gpu_0_status.txt', mode='w') as f:
    for cmd in commands:
        print(f"[#]Running : {cmd}\n")
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()