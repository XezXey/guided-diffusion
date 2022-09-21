import subprocess

n_diffusion = 1000
n_subject = -1
sampling = "uncond_sampling"
sample_pair_json = "./hard_samples.json"
sample_pair_mode = "pair"
interpolate = "spatial_latent"
interpolate_fn = "slerp"
interpolate_step = 5
set_ = "valid"
out_dir = "/data/mint/sampling/uncond_vs_reverse/"
gpu_id = 3


commands = [
    # # Face_woclip
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip.yaml --log_dir Face_woclip --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip.yaml --log_dir Face_woclip --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip.yaml --log_dir Face_woclip --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip.yaml --log_dir Face_woclip --set {set_} --step 200000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # # Masked_Face_woclip+Bg
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+Bg.yaml --log_dir Masked_Face_woclip+Bg --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # # Face_woclip+Bg
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg.yaml --log_dir Face_woclip+Bg --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg.yaml --log_dir Face_woclip+Bg --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg.yaml --log_dir Face_woclip+Bg --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # # Face_woclip+Bg_dilate5
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg_dilate5.yaml --log_dir Face_woclip+Bg_dilate5 --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg_dilate5.yaml --log_dir Face_woclip+Bg_dilate5 --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg_dilate5.yaml --log_dir Face_woclip+Bg_dilate5 --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # # Face_woclip+Bg+Eyes_dilate5
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg+Eyes_dilate5.yaml --log_dir Face_woclip+Bg+Eyes_dilate5 --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg+Eyes_dilate5.yaml --log_dir Face_woclip+Bg+Eyes_dilate5 --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg+Eyes_dilate5.yaml --log_dir Face_woclip+Bg+Eyes_dilate5 --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # # Face_woclip+Bg_dilate5+Eyes_dilate5
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg_dilate5+Eyes_dilate5.yaml --log_dir Face_woclip+Bg_dilate5+Eyes_dilate5 --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg_dilate5+Eyes_dilate5.yaml --log_dir Face_woclip+Bg_dilate5+Eyes_dilate5 --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Face_woclip+Bg_dilate5+Eyes_dilate5.yaml --log_dir Face_woclip+Bg_dilate5+Eyes_dilate5 --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
]                                                                                                  

with open('./gpu_0_status.txt', mode='w') as f:
    for cmd in commands:
        print(f"[#]Running : {cmd}\n")
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()