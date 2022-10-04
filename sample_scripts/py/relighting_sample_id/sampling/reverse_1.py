import subprocess

n_diffusion = 1000
n_subject = 3
sampling = "reverse_sampling"
sample_pair_json = "./hard_samples.json"
sample_pair_mode = "pair"
interpolate = "spatial_latent"
interpolate_fn = "slerp"
interpolate_step = 5
set_ = "valid"
# out_dir = "/data/mint/sampling/intermediate2_darkarea/"
out_dir = "/data/mint/sampling/intermediate3/"
gpu_id = 0

commands = [
    # # Masked_Face_woclip+Bg_5e-1
    # f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+Bg_clip5e-1.yaml --log_dir Masked_Face_woclip+Bg_clip5e-1.yaml --set {set_} --step 200000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+Bg.yaml --log_dir Masked_Face_woclip+Bg --set {set_} --step 400000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+Bg.yaml --log_dir Masked_Face_woclip+Bg --set {set_} --step 300000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+Bg.yaml --log_dir Masked_Face_woclip+Bg --set {set_} --step 200000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Masked_Face_woclip+Bg.yaml --log_dir Masked_Face_woclip+Bg --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
]                                                                                                  

with open('./gpu_0_status.txt', mode='w') as f:
    for cmd in commands:
        print(f"[#]Running : {cmd}\n")
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()