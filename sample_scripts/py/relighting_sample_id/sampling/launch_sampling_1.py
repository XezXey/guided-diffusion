import subprocess

n_diffusion = 1000
n_subject = 7
sampling = "uncond_sampling"
sample_pair_json = "./hard_samples.json"
sample_pair_mode = "pair"
interpolate = "spatial_latent"
interpolate_fn = "slerp"
interpolate_step = 5
set_ = "valid"
out_dir = "/data/mint/sampling/uncond_vs_reverse/"
gpu_id = 1


commands = [
    # # UNetCond_masked_wclip
    # f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond_masked_wclip.yaml --log_dir UNetCond_masked_wclip --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond_masked_wclip.yaml --log_dir UNetCond_masked_wclip --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond_masked_wclip.yaml --log_dir UNetCond_masked_wclip --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond_masked_wclip.yaml --log_dir UNetCond_masked_wclip --set {set_} --step 200000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # # UNetCond_masked_woclip
    # f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond_masked_woclip.yaml --log_dir UNetCond_masked_woclip --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond_masked_woclip.yaml --log_dir UNetCond_masked_woclip --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond_masked_woclip.yaml --log_dir UNetCond_masked_woclip --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # UNetCond64_masked_woclip_ears_eyes_dilate5.yaml
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond64_masked_woclip_ears_eyes_dilate5.yaml --log_dir UNetCond64_masked_woclip_ears_eyes_dilate5.yaml --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # UNetCond128_masked_woclip_ears_eyes_dilate5.yaml
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond128_masked_woclip_ears_eyes_dilate5.yaml --log_dir UNetCond128_masked_woclip_ears_eyes_dilate5.yaml --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name UNetCond128_masked_woclip_ears_eyes_dilate5.yaml --log_dir UNetCond128_masked_woclip_ears_eyes_dilate5.yaml --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --set {set_} --step 200000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --set {set_} --step 200000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    # Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg --set {set_} --step 050000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg --set {set_} --step 100000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg --set {set_} --step 150000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
    f"python ./auto_sampling_rev_itp_Enc_input.py --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg --set {set_} --step 200000 --{sampling} --interpolate {interpolate} --interpolate_step {interpolate_step} --out_dir {out_dir} --n_subject {n_subject} --{interpolate_fn} --diffusion_steps {n_diffusion} --sample_pair_mode {sample_pair_mode} --sample_pair_json {sample_pair_json} --gpu_id {gpu_id}",
                                                                                      
]                                                                                                  

with open('./gpu_0_status.txt', mode='w') as f:
    for cmd in commands:
        print(f"[#]Running : {cmd}\n")
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()