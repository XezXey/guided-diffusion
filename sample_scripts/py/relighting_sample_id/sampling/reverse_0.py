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
    "python auto_sampling_rev_itp_Enc_input_sep_reverse.py --set valid --step 040000 --out_dir /data/mint/sampling/mod_bg/ --cfg_name Masked_Face_woclip_Identity_no_nonspatial.yaml --log_dir Masked_Face_woclip_Identity_no_nonspatial.yaml --diffusion_steps 1000 --separate_reverse_sampling --seed 47 --sample_pair_json ./sample_json/62878_modSH_samples.json --sample_pair_mode pair --interpolate render_face_modSH --interpolate_step 4 --n_subject 1 --gpu_id 0 --slerp"
    "python auto_sampling_rev_itp_Enc_input_sep_reverse.py --set valid --step 040000 --out_dir /data/mint/sampling/mod_bg/ --cfg_name Masked_Face_woclip_Identity_no_nonspatial.yaml --log_dir Masked_Face_woclip_Identity_no_nonspatial.yaml --diffusion_steps 1000 --separate_reverse_sampling --seed 47 --sample_pair_json ./sample_json/62878_modSH_samples.json --sample_pair_mode pair --interpolate render_face_modSH --interpolate_step 4 --n_subject 1 --gpu_id 1 --slerp --ovr_img ./ovr_img/62872_black_bg.png --postfix black_bg"
    "python auto_sampling_rev_itp_Enc_input_sep_reverse.py --set valid --step 040000 --out_dir /data/mint/sampling/mod_bg/ --cfg_name Masked_Face_woclip_Identity_no_nonspatial.yaml --log_dir Masked_Face_woclip_Identity_no_nonspatial.yaml --diffusion_steps 1000 --separate_reverse_sampling --seed 47 --sample_pair_json ./sample_json/62878_modSH_samples.json --sample_pair_mode pair --interpolate render_face_modSH --interpolate_step 4 --n_subject 1 --gpu_id 2 --slerp --ovr_img ./ovr_img/62872_grey_bg.png --postfix grey_bg"
]                                                                                                  

with open(f'./gpu_{gpu_id}_status.txt', mode='w') as f:
    for cmd in commands:
        print(f"[#]Running : {cmd}\n")
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()
