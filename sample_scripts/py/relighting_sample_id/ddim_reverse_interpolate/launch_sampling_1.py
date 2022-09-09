import subprocess

commands = [
    # Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape
    # "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 050000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    # "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 100000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    # "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 150000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    # "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 200000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 0",
    # Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 050000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 2",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 100000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 2",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 150000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 2",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 200000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape.yaml --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 2",
    # Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 050000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 2",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 100000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 2",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 150000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 2",
    "python ./auto_sampling_rev_itp_Enc_input.py --set valid --step 175000 --interpolate spatial_latent --interpolate_step 5 --out_dir ./test_new --n_subject 3 --cfg_name Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg.yaml --log_dir Spatial_Hadamart_AdaGN_ReduceCh-4g-SiLU_Shape+Bg --slerp --diffusion_steps 1000 --seed 47 --sample_pairs ./hard_samples.json --gpu_id 2",
    
    
]

with open('./gpu_1_status.txt', mode='w') as f:
    for cmd in commands:
        f.write(f"[#]Running : {cmd}\n")
        processes = subprocess.run(cmd.split(' '))
        f.write(f"   [#] Exit code : {processes.returncode}\n")
f.close()