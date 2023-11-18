import os, argparse, glob, sys

parser = argparse.ArgumentParser()
parser.add_argument('--save_images_folder', type=str, required=True)
parser.add_argument('--set', type=str, required=True)
parser.add_argument('--params_prefix', type=str, default=None)
parser.add_argument('--save_params_folder', type=str, required=True)
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--index', type=int, nargs='+', default=None)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--estimation_script_folder', type=str, default="/home/mint/Dev/DiFaReli/difareli-faster/preprocess_scripts/Relighting_preprocessing_tools/DECA/script/")
args = parser.parse_args()

if args.params_prefix is None:
    args.params_prefix = args.set


if '/rendered_images/deca_masked_face_images/' not in args.save_images_folder:
    args.save_images_folder += '/rendered_images/deca_masked_face_images/'
if f'/params/{args.set}' not in args.save_params_folder:
    args.save_params_folder += f'/params/{args.set}/'
if not os.path.exists(args.estimation_script_folder):
    print("[#] Estimation script folder does not exist")
    sys.exit(1)
    
s, e = args.index
if s > e:
    print("[#] Start index must be less than end index")
    sys.exit(1)
if s < 0:
    print("[#] Start index must be greater than 0")
    sys.exit(1)
    
print("[#] Re-estimating DECA parameters for DPM dataset")
curdir = os.getcwd()
print("[#} Current directory : ", curdir)
os.chdir(args.estimation_script_folder)
print("[#] Changed directory : ", os.getcwd())
print("[#] Running the estimation script...")
command = f"""CUDA_VISIBLE_DEVICES={args.gpu_id} /home/mint/miniconda3/envs/dpm_sampling_deca/bin/python ./estimate_deca_for_dpm.py \
            --useTex True \
            --useTemplate False \
            --useAvgCam False \
            --useAvgTform False \
            --set valid \
            --params_prefix valid \
            --save_params_folder {args.save_params_folder} \
            --save_images_folder {args.save_images_folder} \
            --masking_flame \
            --fast_save_params False \
            --index {s} {e} \
            --inputpath {args.input_path}"""
            
os.system(command)
os.chdir(curdir)

# command = f"CUDA_VISIBLE_DEVICES=2 /home/mint/miniconda3/envs/dpm_sampling_deca/bin/python ./estimate_deca_for_dpm.py 
#             --useTex True 
#             --useTemplate False 
#             --useAvgCam False 
#             --useAvgTform False 
#             --set valid 
#             --params_prefix valid 
#             --save_params_folder /data/mint/DPM_Dataset/generated_dataset_80perc_deca_estim/params/valid 
#             --save_images_folder /data/mint/DPM_Dataset/generated_dataset_80perc_deca_estim/rendered_images/deca_masked_face_images 
#             --masking_flame 
#             --fast_save_params True 
#             --index 15000 30000
#             --inputpath /data/mint/DPM_Dataset/generated_dataset_80perc/gen_images/valid/"