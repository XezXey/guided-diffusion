import sys, tqdm
import json, os, glob, re, time
from multiprocessing.pool import Pool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str, required=True)
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--sample_file', type=str, required=True)
args = parser.parse_args()

# Generate dataset path
name = 'random_target'
model_name = "/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml_step=250/"
misc = "/ema_085000/train/render_face/reverse_sampling"
genenration_path = f'/data/mint/dataset_generation/{name}/{model_name}/{misc}/*/*'

out_path = args.out_path
set_ = 'train'
deca_clip = f"{out_path}/rendered_images/deca_masked_face_images_wclip/{set_}/"
deca_noclip = f"{out_path}/rendered_images/deca_masked_face_images_woclip/{set_}/"
gen_images = f'{out_path}/images/{set_}/'

for fold in [deca_clip, deca_noclip, gen_images]:
    os.makedirs(fold, exist_ok=True)

with open(args.sample_file) as f:
    gen_data = json.load(f)
    
pattern = r'src=([^\s/]+).*?dst=([^\s/]+)'

def create_copy_or_symlink(each_gen_path):
    complete_count = 0
    incomplete_count = 0
    input_images_count = 0
    relit_images_count = 0
    source_coverage = {}
    if args.mode == 'symlink':
        pre_cmd = 'ln -s'
    elif args.mode == 'copy':
        pre_cmd = 'cp'
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented")
    matches = re.findall(pattern, each_gen_path)

    # Print the results
    if matches:
        src_id, dst_id = matches[0]
        src_id = src_id.split('.')[0]
        dst_id = dst_id.split('.')[0]
        # print(f"src: {src_id}")
        # print(f"dst: {dst_id}")
    else:
        print("[#] No match found on : ", each_gen_path)
        return 0, 0, {}
        
        
    each_gen_path = f'{each_gen_path}/Lerp_1000/n_frames=2/'
    try:
        fn = f'{src_id}_{dst_id}'
        # Input
        if not os.path.exists(f'{gen_images}/{src_id}_input.png'):
            os.system(f"{pre_cmd} {each_gen_path}/res_frame0.png {gen_images}/{src_id}_input.png")
            input_images_count += 1
        # if not os.path.exists(f'{deca_clip}/{src_id}_input.png'):
        #     os.system(f"{pre_cmd} {each_gen_path}/ren_frame0.png {deca_clip}/{src_id}_input.png")
        if not os.path.exists(f'{deca_noclip}/{src_id}_input.npy'):
            os.system(f"{pre_cmd} {each_gen_path}/ren_frame0.npy {deca_noclip}/{src_id}_input.npy")
            
            
        # Relit
        if not os.path.exists(f'{gen_images}/{fn}_relit.png'):
            os.system(f"{pre_cmd} {each_gen_path}/res_frame1.png {gen_images}/{fn}_relit.png")
            relit_images_count += 1
        # if not os.path.exists(f'{deca_clip}/{fn}_relit.png'):
        #     os.system(f"{pre_cmd} {each_gen_path}/ren_frame1.png {deca_clip}/{fn}_relit.png")
        if not os.path.exists(f'{deca_noclip}/{fn}_relit.npy'):
            os.system(f"{pre_cmd} {each_gen_path}/ren_frame1.npy {deca_noclip}/{fn}_relit.npy")
            
        complete_count += 1
        source_coverage[src_id] = 1
        
    except: 
        incomplete_count += 1
    return complete_count, incomplete_count, source_coverage, input_images_count, relit_images_count


if __name__ == '__main__':
    # print(len(glob.glob(genenration_path, recursive=True)))
    # print(glob.glob(genenration_path, recursive=True)[:10])
    pool = Pool(processes=24)
    print(f"[#] Starting {args.mode} the dataset...")
    print("[#] Dataset path : ", genenration_path)
    print("[#] #N pairs : ", len(glob.glob(genenration_path, recursive=True)))
    print(f"[#] {args.mode}ing to {out_path}")
          
    s = time.time()
    out = pool.map(create_copy_or_symlink, glob.glob(genenration_path, recursive=True))
    
    complete_count = 0
    incomplete_count = 0
    input_images_count = 0
    relit_images_count = 0
    source_coverage = {}
    for f in tqdm.tqdm(out):
        complete_count += f[0]
        incomplete_count += f[1]
        input_images_count += f[3]
        relit_images_count += f[4]
        k = list(f[2].keys())[0]
        if k in list(source_coverage.keys()):
            source_coverage[k] += 1
        else:
            source_coverage.update(f[2])
    
    print(f"[#] Finishing {args.mode} the dataset in {time.time() - s} seconds...")
    print(f"[#] Sample file : {args.sample_file}")
    print(f"[#] Complete : {complete_count} images")
    print(f"[#] Incomplete : {incomplete_count} images")
    print(f"[#] Subject coverage : {len(source_coverage)}")
    print(f"[#] Input images: ", input_images_count)
    print(f"[#] Relit images: ", relit_images_count)
    print(f"[#] #N (Input + Relit) images: ", input_images_count+relit_images_count)
