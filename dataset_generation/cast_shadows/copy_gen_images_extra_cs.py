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
name = 'cast_shadows'
model_name = "/log=DiFaReli_FsBg_Sdiff_SS_256_V100_cfg=DiFaReli_FsBg_Sdiff_SS_256_V100_inference.yaml_inv_with_sd_ds256_pt1_dstC_extra_rot2_top500c/"
misc = "/ema_085000/train/render_face/reverse_sampling/"
genenration_path = f'/data/mint/dataset_generation/{name}/{model_name}/{misc}/'
ffhq_path = '/data/mint/DPM_Dataset/ffhq_256_with_anno/'

out_path = args.out_path
set_ = 'train'
shadow_maps = f"{out_path}/shadow_diff_SS_with_c_simplified/{set_}/"
gen_images = f'{out_path}/images/{set_}/'

for fold in [gen_images, shadow_maps]:
    os.makedirs(fold, exist_ok=True)

    
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
        

    n_frames = 20    
    each_gen_path = f'{each_gen_path}/Lerp_1000/n_frames={n_frames}/'
    try:
        fn = f'{src_id}_{dst_id}'
        # Input
        if not os.path.exists(f'{gen_images}/{src_id}_input.png'):
            # os.system(f"{pre_cmd} {each_gen_path}/res_frame0.png {gen_images}/{src_id}_input.png")
            os.system(f"{pre_cmd} {ffhq_path}/ffhq_256/{set_}/{src_id}.jpg {gen_images}/{src_id}_input.png")
            input_images_count += 1
        if not os.path.exists(f'{shadow_maps}/{fn}_relit.png'):
            os.system(f"{pre_cmd} {each_gen_path}/shadm_shad_frame0.png {shadow_maps}/{src_id}_input.png")
            
        # Relit
        for i in range(1, n_frames):
            if not os.path.exists(f'{gen_images}/{fn}_f{i}_relit.png'):
                os.system(f"{pre_cmd} {each_gen_path}/res_frame{i}.png {gen_images}/{fn}_f{i}_relit.png")
                relit_images_count += 1
            if not os.path.exists(f'{shadow_maps}/{fn}_f{i}_relit.png'):
                os.system(f"{pre_cmd} {each_gen_path}/shadm_shad_frame{i}.png {shadow_maps}/{fn}_f{i}_relit.png")
            
        complete_count += 1
        source_coverage[src_id] = 1
        
    except: 
        incomplete_count += 1
    return complete_count, incomplete_count, source_coverage, input_images_count, relit_images_count


if __name__ == '__main__':
    with open(args.sample_file) as f:
        gen_data = json.load(f)['pair']
    genenration_pathlist = [f"{genenration_path}/src={each['src']}/dst={each['dst']}/" for _, each in gen_data.items()]
    print(genenration_pathlist)
    # Check all the paths are valid
    assert all([os.path.exists(each) for each in tqdm.tqdm(genenration_pathlist)])
        
    pool = Pool(processes=24)
    print(f"[#] Starting {args.mode} the dataset...")
    print("[#] Dataset path : ", genenration_path)
    print("[#] #N available pairs : ", len(glob.glob(genenration_path + '/*/*', recursive=True)))
    print(f"[#] {args.mode}ing to {out_path}")
          
    s = time.time()
    out = pool.map(create_copy_or_symlink, genenration_pathlist)
    
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
