import numpy as np
import glob, json
import argparse
import os, time, tqdm, datetime
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=int, nargs='+', default=[])
parser.add_argument('--local', action='store_true', default=False)
parser.add_argument('--mount_dir', required=True)
parser.add_argument('--dataset_dir', required=True)
parser.add_argument('--do_mount', action='store_true', default=False)
parser.add_argument('--curr_vid', type=int, default=9)
# parser.add_argument('--source_sampling_dir', required=True)

args = parser.parse_args()
# Generate dataset path
name = 'random_target'
model_name = "/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml_step=250/"
misc = "/ema_085000/train/render_face/reverse_sampling"
# genenration_path = f'/data/mint/dataset_generation/{name}/{model_name}/{misc}/*'
# sj_coverage = glob.glob(genenration_path)

def mounting(from_path, to_path):
    assert os.path.isabs(to_path)
    for id in args.vid:
        print(f"========== V{id} (mint@10.204.100.1{id:02d} / mint@10.0.0.{id:02d}) ==========")
        mount_path = f'{to_path}/' + f'v{id:02d}'
        
        if not os.path.exists(mount_path):
            os.makedirs(mount_path, exist_ok=True)
            
        if len(os.listdir(mount_path)) != 0:
            print(f"[#] Umounting : {mount_path}")
            umount = f"sudo umount {mount_path}"
            _ = subprocess.run(umount.split(' '))
            print("... Done!")

        if args.local:
            ip = f"10.0.0.{id:02d}"
        else:
            ip = f"10.204.100.1{id:02d}"
        
        cmd = f"sshfs -o ro mint@{ip}:{from_path} {mount_path}"
        print(f'''
              Mounting...
              from : {from_path}
              to : {to_path}
              ''')
        _ = subprocess.run(cmd.split(' '))
        print("... Done!")
    print("#"*100)
    
    
if __name__ == '__main__':
    # Mounting the dataset from all machines
    if (args.mount_dir is not None) and (args.dataset_dir is not None) and (args.do_mount):
        print(f"[#] Mounting the dataset from: {args.dataset_dir}")
        mounting(from_path=args.dataset_dir, to_path=args.mount_dir + '/mount/')

    # ln -s from the dataset to the symlink dir
    print(f"[#] Linking the mount_dir: ln -s {args.dataset_dir}/ {args.mount_dir}/mount/v9")
    if not os.path.exists(f"{args.mount_dir}/mount/v{args.curr_vid:02d}"):
        os.system(f"ln -s {args.dataset_dir} {args.mount_dir}/mount/v{args.curr_vid:02d}")
    
    t_start = time.time()
    sj_dict_v = {}
    sj_dict_all = {}
    for id in args.vid + [args.curr_vid]:
        sj_dict_v[id] = {}
        sj_folder = glob.iglob(f'{args.mount_dir}/mount/v{id:02d}/{name}/{model_name}/{misc}/*')
        for sj in sj_folder:
            sj_name = sj.split('/')[-1]
            target_name = glob.iglob(f'{args.mount_dir}/mount/v{id:02d}/{name}/{model_name}/{misc}/{sj_name}/*')
            sj_dict_v[id][sj_name] = 1
            if sj_name not in sj_dict_all.keys():
                sj_dict_all[sj_name] = {'count': 1}
                # sj_dict_all[sj_name] = {'count': 1, 'target': [t.split('/')[-1] for t in target_name]}
            else: 
                sj_dict_all[sj_name]['count'] += 1
                # sj_dict_all[sj]['target'].extend([t.split('/')[-1] for t in target_name if t.split('/')[-1] not in sj_dict_all[sj]['target']])
        print(f"[#] V{id:02d} has {len(sj_dict_v[id])} sj...")
            
    print(f"[#] Total time: {time.time() - t_start}")
    print(f"[#] Total sj: {len(sj_dict_all.keys())}")
    
    template_sj = dict.fromkeys([f'src={i}.jpg' for i in range(0, 60000)])
    # print(list(template_sj.keys())[:10])
    # print(list(sj_dict_all.keys())[:10])
    # exit()
    unvisited_sj = set(template_sj.keys()) - set(sj_dict_all.keys())
    print(f"[#] Unvisited sj: {unvisited_sj}")
    print(f"[#] Unvisited sj: {len(unvisited_sj)}")
    
    # Find index of unvisited sj in /home/mint/Dev/DiFaReli/difareli-faster/dataset_generation/sampler/generated_dataset_seed=47.json
    with open('/home/mint/Dev/DiFaReli/difareli-faster/dataset_generation/sampler/generated_dataset_seed=47.json', 'r') as f:
        dat = json.load(f)
        
    # Find pair-id based on unvisited_sj
    found_pairs = []
    found_id = []
    found_pairs_json = {'pair': {}}

    # for pair_id, pair_data in tqdm.tqdm(dat['pair'].items()):
    #     if pair_data["src"] in list(unvisited_sj):
    #         found_pairs.append(pair_id)
    
    for sj in tqdm.tqdm(list(unvisited_sj)):
        for pair_id, pair_data in dat['pair'].items():
            if f'src={pair_data["src"]}' == sj:
                found_pairs.append(pair_id)
                found_id.append(pair_data["src"].split('=')[-1].split('.')[0])
                found_pairs_json['pair'][pair_id] = pair_data
                break

    # Print or use the found pair-ids as needed
    print("[#] Found pair-ids:", found_pairs)
    print("[#] Found pair-ids:", len(found_pairs))
    found_pairs_json['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'./fill_sj_generated_dataset_seed=47.json', 'w') as f:
        json.dump(found_pairs_json, f, indent=4)
        
    
    
    
    # Just get the first pair of each sj to fulfill the coverage
    
    