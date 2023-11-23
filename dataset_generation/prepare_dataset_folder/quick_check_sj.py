import numpy as np
import glob
import argparse
import os, time
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
    
    # # rsync to combined all files for checking the coverage, progress, etc.
    # os.makedirs(f'{args.mount_dir}/combined', exist_ok=True)
    # for id in args.vid:
    #     os.system(f"rsync -avL {args.mount_dir}/mount/v{id:02d}/{name} {args.mount_dir}/combined/")