import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=int, nargs='+', default=[])
parser.add_argument('--local', action='store_true', default=False)
parser.add_argument('--dataset_dir', required=True)
parser.add_argument('--source_sampling_dir', required=True)

args = parser.parse_args()
def mounting(from_path, to_path):
    assert os.path.isabs(to_path)
    for id in args.vid:
        print(f"========== V{id} (mint@10.204.100.1{id:02d} / mint@10.0.0.{id:02d}) ==========")
        mount_path = f'{to_path}/'
        
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
        
        cmd = f"sshfs mint@{ip}:{from_path} {mount_path}"
        print(f'''
              Mounting...
              from : {from_path}
              to : {to_path}
              ''')
        _ = subprocess.run(cmd.split(' '))
        print("... Done!")
    print("#"*100)
    
    
if __name__ == '__main__':
    if len(args.vid) > 0:
        print("[#] Auto sshfs to ...", [f"v{id}" for id in args.vid])
    
    if args.model_dir is not None:
        print(f"[#] Mounting the dataset from: /data/mint/DPM_Dataset/Generated_Dataset_TargetLight")
        mounting(from_path='/data/mint/DPM_Dataset/Generated_Dataset_TargetLight/', to_path=args.dataset_dir)
        print(f"[#] Mounting the dataset from: /data/mint/dataset_generation")
        mounting(from_path='/data/mint/dataset_generation/', to_path=args.source_sampling_dir)