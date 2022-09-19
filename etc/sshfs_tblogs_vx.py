import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=int, nargs='+')
parser.add_argument('--folder', type=str)
parser.add_argument('--local', action='store_true', default=False)
args = parser.parse_args()

assert os.path.isabs(args.folder)
print("[#] Auto sshfs to ...", [f"v{id}" for id in args.vid])

for id in args.vid:
    print(f"========== V{id} (mint@10.204.100.1{int(id+10)} / mint@10.0.0.{int(id+10)}) ==========")
    mount_path = f'{args.folder}/v{id}'
    
    if not os.path.exists(mount_path):
        os.makedirs(mount_path, exist_ok=True)
        
    if len(os.listdir(mount_path)) != 0:
        print(f"[#] Umounting : {mount_path}")
        umount = f"umount {mount_path}"
        processes = subprocess.run(umount.split(' '))
        print("... Done!")

    if args.local:
      ip = f"10.0.0.{int(id+10)}"
    else:
      ip = f"10.204.100.1{int(id+10)}"
      
    cmd = f"sshfs mint@{ip}:{} {mount_path}"
    print(f"Mounting : {cmd}")
    processes = subprocess.run(cmd.split(' '))
    print("... Done!")



