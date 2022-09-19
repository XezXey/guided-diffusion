import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=int, nargs='+', required=True)
parser.add_argument('--folder', type=str, required=True)
args = parser.parse_args()

assert os.path.isabs(args.folder)
print("[#] Auto sshfs to ...", [f"v{id}" for id in args.vid])

for id in args.vid:
    print(f"========== V{id} (mint@10.204.100.1{int(id+10)} / (mint@10.0.0.{int(id+10)}) ==========")
    print(f"[#] umount v{id}", end='')
    
    mount_path = f'{args.folder}/v{id}'
    umount = f"sudo umount {mount_path}"
    os.system(umount)
    print("... Done!")

    if not os.path.exists(mount_path):
        os.makedirs(mount_path)
        
    print(f"[#] mounting v{id}", end='')
    cmd = f"sshfs mint@10.0.0.{int(id+10)}:/home/mint/guided-diffusion/tb_logs/ {mount_path}"
    os.system(cmd)
    print("... Done!")

