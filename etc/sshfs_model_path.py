import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=int, nargs='+')
args = parser.parse_args()

print("[#] Auto sshfs to ...", [f"v{id}" for id in args.vid])

for id in args.vid:
    print(f"========== V{id} (mint@10.204.100.1{int(id+10)}) ==========")
    print(f"[#] umount v{id}", end='')
    umount = f"sudo umount v{id}"
    os.system(umount)
    print("... Done!")
    
    print("[#] mounting v{id}", end='')
    cmd = f"sshfs mint@10.204.100.1{int(id+10)}:/data/mint/model_logs/ ./v{id}"
    os.system(cmd)
    print("... Done!")

