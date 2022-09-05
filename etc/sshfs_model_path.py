import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=int, nargs='+')
parser.add_argument('--folder', type=str)
args = parser.parse_args()

assert os.path.isabs(args.folder)
print("[#] Auto sshfs to ...", [f"v{id}" for id in args.vid])

for id in args.vid:
    print(f"========== V{id} (mint@10.204.100.1{int(id+10)} / mint@10.0.0.{int(id+10)}) ==========")
    print(f"[#] umount v{id}", end='')
    umount = f"sudo umount {args.folder}/v{id}"
    os.system(umount)
    print("... Done!")

    print(f"[#] mounting v{id}", end='')
    cmd = f"sshfs mint@10.0.0.{int(id+10)}:/data/mint/model_logs/ {args.folder}/v{id}"
    os.system(cmd)
    print("... Done!")

