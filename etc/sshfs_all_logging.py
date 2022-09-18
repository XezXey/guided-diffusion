import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=int, nargs='+', required=True)
parser.add_argument('--lan', action='store_true', default=False)
parser.add_argument('--tb_dir', type=str, required=True)
parser.add_argument('--model_dir', type=str, required=True)
args = parser.parse_args()

assert os.path.isabs(args.model_dir)
assert os.path.isabs(args.tb_dir)
print("[#] Auto sshfs to ...", [f"v{id}" for id in args.vid])

for id in args.vid:
    print(f"========== V{id} (mint@10.204.100.1{int(id+10)} / (mint@10.0.0.{int(id+10)}) ==========")
    print(f"[#] umount v{id}", end='')
    umount = f"sudo umount {args.tb_dir}/v{id}"
    os.system(umount)
    umount = f"sudo umount {args.model_dir}/v{id}"
    os.system(umount)
    print("... Done!")

    if args.lan:
        ip = f"10.0.0.{int(id+10)}"
    else:
        ip = f"10.204.100.{int(id+10)}"
    print(f"[# Tblogs] mounting v{id}", end='')
    cmd = f"sshfs mint@{ip}:/home/mint/guided-diffusion/tb_logs/ {args.tb_dir}/v{id}"
    os.system(cmd)

    print(f"[# Model] mounting v{id}", end='')
    cmd = f"sshfs mint@{ip}:/data/mint/model_logs/ {args.model_dir}/v{id}"
    os.system(cmd)
    print("... Done!")

