import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=int, nargs='+', default=[])
parser.add_argument('--local', action='store_true', default=False)
parser.add_argument('--tb_dir', default=None)
parser.add_argument('--model_dir', default=None)
parser.add_argument('--ist_cluster_dir', default=None)
args = parser.parse_args()


def mounting(from_path, to_path):
    assert os.path.isabs(to_path)
    for id in args.vid:
        print(f"========== V{id} (mint@10.204.100.1{int(id+10)} / mint@10.0.0.{int(id+10)}) ==========")
        mount_path = f'{to_path}/v{id}'
        
        if not os.path.exists(mount_path):
            os.makedirs(mount_path, exist_ok=True)
            
        if len(os.listdir(mount_path)) != 0:
            print(f"[#] Umounting : {mount_path}")
            umount = f"sudo umount {mount_path}"
            _ = subprocess.run(umount.split(' '))
            print("... Done!")

        if args.local:
            ip = f"10.0.0.{int(id+10)}"
        else:
            ip = f"10.204.100.1{int(id+10)}"
        
        cmd = f"sshfs mint@{ip}:{from_path} {mount_path}"
        print(f'''
              Mounting...
              from : {from_path}
              to : {to_path}
              ''')
        _ = subprocess.run(cmd.split(' '))
        print("... Done!")
    print("#"*100)

def mounting_ist_cluster(from_path, to_path):
    assert os.path.isabs(to_path)
    mount_path = f'{to_path}/ist-cluster'
    if not os.path.exists(mount_path):
        os.makedirs(mount_path, exist_ok=True)
        
    if len(os.listdir(mount_path)) != 0:
        print(f"[#] Umounting : {mount_path}")
        umount = f"sudo umount {mount_path}"
        _ = subprocess.run(umount.split(' '))
        print("... Done!")
    
    ip = '10.204.100.209'
    cmd = f"sshfs puntawatp@{ip}:{from_path} {mount_path} -o IdentityFile=~/.ssh/v100_vistec_id_rsa"
    print(f'''
            Mounting...
            from : {from_path}
            to : {to_path}
            ''')
    _ = subprocess.run(cmd.split(' '))
    print("#"*100)
    

if __name__ == '__main__':
    if len(args.vid) > 0:
        print("[#] Auto sshfs to ...", [f"v{id}" for id in args.vid])
        
    if args.tb_dir is not None:
        print("[#] Mounting the tensorboard log from \"/guided-diffusion/tb_logs\"")
        mounting(from_path='/home/mint/guided-diffusion/tb_logs/', to_path=args.tb_dir)
    if args.model_dir is not None:
        print("[#] Mounting the model log from \"/model_logs\"")
        mounting(from_path='/data/mint/model_logs/', to_path=args.model_dir)
    if args.ist_cluster_dir is not None:
        print("[#] Mounting the model log from ist-cluster")
        mounting_ist_cluster(from_path="/ist/ist-share/vision/mint/model_logs/", to_path=args.ist_cluster_dir)
        
