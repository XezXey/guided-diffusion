# Always Run at /home/mint/

import subprocess
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--conda_name', nargs='+')
parser.add_argument('--gpu_model', type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    conda_ssh_dir = './miniconda_ssh/'
    assert os.getcwd() == '/home/mint'
    os.makedirs(conda_ssh_dir, exist_ok=True)
    
    # Load miniconda if not exists
    if not os.path.isdir(f'/home/mint/miniconda3'):
        print("#" * 100)
        print("[#] Install conda/miniconda...")
        download_conda_cmd = 'wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
        processes = subprocess.run(download_conda_cmd.split(' '))
        install_conda_cmd = 'bash ./Miniconda3-latest-Linux-x86_64.sh'
        processes = subprocess.run(install_conda_cmd.split(' '))
    else: print("[#] Found conda/miniconda...")
    
    if args.gpu_model in ['a4000', 'rtx3090']:
        ip = '10.204.100.122'
    elif args.gpu_model in ['rtx2080']:
        ip = '10.204.100.119'
    else: raise ValueError(f"Unknown gpu model : {args.gpu_model}")
    
    if len(os.listdir(f'/home/mint/{conda_ssh_dir}/')) != 0:
        print(f"[#] Umounting : {conda_ssh_dir}")
        umount_cmd = 'umount {conda_ssh_dir}'
        processes = subprocess.run(umount_cmd.split(' '))
        
    print("#" * 100)
    print("[#] sshfs the miniconda3 from v9 or v12(mothership)")
    ssh_cmd = f'sshfs mint@{ip}:/home/mint/miniconda3 {conda_ssh_dir}'
    processes = subprocess.run(ssh_cmd.split(' '))

    for c in args.conda_name:
        if os.path.isdir(f'{conda_ssh_dir}/envs/{c}'):
            print("#" * 100)
            print(f"[#] Cloning the conda envs : {c}")
            clone_cmd = f'conda create --name {c} --clone {conda_ssh_dir}/envs/{c}'
            processes = subprocess.run(clone_cmd.split(' '))
        else:
            raise FileNotFoundError(f"Conda name : {c} is not exists")
    