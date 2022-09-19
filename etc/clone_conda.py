# Always Run at /home/mint/

import subprocess
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--conda_name', nargs='+')
args = parser.parse_args()

if __name__ == '__main__':
    conda_ssh_dir = './miniconda_ssh/'
    assert os.getcwd() == '/home/mint'
    os.makedirs(conda_ssh_dir, exist_ok=True)
    
    # Load miniconda if not exists
    if not os.path.isdir(f'/home/mint/miniconda3'):
        print("[#] Install conda/miniconda...")
        download_conda_cmd = 'wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
        processes = subprocess.run(download_conda_cmd.split(' '))
        install_conda_cmd = 'bash ./Miniconda3-latest-Linux-x86_64.sh'
        processes = subprocess.run(install_conda_cmd.split(' '))
    else: print("[#] Found conda/miniconda...")
    
    if len(os.listdir(f'/home/mint/{conda_ssh_dir}/')) == 0:
        print("[#] sshfs the miniconda3 from v9(mothership)")
        ssh_cmd = f'sshfs mint@10.204.100.119:/home/mint/miniconda3 {conda_ssh_dir}'
        processes = subprocess.run(ssh_cmd.split(' '))
    else:
        raise Exception("Folder is not empty!")

    for c in args.conda_name:
        if os.path.isdir(f'{conda_ssh_dir}/envs/{c}'):
            print("[#] Cloning the conda envs : {c}")
            clone_cmd = f'conda create --name {c} --clone {conda_ssh_dir}/envs/{c}'
            processes = subprocess.run(clone_cmd.split(' '))
        else:
            raise FileNotFoundError(f"Conda name : {c} is not exists")
    