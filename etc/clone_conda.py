# Always Run at /home/mint/

import subprocess
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--conda_name')
args = parser.parse_args()

if __name__ == '__main__':
    conda_ssh_dir = './miniconda_ssh/'
    assert os.getcwd() == '/home/mint'
    os.makedirs(conda_ssh_dir, exist_ok=True)
    
    if len(os.listdir(f'/home/mint/{conda_ssh_dir}/')) == 0:
        ssh_cmd = f'sshfs mint@10.204.100.119:/home/mint/miniconda3 {conda_ssh_dir}'
        processes = subprocess.run(ssh_cmd.split(' '))
    else:
        raise Exception("Folder is not empty!")

    if os.path.isdir(f'{conda_ssh_dir}/envs/{args.conda_name}'):
        clone_cmd = f'conda create --name {args.conda_name} --clone {conda_ssh_dir}/envs/{args.conda_name}'
        processes = subprocess.run(clone_cmd.split(' '))
    else:
        raise FileNotFoundError(f"Conda name : {args.conda_name} is not exists")
    