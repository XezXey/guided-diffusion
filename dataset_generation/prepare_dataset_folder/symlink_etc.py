import numpy as np
import os, glob

dataset_path = '/data/mint/DPM_Dataset/Generated_Dataset/'
source_data = '/data/mint/DPM_Dataset/ffhq_256_with_anno/'

print(f"[#] Symlinking data from {source_data} to {dataset_path} (if not already symlinked)")
# Symlink the face segment

def symlink(folder):
    print("="*150)
    print(f"[#] Symlinking \"{folder}\" from {source_data} to {dataset_path}")
    if not os.path.exists(f'{dataset_path}/face_segment'):
        os.system(f'ln -s {source_data}/{folder} {dataset_path}/{folder}')
        print("[#] Symlinking done on \"{folder}\"")
    else:
        print(f"[#] \"{folder}\" already symlinked")


if __name__ == "__main__":
    symlink('face_segment')
    symlink('params')
    
    print("="*150)
    print("[#] Symlinking done")

