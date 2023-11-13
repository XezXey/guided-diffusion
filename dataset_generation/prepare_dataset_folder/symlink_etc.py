import numpy as np
import os, glob

dataset_path = '/data/mint/DPM_Dataset/Generated_Dataset/'
source_data = '/data/mint/DPM_Dataset/ffhq_256_with_anno/'

print(f"[#] Symlinking data from {source_data} to {dataset_path} (if not already symlinked)")
# Symlink the face segment

def symlink(fsrc, fdst=None):
    if fdst is None:
        fdst = fsrc
    print("="*150)
    print(f"[#] Symlinking \"{fsrc}\" from {source_data} to {dataset_path}")
    if not os.path.exists(f'{dataset_path}/{fdst}'):
        os.system(f'ln -s {source_data}/{fsrc} {dataset_path}/{fdst}')
        print(f"[#] Symlinking done on \"{fdst}\"")
    else:
        print(f"[#] \"{fsrc}\" already symlinked")


if __name__ == "__main__":
    # symlink('face_segment')
    symlink('params', 'params_recreate')
    
    print("="*150)
    print("[#] Symlinking done")

