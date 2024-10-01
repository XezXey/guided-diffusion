import numpy as np
import os

if __name__ == '__main__':
    
    # DiFaReli's 1000 steps
    src_path = '/data/mint/sampling/infinite/'
    src_folder = 'log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml'
    
    
    dst_path = '/data/mint/sampling/TPAMI/main_result/ffhq/'
    dst_folder = 'log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml'
    
    
    os.makedirs(f"{dst_path}/{dst_folder}/", exist_ok=True)
    # SSHFS with read-only mode
    to_p = f"{dst_path}/{dst_folder}/"
    from_p = f"mint@10.204.100.109:/{src_path}/{src_folder}/"
    
    os.system(f"sshfs -o ro {from_p} {to_p}")
    
    # DiFaReli's 250 steps
    src_path = '/data/mint/sampling/TPAMI/main_result/ffhq/'
    src_folder = 'log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml_step=250'
    
    
    dst_path = '/data/mint/sampling/TPAMI/main_result/ffhq/'
    dst_folder = 'log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml_step=250'
    
    
    os.makedirs(f"{dst_path}/{dst_folder}/", exist_ok=True)
    # SSHFS with read-only mode
    to_p = f"{dst_path}/{dst_folder}/"
    from_p = f"mint@10.204.100.109:/{src_path}/{src_folder}/"
    
    os.system(f"sshfs -o ro {from_p} {to_p}")
    
    

    
    
