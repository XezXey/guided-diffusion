import pandas as pd
import numpy as np
import glob
import tqdm
from collections import defaultdict

def read_params(path):
    params = pd.read_csv(path, header=None, sep=" ", index_col=False, lineterminator='\n')
    params.rename(columns={0:'img_name'}, inplace=True)
    params = params.set_index('img_name').T.to_dict('list')
    return params

def swap_key(params):
    params_s = defaultdict(dict)
    for params_name, v in params.items():
        for img_name, params_value in v.items():
            params_s[img_name][params_name] = np.array(params_value).astype(np.float64)

    return params_s

def load_deca_params(deca_dir, cfg, norm_shadow_val=False):
    deca_params = {}

    # face params 
    params_key = ['shadow', 'shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'tform', 'albedo', 'detail']
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        params_path = glob.glob(f"{deca_dir}/*{k}-anno.txt")
        for path in params_path:
            deca_params[k] = read_params(path=path)
            if k == 'shadow':
                if norm_shadow_val:
                    print(f"[#] Normalizing the shadow values...")
                    deca_params[k] = process_shadow(deca_params[k], cfg)
        deca_params[k] = preprocess_light(deca_params[k], k, cfg)
    
    avg_dict = avg_deca(deca_params)
    
    deca_params = swap_key(deca_params)
    return deca_params, avg_dict

def process_shadow(shadow_params, cfg):
    max_c = 8.481700287326827 # 7.383497233314015
    min_c = -4.989461058405101 # -4.985533880236826
    for img_name in shadow_params.keys():
        c_val = np.array(shadow_params[img_name])
        c_val = (c_val - min_c) / (max_c - min_c)
        if cfg.param_model.shadow_val.inverse:
            c_val = 1 - c_val
        shadow_params[img_name] = c_val
    return shadow_params

def avg_deca(deca_params):
    
    avg_dict = {}
    for p in deca_params.keys():
        avg_dict[p] = np.stack(list(deca_params[p].values()))
        assert avg_dict[p].shape[0] == len(deca_params[p])
        avg_dict[p] = np.mean(avg_dict[p], axis=0)
    return avg_dict

def preprocess_light(deca_params, k, cfg):
    """
    # Remove the SH component from DECA (This for reduce SH)
    """
    if k != 'light':
        return deca_params
    else:
        num_SH = 27
        for img_name in deca_params.keys():
            params = np.array(deca_params[img_name])
            params = params.reshape(9, 3)
            params = params[:num_SH]
            params = params.flatten()
            deca_params[img_name] = params
        return deca_params