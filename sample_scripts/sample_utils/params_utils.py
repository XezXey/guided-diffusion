import numpy as np
import pandas as pd
import torch as th
import glob, os, sys
from collections import defaultdict
from . import file_utils

def params_to_model(shape, exp, pose, cam, i, uvdn=None):

    from model_3d.FLAME import FLAME
    from model_3d.FLAME.config import cfg as flame_cfg
    from model_3d.FLAME.utils.renderer import SRenderY
    import model_3d.FLAME.utils.util as util

    flame = FLAME.FLAME(flame_cfg.model).cuda()
    verts, landmarks2d, landmarks3d = flame(shape_params=shape, 
            expression_params=exp, 
            pose_params=pose)
    renderer = SRenderY(image_size=256, obj_filename=flame_cfg.model.topology_path, uv_size=flame_cfg.model.uv_size).cuda()

    ## projection
    landmarks2d = util.batch_orth_proj(landmarks2d, cam)[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
    landmarks3d = util.batch_orth_proj(landmarks3d, cam); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:] #; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
    trans_verts = util.batch_orth_proj(verts, cam); trans_verts[:,:,1:] = -trans_verts[:,:,1:]

    ## rendering
    shape_images = renderer.render_shape(verts, trans_verts)

    # opdict = {'verts' : verts,}
    # os.makedirs('./rendered_obj', exist_ok=True)
    # save_obj(renderer=renderer, filename=(f'./rendered_obj/{i}.obj'), opdict=opdict)
    
    return {"shape_images":shape_images, "landmarks2d":landmarks2d, "landmarks3d":landmarks3d}

def save_obj(renderer, filename, opdict):
    '''
    vertices: [nv, 3], tensor
    texture: [3, h, w], tensor
    '''
    import model_3d.FLAME.utils.util as util
    i = 0
    vertices = opdict['verts'][i].cpu().numpy()
    faces = renderer.faces[0].cpu().numpy()
    colors = np.ones(shape=vertices.shape) * 127.5

    # save coarse mesh
    util.write_obj(filename, vertices, faces, colors=colors)

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

def normalize(arr, min_val=None, max_val=None, a=-1, b=1):
    '''
    Normalize any vars to [a, b]
    :param a: new minimum value
    :param b: new maximum value
    :param arr: np.array shape=(N, #params_dim) e.g. deca's params_dim = 159
    ref : https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    '''
    if max_val is None and min_val is None:
        max_val = np.max(arr, axis=0)    
        min_val = np.min(arr, axis=0)

    arr_norm = ((b-a) * (arr - min_val) / (max_val - min_val)) + a
    return arr_norm, min_val, max_val

def denormalize(arr_norm, min_val, max_val, a=-1, b=1):
    arr_denorm = (((arr_norm - a) * (max_val - min_val)) / (b - a)) + min_val
    return arr_denorm

def load_params(path, params_key):
    '''
    Load & Return the params
    Input : 
    :params path: path of the pre-computed parameters
    :params params_key: list of parameters name e.g. ['pose', 'light']
    Return :
    :params params_s: the dict-like of {'0.jpg':}
    '''

    anno_path = glob.glob(f'{path}/*.txt')
    params = {}
    for k in params_key:
        for p in anno_path:
            # Params
            if k in p:
                print(f'Key=> {k} : Filename=>{p}')
                params[k] = read_params(path=p)

    params_s = swap_key(params)

    all_params = []
    for img_name in params_s:
        each_img = []
        for k in params_key:
            each_img.append(params_s[img_name][k])
        all_params.append(np.concatenate(each_img))
    all_params = np.stack(all_params, axis=0)
    return params_s, all_params
    
def get_params_set(set, cfg=None):
    if set == 'itw':
        # In-the-wild
        sys.path.insert(0, '../../cond_utils/arcface/')
        sys.path.insert(0, '../../cond_utils/arcface/detector/')
        sys.path.insert(0, '../../cond_utils/deca/')
        from cond_utils.arcface import get_arcface_emb
        from cond_utils.deca import get_deca_emb

        itw_path = "../../itw_images/aligned/"
        device = 'cuda:0'
        # ArcFace
        faceemb_itw, emb = get_arcface_emb.get_arcface_emb(img_path=itw_path, device=device)

        # DECA
        deca_itw = get_deca_emb.get_deca_emb(img_path=itw_path, device=device)

        assert deca_itw.keys() == faceemb_itw.keys()
        params_itw = {}
        for img_name in deca_itw.keys():
            params_itw[img_name] = deca_itw[img_name]
            params_itw[img_name].update(faceemb_itw[img_name])
            
        params_set = params_itw
            
    elif set == 'valid' or set == 'train':
        # Load params
        # params_key = cfg.param_model.params_selector
        params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']

        if set == 'train':
            params_train, params_train_arr = load_params(path="/data/mint/ffhq_256_with_anno/params/train/", params_key=params_key)
            params_set = params_train
        elif set == 'valid':
            params_valid, params_valid_arr = load_params(path="/data/mint/ffhq_256_with_anno/params/valid/", params_key=params_key)
            params_set = params_valid
        else:
            raise NotImplementedError

    else: raise NotImplementedError

    return params_set