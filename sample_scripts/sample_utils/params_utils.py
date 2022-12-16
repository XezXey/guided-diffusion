import numpy as np
import pandas as pd
import torch as th
import glob, os, sys
import cv2
from collections import defaultdict

def params_to_model(shape, exp, pose, cam, lights):

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
    shape_images = renderer.render_shape(verts, trans_verts, lights=lights)

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

# def get_R_normals(n_step):
#     src = np.array([0, 0, 2.50])
#     dst = np.array([0, 0, 6.50])
#     rvec = np.linspace(src, dst, n_step)
#     R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
#     R = np.stack(R, axis=0)
#     return R

def get_R_normals(n_step):
    if n_step % 2 == 0:
        fh = sh = n_step//2
    else:
        fh = int(n_step//2)
        sh = fh + 1
        
    src = np.array([0, 6.50, 0])
    # dst = np.array([0, 2.50, 0])
    dst = np.array([0, 1.00, 0])
    rvec_f = np.linspace(src, dst, fh)
    
    src = rvec_f[-1]
    # dst = np.array([0, rvec_f[-1][1], -8.00])
    dst = np.array([0, 2.50, -8.00])
    rvec_s = np.linspace(rvec_f[-1], dst, sh)
    # print(rvec_f.shape, rvec_s.shape)
    rvec = np.concatenate((rvec_f, rvec_s), axis=0)
    # print(rvec_f)
    # print(rvec_s)
    # print(rvec)
    # print(rvec.shape)
    R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
    R = np.stack(R, axis=0)
    return R

def load_flame_mask(part='face'):
    f_mask = np.load('/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/data/FLAME_masks_face-id.pkl', allow_pickle=True, encoding='latin1')
    v_mask = np.load('/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/data/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')
    mask={
        'v_mask':v_mask[part].tolist(),
        'f_mask':f_mask[part].tolist() 
    }
    return mask        

def init_deca(useTex=False, extractTex=True, device='cuda', 
              deca_mode='only_renderer', mask=None, deca_obj=None):
    
    # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
    sys.path.insert(1, '/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/')

    from decalib import deca
    from decalib.utils.config import cfg as deca_cfg
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = 'standard'
    deca_cfg.model.extract_tex = extractTex
    deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    return deca_obj

def render_deca(deca_params, idx, n, render_mode='shape', 
                useTex=False, extractTex=False, device='cuda', 
                avg_dict=None, rotate_normals=False, use_detail=False,
                deca_mode='only_renderer', mask=None, repeat=True,
                deca_obj=None):
    '''
    TODO: Adding the rendering with template shape, might need to load mean of camera/tform
    # Render the deca face image that used to condition the network
    :param deca_params: dict of deca params = {'light': Bx27, 'shape':BX50, ...}
    :param idx: index of data in batch to render
    :param n: n of repeated tensor (For interpolation)
    :param render_mode: render mode = 'shape', 'template_shape'
    :param useTex: render with texture ***Need the codedict['albedo'] data***
    :param extractTex: for deca texture (set by default of deca decoding pipeline)
    :param device: device for 'cuda' or 'cpu'
    '''
    #import warnings
    #warnings.filterwarnings("ignore")
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cond_utils/DECA/')))
    if deca_obj is None:
        # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
        sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/')

        from decalib import deca
        from decalib.utils.config import cfg as deca_cfg
        deca_cfg.model.use_tex = useTex
        deca_cfg.rasterizer_type = 'standard'
        deca_cfg.model.extract_tex = extractTex
        deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    else:
        deca_obj = deca_obj
        
    from decalib.datasets import datasets 
    testdata = datasets.TestData([deca_params['raw_image_path'][0]], iscrop=True, face_detector='fan', sample_step=10)
    if repeat:
        codedict = {'shape':deca_params['shape'][[idx]].repeat(n, 1).to(device).float(),
                    'pose':deca_params['pose'][[idx]].repeat(n, 1).to(device).float(),
                    # 'pose':th.tensor(deca_params['pose']).to(device).float(),
                    'exp':deca_params['exp'][[idx]].repeat(n, 1).to(device).float(),
                    'cam':deca_params['cam'][[idx]].repeat(n, 1).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    # 'tform':deca_params['tform'][[idx]].repeat(n, 1).to(device).reshape(-1, 3, 3).float(),
                    'tform':testdata[idx]['tform'][None].repeat(n, 1, 1).to(device).float(),
                    'images':testdata[idx]['image'].to(device)[None,...].float().repeat(n, 1, 1, 1),
                    'tex':deca_params['albedo'][[idx]].repeat(n, 1).to(device).float(),
                    'detail':deca_params['detail'][[idx]].repeat(n, 1).to(device).float(),
        }
        # print(codedict['pose'])
        # print(codedict['light'])
        # exit()
        original_image = deca_params['raw_image'][[idx]].to(device).float().repeat(n, 1, 1, 1) / 255.0
    else:
        codedict = {'shape':th.tensor(deca_params['shape']).to(device).float(),
                    'pose':th.tensor(deca_params['pose']).to(device).float(),
                    'exp':th.tensor(deca_params['exp']).to(device).float(),
                    'cam':th.tensor(deca_params['cam']).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    # 'tform':th.tensor(deca_params['tform']).to(device).reshape(-1, 3, 3).float(),
                    'tform':testdata[idx]['tform'][None].to(device).float(),
                    'images':th.stack([testdata[i]['image'] for i in range(len(deca_params['raw_image_path']))]).to(device).float(),
                    'tex':th.tensor(deca_params['albedo']).to(device).float(),
                    'detail':(deca_params['detail']).to(device).float(),
        }
        original_image = deca_params['raw_image'].to(device).float() / 255.0
        
    if rotate_normals:
        codedict.update({'R_normals': th.tensor(deca_params['R_normals']).to(device).float()})
        
    if render_mode == 'shape':
        use_template = False
        mean_cam = None
        tform_inv = th.inverse(codedict['tform']).transpose(1,2)
    elif render_mode == 'template_shape':
        use_template = True
        mean_cam = th.tensor(avg_dict['cam'])[None, ...].repeat(n, 1).to(device).float()
        tform = th.tensor(avg_dict['tform'])[None, ...].repeat(n, 1).to(device).reshape(-1, 3, 3).float()
        tform_inv = th.inverse(tform).transpose(1,2)
    else: raise NotImplementedError
    _, orig_visdict = deca_obj.decode(codedict, 
                                  render_orig=True, 
                                  original_image=original_image, 
                                  tform=tform_inv, 
                                  use_template=use_template, 
                                  mean_cam=mean_cam, 
                                  use_detail=use_detail,
                                  rotate_normals=rotate_normals,
                                  )  
    rendered_image = orig_visdict['shape_images']
    return rendered_image, orig_visdict
    


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
    
def get_params_set(set, params_key, path="/data/mint/ffhq_256_with_anno/params/"):
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
        if params_key is None:
            params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']

        if set == 'train':
            params_train, params_train_arr = load_params(path=f"{path}/{set}/", params_key=params_key)
            params_set = params_train
        elif set == 'valid':
            params_valid, params_valid_arr = load_params(path=f"{path}/{set}/", params_key=params_key)
            params_set = params_valid
        else:
            raise NotImplementedError

    else: raise NotImplementedError

    return params_set

def preprocess_cond(deca_params, k, cfg):
    if k != 'light':
        return deca_params
    else:
        num_SH = cfg.relighting.num_SH
        params = deca_params[k]
        params = params.reshape(params.shape[0], 9, 3)
        params = params[:, :num_SH, :]
        # params = params.flatten(start_dim=1)
        params = params.reshape(params.shape[0], -1)
        deca_params = params
        return deca_params
    
def write_params(path, params, keys):
    tmp = {}
    for k in keys:
        if th.is_tensor(params[k]):
            tmp[k] = params[k].cpu().numpy()
        else:
            tmp[k] = params[k]
            
    np.save(file=path, arr=tmp)
        