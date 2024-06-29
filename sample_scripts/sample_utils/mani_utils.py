from tkinter import W
import numpy as np
import torch as th
import blobfile as bf
import PIL
import vis_utils, img_utils, file_utils
from scipy.spatial.transform import Rotation as R
import itertools

def lerp(r, src, dst):
    return ((1-r) * src) + (r * dst)

def slerp(r, src, dst):
    low = src; high=dst; val=r
    low_norm = low/th.norm(low, dim=1, keepdim=True)
    high_norm = high/th.norm(high, dim=1, keepdim=True)
    omega = th.acos((low_norm*high_norm).sum(1))
    so = th.sin(omega)
    res = (th.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (th.sin(val*omega)/so).unsqueeze(1) * high
    return res

def interchange_cond_img(cond, src_idx, dst_idx, itc_img_key, cfg):
    '''
    Change the condition image with respect to the key
    '''
    for k in itc_img_key:
        assert k in cfg.img_cond_model.in_image
        cond[f'{k}_img'][dst_idx] = cond[f'{k}_img'][src_idx]

    # Re-create cond_img    
    cond_img = []
    for k in cfg.img_cond_model.in_image:
        cond_img.append(cond[f'{k}_img'])
    
    cond['cond_img'] = th.cat(cond_img, dim=1)  # BxCxHxW
    return cond

def iter_interp_cond(cond, src_idx, dst_idx, n_step, interp_set, interp_fn, add_remove_shadow=False, vary_shadow=None, vary_shadow_nobound=None, vary_shadow_range=None, add_shadow_to=None, remove_shadow_to=None):
    '''
    Interpolate the condition following the keys in interp_set
    :params src_idx: the source index of condition
    :params dst_idx: the destination index of condition
    :params n_step: the number of interpolation step
    :params interp_fn: interpolation function e.g. lerp(), slerp()
    :params interp_set: list contains keys of params to be interpolated e.g. ['light', 'shape']
    
    :return interp_cond: interpolated between src->dst in dict-like 
        e.g. {'light': tensor of [n_step x ...], 'shape': tensor of [n_step x ...]}
    '''
    out_interp = {}

    for itp in interp_set:
        assert itp in cond.keys()
        assert src_idx < len(cond[itp]) and dst_idx < len(cond[itp])
        
        if itp == 'shadow':
            if vary_shadow:
                import json
                with open(cond['misc']['args'].sample_pair_json, 'r') as f:
                    sample_pairs = json.load(f)[cond['misc']['args'].sample_pair_mode]
                    pair_idx = list(sample_pairs.keys())[cond['sample_idx']]
                    assert sample_pairs[pair_idx]['src'] == cond['image_name'][src_idx]
                    min_c = sample_pairs[pair_idx]['rmv_shadow_bound']
                    max_c = sample_pairs[pair_idx]['add_shadow_bound']
                    n_to_min = n_step // 2 if n_step % 2 == 0 else n_step // 2 + 1
                    n_to_max = n_step // 2
                    min_c = np.linspace(min_c, cond[itp][src_idx][0], n_to_min)[..., None]
                    max_c = np.linspace(cond[itp][src_idx][0], max_c, n_to_max)[..., None]
                    interp = np.concatenate((max_c[0:1, :], min_c, max_c[1:, :]), axis=0)
                    print(min_c.shape, max_c.shape, interp.shape)
                    print(f"[#] Shadow = {cond[itp][src_idx][0]}")
                    print(f"[#] Varying shadow ({vary_shadow}) with : \n{interp}")
            elif vary_shadow_range:
                assert len(vary_shadow_range) == 2
                min_c, max_c = vary_shadow_range
                min_c = float(min_c)
                max_c = float(max_c)
                n_to_min = n_step // 2 if n_step % 2 == 0 else n_step // 2 + 1
                n_to_max = n_step // 2
                min_c = np.linspace(min_c, cond[itp][src_idx][0], n_to_min)[..., None]
                max_c = np.linspace(cond[itp][src_idx][0], max_c, n_to_max)[..., None]
                interp = np.concatenate((max_c[0:1, :], min_c, max_c[1:, :]), axis=0)   # max_c[0:1, :] is the same as cond[itp][src_idx][0] => for inversion
                print(min_c.shape, max_c.shape, interp.shape)
                print(f"[#] Shadow = {cond[itp][src_idx][0]}")
                print(f"[#] Varying shadow ({vary_shadow}) with : \n{interp}")
            elif vary_shadow_nobound:
                min_c = -10
                max_c = +10
                if n_step % 2 == 0:
                    min_c = np.linspace(min_c, cond[itp][src_idx][0], n_step//2)[..., None]
                    max_c = np.linspace(cond[itp][src_idx][0], max_c, (n_step//2)+1)[..., None]
                else:
                    min_c = np.linspace(min_c, cond[itp][src_idx][0], (n_step//2)+1)[..., None]
                    max_c = np.linspace(cond[itp][src_idx][0], max_c, (n_step//2)+1)[..., None]
                # interp = np.concatenate((max_c[0:1, :], min_c, max_c[1:, :]), axis=0)
                interp = np.concatenate((min_c, max_c[1:, :]), axis=0)
                print(min_c.shape, max_c.shape, interp.shape)
                print(f"[#] Shadow = {cond[itp][src_idx][0]}")
                print(f"[#] Varying shadow ({vary_shadow}) with : \n{interp}")
                
            elif add_remove_shadow:
                if add_shadow_to is not None:
                    max_c = 8.481700287326827
                    interp = np.linspace(cond[itp][src_idx][0], add_shadow_to, n_step)[..., None]
                    print(f"Interpolating shadow with : \n{interp}, Adding shadow = {cond[itp][src_idx][0]} to {add_shadow_to} (default max = {max_c})")
                elif remove_shadow_to is not None:
                    min_c = -4.989461058405101
                    interp = np.linspace(cond[itp][src_idx][0], remove_shadow_to, n_step)[..., None]
                    print(f"Interpolating shadow with : \n{interp}, Removing shadow = {cond[itp][src_idx][0]} to {remove_shadow_to} (default min = {min_c})")
                else: raise NotImplementedError("Please specify the shadow value to be added or removed...")
            else: 
                raise NotImplementedError("With itp = shadow, please specify the add/remove shadow mode (e.g., vary_shadow, vary_shadow_nobound, vary_shadow_range, add_shadow_to, remove_shadow_to)")
        else:
            # print(cond[itp])
            if isinstance(cond[itp], list):
                #NOTE: interpolate the condition (list-type)
                interp = []
                for i in range(len(cond[itp])):
                    assert cond[itp][i][[src_idx]].shape == cond[itp][i][[dst_idx]].shape
                    interp_temp = interp_cond(src_cond=cond[itp][i][[src_idx]],
                                    dst_cond=cond[itp][i][[dst_idx]],
                                    n_step=n_step,
                                    interp_fn=interp_fn)
                    interp.append(interp_temp)
            elif th.is_tensor(cond[itp]) or isinstance(cond[itp], np.ndarray):
                #NOTE: interpolate the condition (tensor-type)
                interp = interp_cond(src_cond=cond[itp][[src_idx]],
                                    dst_cond=cond[itp][[dst_idx]],
                                    n_step=n_step,
                                    interp_fn=interp_fn)
            else: raise NotImplementedError
        out_interp[itp] = interp
        # print(out_interp[itp], itp)
        # exit()
    return out_interp 


def iter_interp_cond_lightfile(cond, src_idx, n_step, itp_src, itp_dst, interp_fn):
    '''
    Interpolate the condition following the keys in interp_set
    :params src_idx: the source index of condition
    :params dst_idx: the destination index of condition
    :params n_step: the number of interpolation step
    :params interp_fn: interpolation function e.g. lerp(), slerp()
    :params interp_set: list contains keys of params to be interpolated e.g. ['light', 'shape']
    
    :return interp_cond: interpolated between src->dst in dict-like 
        e.g. {'light': tensor of [n_step x ...], 'shape': tensor of [n_step x ...]}
    '''
    out_interp = {}

    assert (itp_src in cond.keys()) and (itp_dst in cond.keys())
    if itp_src == 'light':
        if isinstance(cond[itp_src], list):
            #NOTE: interpolate the condition (list-type)
            interp = []
            for i in range(len(cond[itp_src])):
                assert cond[itp_src][i][[src_idx]].shape == cond[itp_dst][i].shape
                interp_temp = interp_cond(src_cond=cond[itp_src][i][[src_idx]],
                                dst_cond=cond[itp_dst][i],
                                n_step=n_step,
                                interp_fn=interp_fn)
                interp.append(interp_temp)
        elif th.is_tensor(cond[itp_src]) or isinstance(cond[itp_src], np.ndarray):
            #NOTE: interpolate the condition (tensor-type)
            interp = interp_cond(src_cond=cond[itp_src][[src_idx]],
                                dst_cond=cond[itp_dst],
                                n_step=n_step,
                                interp_fn=interp_fn)
        else: raise NotImplementedError
    out_interp[itp_src] = interp

    return out_interp 

def interchange_cond(cond, interchange, base_idx, n):
    '''
    Condition parameters interchange
    :params cond: condition parameters in BxD, e.g. D = #shape + #pose
    :params interchange: list of parameters e.g. ['pose'], ['pose', 'shape']
    :params base_idx: base_idx that repeat itself and make change a condition from another sample.
    '''
    
    for p in ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']:
        if p in interchange:
            # Interchange the condition
            pass
        else:
            # Keep the base-idx of condition
            cond[p] = np.repeat(cond[p][[base_idx]], repeats=n, axis=0)
    return cond

def rotate_sh(cond, interp_set, src_idx, dst_idx, n_step):

    import pyshtools as pysh
    def toCoeff(c):
      t = pysh.SHCoeffs.from_zeros(2)
      t.set_coeffs(c[0], 0, 0)
      t.set_coeffs(c[1], 1, 1)
      t.set_coeffs(c[2], 1, -1)
      t.set_coeffs(c[3], 1, 0)
      t.set_coeffs(c[4], 2, -2)
      t.set_coeffs(c[5], 2, 1)
      t.set_coeffs(c[6], 2, -1)
      t.set_coeffs(c[7], 2, 2)
      t.set_coeffs(c[8], 2, 0)
      return t

    def toRGBCoeff(c):
      return [toCoeff(c[::3]), toCoeff(c[1::3]), toCoeff(c[2::3])]

    def toDeca(c):
      a = c.coeffs
      lst = [a[0, 0, 0],
             a[0, 1, 1],
             a[1, 1, 1],
             a[0, 1, 0],
             a[1, 2, 2],
             a[0, 2, 1],
             a[1, 2, 1],
             a[0, 2, 2],
             a[0, 2, 0]]
      return np.array(lst)

    def toRGBDeca(cc):
      return list(itertools.chain(*zip(toDeca(cc[0]), toDeca(cc[1]), toDeca(cc[2]))))

    def axisAngleToEuler(x, y, z, degree):
      xyz = np.array([x, y, z])
      xyz = xyz / np.linalg.norm(xyz)

      rot = R.from_mrp(xyz * np.tan(degree * np.pi / 180 / 4))
      return rot.as_euler('zyz', degrees=True)

    def rotateSH(sh_np, x, y, z, degree):
      cc = toRGBCoeff(sh_np)
      euler = axisAngleToEuler(x, y, z, degree)
      cc[0] = cc[0].rotate(*euler)
      cc[1] = cc[1].rotate(*euler)
      cc[2] = cc[2].rotate(*euler)
      return toRGBDeca(cc)
    
    def genSurfaceNormals(n):
        x = th.linspace(-1, 1, n)
        y = th.linspace(1, -1, n)
        y, x = th.meshgrid(y, x)

        z = (1 - x ** 2 - y ** 2)
        z[z < 0] = 0
        z = th.sqrt(z)
        return th.stack([x, y, z], 0)

    img_size = 128
    xyz = genSurfaceNormals(img_size)
    cx = img_size // 2
    cy = img_size // 2
    v = xyz[:, cy, cx]

    inp_sh = cond['light'][[src_idx]].flatten()   # [1, 27] -> [27,]
    rounds = 5
    n = n_step
    out_sh = []
    centered = rotateSH(inp_sh,    0, 1, 0, np.arcsin(float(v[0])) * 180 / np.pi)
    centered = rotateSH(centered, 1, 0, 0, np.arcsin(float(v[1])) * 180 / np.pi)
    for i in range(n):
        # print(i)
        t = i / n # Fraction of rotation
        tt = t * rounds * 2 * np.pi
        rad = t * 0.9

        x = np.sin(tt) * rad
        y = np.cos(tt) * rad
        
        moved = rotateSH(centered, 0, 1, 0, -np.arcsin(x) * 180 / np.pi)  # Rotate over y axis by -np.arcsin(x) * 180 / np.pi
        moved = rotateSH(moved   , 1, 0, 0, -np.arcsin(y) * 180 / np.pi)  # Rotate over x axis by -np.arcsin(y) * 180 / np.pi
        sh_moved = np.array(moved)
        out_sh.append(sh_moved)

    out_sh = np.stack(out_sh, 0)

    return {'light':out_sh}


def interp_cond(src_cond, dst_cond, n_step, interp_fn):
    '''
    Interpolate the condition
    :params src_cond: the source condition [BxC] ; C = number of condition dimension
    :params dst_cond: the destination condition [BxC] ; C = number of condition dimension
    :params n_step: the number of interpolation step
    :params interp_fn: interpolation function e.g. lerp(), slerp()
    
    :return interp: interpolated between src->dst with same shape of input
    '''
    print(f"[#] Interpolate with {interp_fn}")
    if n_step <= 1:
        return src_cond
    else:
        r_interp = np.linspace(0, 1, num=n_step)

        src = src_cond
        dst = dst_cond
        interp = []
        for r in r_interp:
            tmp = interp_fn(r=r, src=src, dst=dst)
            if th.is_tensor(tmp):
                tmp = tmp.detach().cpu().numpy()
            interp.append(tmp.copy())

        interp = np.concatenate((interp), axis=0)

        return interp 

def interp_by_dir(cond, src_idx, itp_name, direction, n_step):
    step = np.linspace(0, 2, num=n_step)
    src_cond = cond[itp_name][[src_idx]]
    if th.is_tensor(src_cond):
        src_cond = src_cond.detach().cpu().numpy()
    else:
        src_cond = np.array(cond[itp_name][[0]])
    itp = []
    for i in range(n_step):
        tmp = src_cond + step[i] * direction
        itp.append(tmp)

    return {itp_name:np.concatenate(itp, axis=0)}

def interp_noise(src_noise, dst_noise, n_step, interp_fn=lerp):
    '''
    Interpolate the noise
    :params src_cond: the source noise [BxCxHxW]
    :params dst_cond: the destination noise [BxCxHxW]
    :params n_step: the number of interpolation step
    :params interp_fn: interpolation function e.g. lerp(), slerp()
    
    :return interp: interpolated between src->dst with same shape of input
    '''

    r_interp = np.linspace(0, 1, num=n_step)

    src = src_noise
    dst = dst_noise
    interp = []
    for r in r_interp:
        tmp = interp_fn(r=r, src=src, dst=dst)
        interp.append(tmp.clone())

    interp = th.cat((interp), dim=0)

    return interp

def repeat_cond_params(cond, base_idx, n, key):
    print("[#] Repeating cond : ", key)
    repeat = {}
    for p in key:
        if th.is_tensor(cond[p][[base_idx]]):
            rep = cond[p][[base_idx]].cpu().detach().numpy()
        else: rep = cond[p][[base_idx]]
        repeat[p] = np.repeat(rep, repeats=n, axis=0)
    
    return repeat

def create_cond_params(cond, key):
    '''
    Create the cond_params for conditioning the model by concat
    :params cond: condition dict-like e.g. {'light': tensor of [Bx27], 'pose': tensor of [Bx6], ...}
    :params key: key contains parameters name to be used for an input
    
    :return cond: condition dict-like with addition 'cond_params' key that ready to used for inference
    '''
    print("[#] Condition build from parameters in ", key)
    tmp = []
    for p in key:
        if th.is_tensor(cond[p]):
            tmp.append(cond[p].cpu().detach().numpy())
        else:
            tmp.append(cond[p])
    if tmp != []:
        print(np.concatenate(tmp, axis=1).shape)
        cond['cond_params'] = np.concatenate(tmp, axis=1)
    return cond
    
def create_cond_imgs(cond, key):
    '''
    Create the cond_params for conditioning the model by concat
    :params cond: condition dict-like e.g. {'deca_shape_image':BxCxHxW, 'deca_template_shape_image':BxCxHxW}
    :params key: key contains parameters name to be used for an input
    
    :return cond: condition dict-like with addition 'cond_params' key that ready to used for inference
    '''
    print("[#] Condition build from image(s) in ", key)
    tmp = []
    for p in key:
        if th.is_tensor(cond[p]):
            tmp.append(cond[p].cpu())
        else:
            tmp.append(cond[p])
    print(np.concatenate(tmp, axis=1).shape)
    cond['cond_img'] = np.concatenate(tmp, axis=1)
    return cond

def perturb_img(cond, key, p_where, p_mode):
    print(f"[#] Perturbing images condition at {p_where} with {p_mode}")

    def perturb_mode(x, p_mode):
        if p_mode == 'zero':
            return x * 0
        elif p_mode == 'neg1':
            return (x * 0) - 1
        elif p_mode == 'rand':
            return th.FloatTensor(size=x.shape).uniform_(-1, 1)

    for p in key:
        if p in p_where:
            cond_perturb = perturb_mode(cond[f'{p}_img'], p_mode)
            cond[p] = cond_perturb
        else: 
            cond[p] = cond[f'{p}_img']
    return cond
    
def modify_cond(mod_idx, cond_params, params_loc, params_sel, n_step, bound, mod_cond, force_zero=False):
    '''
    Manually change/scale the condition parameters at i-th index e.g. [c1, c2, c3, ..., cN] => [c1 * 2.0, c2, c3, ..., cN]
    :params offset: offset to +- from condition

    '''
    # Fixed the based-idx image
    mod_interp = np.linspace(-bound, bound, num=n_step)
    mod_interp = np.stack([mod_interp]*len(mod_idx), axis=-1)
    mod_interp = th.tensor(mod_interp).cuda()
    params_selected_loc = params_loc
    params_selector = params_sel

    final_cond = cond_params.clone().repeat(n_step, 1)

    for itp in mod_cond:
        assert itp in params_selector
        i, j = params_selected_loc[itp]
        mod_idx = np.arange(i, j)[mod_idx]
        if force_zero:
            mod = (cond_params[:, mod_idx] * 0) + mod_interp
        else:
            mod = cond_params[:, mod_idx] + mod_interp
        final_cond[:, mod_idx] = mod.float()
    return final_cond

def load_condition(params, img_name):
    '''
    Load deca condition into dict-like
    '''
    
    load_cond = {}

    # Choose only param in params_selector
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']
    
    for p in params_key:
        each_param = []
        for name in img_name:
            each_param.append(params[name][p])
        load_cond[p] = np.stack(each_param, axis=0)

    return load_cond

def load_image(all_path, cfg, vis=False):
    '''
    Load image and stack all of thems into BxCxHxW
    '''

    imgs = []
    for path in all_path:
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        raw_img = img_utils.augmentation(pil_image=pil_image, cfg=cfg)

        raw_img = (raw_img / 127.5) - 1

        imgs.append(np.transpose(raw_img, (2, 0, 1)))
    imgs = np.stack(imgs)
    if vis:
        if imgs.shape[0] > 30:
            vis_utils.plot_sample(th.tensor(imgs[:30]))
        else:
            vis_utils.plot_sample(th.tensor(imgs))
    return {'image':th.tensor(imgs)}

def load_image_by_name(img_name, img_dataset_path, cfg, vis=False):
    '''
    Load image and stack all of thems into BxCxHxW
    '''

    imgs = []
    for name in img_name:
        path = f"{img_dataset_path}/{name}"
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        raw_img = img_utils.augmentation(pil_image=pil_image, cfg=cfg)

        raw_img = (raw_img / 127.5) - 1

        imgs.append(np.transpose(raw_img, (2, 0, 1)))
    imgs = np.stack(imgs)
    if vis:
        vis_utils.plot_sample(th.tensor(imgs))
    return {'image':th.tensor(imgs)}

def without(src, rmv):
    '''
    Remove element in rmv-list out of src-list by preserving the order
    '''
    out = []
    for s in src:
        if s not in rmv:
            out.append(s)
    return out

def get_samples_list(sample_pair_json, sample_pair_mode, src_dst, img_path, n_subject):
    '''
    return
    output of pre-finding pair is list of list : [['60065.jpg', '60001.jpg'], ['60065.jpg', '60012.jpg'], ..., n_subject]
    '''
    import json, os
    if (sample_pair_json is not None) and (sample_pair_mode is not None):
        #NOTE: Sampling with defined pairs
        assert os.path.isfile(sample_pair_json)
        f = open(sample_pair_json)
        sample_pairs = json.load(f)[sample_pair_mode]
        if sample_pair_mode == 'pair':
            src_dst = [[sample_pairs[pair_i]['src'], sample_pairs[pair_i]['dst']]
                        for pair_i in list(sample_pairs.keys())]
                        # for pair_i in list(sample_pairs.keys())[:1000]]
            all_img_idx = [file_utils.search_index_from_listpath(list_path=img_path, search=sd) 
                    for sd in src_dst]
            all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
            if n_subject > len(sample_pairs) or n_subject == -1:
                n_subject = len(sample_pairs.keys())
            
        elif sample_pair_mode == 'pairwise':
            assert len(sample_pairs['src']) == len(sample_pairs['dst']) 
            all_img_idx = [file_utils.search_index_from_listpath(list_path=img_path, search=[s, d]) 
                       for s in sample_pairs['src'] for d in sample_pairs['dst']]
            all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
            if n_subject > len(sample_pairs['dst']) or n_subject == -1:
                n_subject = len(sample_pairs['dst'])
            else: n_subject = n_subject * len(sample_pairs['dst'])
            
        elif sample_pair_mode == 'list':
            all_img_idx = file_utils.search_index_from_listpath(list_path=img_path, search=sample_pairs)
            all_img_name = [img_path[i].split('/')[-1] for i in all_img_idx]
            n_subject = -1
            
        else: raise NotImplementedError
        
    elif len(src_dst) == 2:
        #NOTE: Sampling with a specific pair
        n_subject = 1
        all_img_idx = [file_utils.search_index_from_listpath(list_path=img_path, search=src_dst)]
        all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
    else:
        #NOTE: Random samples
        all_img_idx = np.random.choice(a=range(len(img_path)), replace=False, size=n_subject * 2)
        all_img_idx = np.array_split(all_img_idx, n_subject)
        all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
    
    return all_img_idx, all_img_name, n_subject


def get_samples_pair(sample_pair_json, sample_pair_mode, src_dst, img_path, start, end, n_subject):
    '''
    return
    output of pre-finding pair is list of list : [['60065.jpg', '60001.jpg'], ['60065.jpg', '60012.jpg'], ..., n_subject]
    '''
    import json, os
    if (sample_pair_json is not None) and (sample_pair_mode is not None):
        #NOTE: Sampling with defined pairs
        assert os.path.isfile(sample_pair_json)
        f = open(sample_pair_json)
        sample_pairs = json.load(f)[sample_pair_mode]
        print("[#] Sample pair length : ", len(sample_pairs.keys()))
        if sample_pair_mode == 'pair':
            if start > len(sample_pairs.keys()):
                raise ValueError(f"start={start} is out of range of {len(sample_pairs.keys())}")
            elif end > len(sample_pairs.keys()):
                end = len(sample_pairs.keys())
            elif start > end:
                raise ValueError(f"start={start} is greater than end={end}")
            src_dst = [[sample_pairs[pair_i]['src'], sample_pairs[pair_i]['dst']]
                        for pair_i in list(sample_pairs.keys())[start:end]]
            # print(src_dst)
            all_img_idx = [file_utils.search_index_from_listpath(list_path=img_path, search=sd) 
                    for sd in src_dst]
            # print(all_img_idx)
            all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
            if n_subject > len(sample_pairs) or n_subject == -1:
                n_subject = len(sample_pairs.keys())
        else: raise NotImplementedError
        
    elif len(src_dst) == 2:
        #NOTE: Sampling with a specific pair
        n_subject = 1
        all_img_idx = [file_utils.search_index_from_listpath(list_path=img_path, search=src_dst)]
        all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
    else:
        #NOTE: Random samples
        all_img_idx = np.random.choice(a=range(len(img_path)), replace=False, size=n_subject * 2)
        all_img_idx = np.array_split(all_img_idx, n_subject)
        all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
    
    return all_img_idx, all_img_name, n_subject

def get_samples_pair_from_indices(sample_pair_json, sample_pair_mode, src_dst, img_path, sample_indices):
    '''
    return
    output of pre-finding pair is list of list : [['60065.jpg', '60001.jpg'], ['60065.jpg', '60012.jpg'], ..., n_subject]
    '''
    import json, os
    if (sample_pair_json is not None) and (sample_pair_mode is not None):
        #NOTE: Sampling with defined pairs
        assert os.path.isfile(sample_pair_json)
        f = open(sample_pair_json)
        sample_pairs = json.load(f)[sample_pair_mode]
        print("[#] Sample pair length : ", len(sample_pairs.keys()))
        if sample_pair_mode == 'pair':
            if np.any(np.array(sample_indices) > len(sample_pairs.keys())):
              raise ValueError(f"sample indices are out of range of {len(sample_pairs.keys())}")
            elif np.any(np.array(sample_indices) < 0):
                raise ValueError(f"sample indices are less than 0")
            pair_indices = [list(sample_pairs.keys())[i] for i in sample_indices]
            src_dst = [[sample_pairs[pair_i]['src'], sample_pairs[pair_i]['dst']]
                        for pair_i in pair_indices]
            # print(src_dst)
            all_img_idx = [file_utils.search_index_from_listpath(list_path=img_path, search=sd) 
                    for sd in src_dst]
            # print(all_img_idx)
            all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
            n_subject = len(sample_pairs.keys())
        else: raise NotImplementedError
    else: raise NotImplementedError
    
    return all_img_idx, all_img_name, n_subject


def ext_sub_step(n_step, batch_size=5):
    sub_step = []
    bz = batch_size
    tmp = n_step
    while tmp > 0:
        if tmp - bz > 0:
            sub_step.append(bz)
        else:
            sub_step.append(tmp)
        tmp -= bz
    return np.cumsum([0] + sub_step)