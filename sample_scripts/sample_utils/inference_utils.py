import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf
import mani_utils, params_utils
import cv2, PIL
import time

class PLSampling(pl.LightningModule):
    def __init__(self, 
                 model_dict, 
                 diffusion, 
                 forward_fn, 
                 reverse_fn, 
                 cfg, 
                 args=None, 
                 denoised_fn=None):
        
        super(PLSampling, self).__init__()
        self.forward_fn = forward_fn
        self.reverse_fn = reverse_fn
        self.denoised_fn = denoised_fn
        self.model_dict = model_dict 
        self.diffusion = diffusion
        self.cfg = cfg
        self.args = args
        self.const_noise = th.randn((len(cfg.img_model.dpm_cond_img), 3, cfg.img_model.image_size, cfg.img_model.image_size)).cuda()
        
    def forward_cond_network(self, model_kwargs):
        with th.no_grad():
            if self.cfg.img_cond_model.apply:
                x = model_kwargs['cond_img'].cuda().float()
                img_cond = self.model_dict[self.cfg.img_cond_model.name](
                    x=x,
                    emb=None,
                )
                # Override the condition
                if self.cfg.img_cond_model.override_cond != "":
                    model_kwargs[self.cfg.img_cond_model.override_cond] = img_cond
                else: raise NotImplementedError
        return model_kwargs

    def reverse_proc(self, x, model_kwargs, progress=True, store_intermediate=True, store_mean=False):
        # Mimic the ddim_sample_loop or p_sample_loop
        model_kwargs['const_noise'] = self.const_noise
        if self.reverse_fn == self.diffusion.ddim_reverse_sample_loop:
            sample, intermediate = self.reverse_fn(
                model=self.model_dict[self.cfg.img_model.name],
                x=x.cuda(),
                clip_denoised=True,
                denoised_fn = self.denoised_fn,
                model_kwargs=model_kwargs,
                progress=progress,
                store_intermidiate=store_intermediate,
                cond_xt_fn=cond_xt_fn if model_kwargs['use_cond_xt_fn'] else None,
                store_mean=store_mean
            )
        else: raise NotImplementedError

        # if store_intermediate:
            # assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    
    def forward_proc(self,
                     model_kwargs,
                     noise,
                     store_intermediate=True,
                     sdedit=None,
                     rev_mean=None,
                     add_mean=None):
        model_kwargs['const_noise'] = self.const_noise
        sample, intermediate = self.forward_fn(
            model=self.model_dict[self.cfg.img_model.name],
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.cfg.diffusion.clip_denoised,
            denoised_fn=self.denoised_fn,
            model_kwargs=model_kwargs,
            store_intermidiate=store_intermediate,
            cond_xt_fn=cond_xt_fn if model_kwargs['use_cond_xt_fn'] else None,
            sdedit=sdedit,
            rev_mean=rev_mean,
            add_mean=add_mean
        )
        if store_intermediate:
            assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    
    def override_modulator(self, cond, val=1):
        print(f"[#] Override the modulator with {val}")
        for i in range(len(cond)):
            if val == 1:
                cond[i] = th.ones_like(cond[i])
        return cond
    
def cond_xt_fn(cond, cfg, use_render_itp, t, diffusion, noise, device='cuda'):
    #NOTE: This specifically run for ['dpm_cond_img']
    
    def dpm_noise(x_start, p, k, noise, i):
        if p is None: return x_start
        share = True if p.split('-')[0] == 'share' else False
        masking = p.split('-')[-1]
        
        #NOTE: Repeat the noise to used following share/sep noise
        if share:
            noise = th.repeat_interleave(input=noise[0:1], dim=0, repeats=x_start.shape[0])
        else:
            noise = th.repeat_interleave(input=noise[[i]], dim=0, repeats=x_start.shape[0])
        if masking == 'dpm_noise_masking':
            img = cond['image']
            mask =  cond[f'{k}_mask'].bool()
            assert th.all(mask == cond[f'{k}_mask'])
            xt = (diffusion.q_sample(img, t, noise=noise) * mask) + (-th.ones_like(img) * ~mask)
        elif masking == 'dpm_noise':
                xt = diffusion.q_sample(x_start, t, noise=noise)
        else: raise NotImplementedError("[#] Only dpm_noise_masking and dpm_noise is available")
        return xt
    
    if cfg.img_model.apply_dpm_cond_img:
        dpm_cond_img = []
        for i, (k, p) in enumerate(zip(cfg.img_model.dpm_cond_img, cfg.img_model.noise_dpm_cond_img)):
            if ('faceseg' in k) or ('deca' in k): # if 'faceseg' in k:
                if use_render_itp:
                    xt = dpm_noise(x_start=cond[f'{k}'], p=p, k=k, i=i, noise=noise)
                    dpm_cond_img.append(xt)
                else:
                    xt = dpm_noise(x_start=cond[f'{k}_img'], p=p, k=k, i=i, noise=noise)
                    dpm_cond_img.append(xt)
            else:
                if use_render_itp: 
                    tmp_img = cond[f'{k}']
                else: 
                    tmp_img = cond[f'{k}_img']
                dpm_cond_img.append(tmp_img)
        cond['dpm_cond_img'] = th.cat(dpm_cond_img, dim=1)
    else:
        cond['dpm_cond_img'] = None
    return cond

def get_init_noise(n, mode, img_size, device):
    '''
    Return the init_noise used as input.
    :params mode: mode for sampling noise => 'vary_noise', 'fixed_noise'
    '''
    if mode == 'vary_noise':
        init_noise = th.randn((n, 3, img_size, img_size))
    elif mode == 'fixed_noise':
        init_noise = th.cat([th.randn((1, 3, img_size, img_size))] * n, dim=0)
    else: raise NotImplementedError

    return init_noise.to(device)

def to_tensor(cond, key, device):
    for k in key:
        if k not in cond.keys():
            continue
        else:
            if isinstance(cond[k], list):
                for i in range(len(cond[k])):
                    cond[k][i] = th.tensor(cond[k][i]).to(device)
            else:
                if th.is_tensor(cond[k]):
                    cond[k] = cond[k].clone().to(device)
                elif isinstance(cond[k], np.ndarray):
                    cond[k] = th.tensor(cond[k]).to(device)
                else:
                    continue
    return cond

def eval_mode(model_dict):
    for k, _ in model_dict.items():
        model_dict[k].eval()
    return model_dict

def prepare_cond_sampling(cond, cfg, use_render_itp=False, device='cuda'):
    """
    Prepare a condition for encoder network (e.g., adding noise, share noise with DPM)
    :param noise: noise map used in DPM
    :param t: timestep
    :param model_kwargs: model_kwargs dict
    ###Note: 
     - cond[f'{k}'_img] is the original one from dataloader
     - cond[f'{k}'] is the original one render & build_condition_image() fn
    """
    
    # print("PREPCOND", cond.keys())
    # print(cond['shadow_mask_img'].shape)
    # print(cond['deca_masked_face_images_woclip_img'].shape)
    # exit()
    if cfg.img_model.apply_dpm_cond_img:
        dpm_cond_img = []
        for k in cfg.img_model.dpm_cond_img:
            if use_render_itp: 
                tmp_img = cond[f'{k}']
            else: 
                tmp_img = cond[f'{k}_img']
            dpm_cond_img.append(tmp_img.to(device))
        cond['dpm_cond_img'] = th.cat((dpm_cond_img), dim=1)
    else:
        cond['dpm_cond_img'] = None
        
    if cfg.img_cond_model.apply:
        cond_img = []
        for k in cfg.img_cond_model.in_image:
            print(k)
            if use_render_itp: 
                tmp_img = cond[f'{k}']
            else: 
                tmp_img = cond[f'{k}_img']
            print(k, tmp_img.shape)
            cond_img.append(tmp_img.to(device))
        cond['cond_img'] = th.cat((cond_img), dim=1).to(device)
        # cond['cond_img'] = th.flip(cond['cond_img'], [0])
        
    else:
        cond['cond_img'] = None
        
    return cond

def shadow_diff_with_weight_postproc(cond, misc, device='cuda'):
    # This function is used to post-process the shadow_diff mask when we inject shadow weight into shadow_diff
    condition_img = misc['condition_img']
    n_step = misc['n_step']
    src_idx = misc['src_idx']
    dst_idx = misc['dst_idx']
    args = misc['args']
    # Max-Min c-values:  -4.985533880236826 7.383497233314015 from valid set
    # Max-Min c-values: -4.989461058405101 8.481700287326827
    max_c = 8.481700287326827 # 7.383497233314015
    min_c = -4.989461058405101 # -4.985533880236826
    c_val_src = (cond['shadow'][src_idx] - min_c) / (max_c - min_c)  # Scale to 0-1
    c_val_src = 1 - c_val_src # Inverse the shadow value
    c_val_dst = (cond['shadow'][dst_idx] - min_c) / (max_c - min_c)  # Scale to 0-1
    c_val_dst = 1 - c_val_dst # Inverse the shadow value

    if args.shadow_diff_inc_c:
        print(f"[#] Increasing the shadow_diff with weight...")
        print(f"[#] Default C is {c_val_src.item()}")
        weight_src = th.linspace(start=c_val_src.item(), end=1.0, steps=n_step).to(device)
        weight_src = weight_src[..., None, None, None]
        fix_frame = True 
    elif args.shadow_diff_dec_c:
        print("[#] Decreasing the shadow_diff with weight...")
        print(f"[#] Default C is {c_val_src.item()}")
        weight_src = th.linspace(start=c_val_src.item(), end=0.0, steps=n_step).to(device)
        weight_src = weight_src[..., None, None, None]
        fix_frame = True 
    else:
        print("[#] No re-weighting for shadow_diff...")
        print(f"[#] Processing with weight = {c_val_src} and relight...")
        if 'shadow_diff_with_weight_simplified' in condition_img:
            print("[#] Conditioning is shadow_diff_with_weight_simplified, set the weight to c_val...")
            weight_src = c_val_src[..., None, None, None].to(device)
            weight_dst = c_val_dst[..., None, None, None].to(device)
            weight_src_inversion = (c_val_src).to(device)
            if args.relight_with_strongest_c:
                print("[#] Relight with the strongest c_val, set the weight to 0.0...")
                weight_src = (c_val_src * 0.0).to(device)
                weight_dst = (c_val_dst * 0.0).to(device)
        fix_frame = False

    for _, cond_img_name in enumerate(condition_img):
        if (cond_img_name == 'shadow_diff_with_weight_simplified') or (cond_img_name == 'shadow_diff_with_weight_simplified_inverse'):
            if fix_frame:
                fidx = int(args.shadow_diff_fidx_frac * n_step)
                tmp = th.repeat_interleave(cond[cond_img_name][fidx:fidx+1], repeats=n_step-1, dim=0)
                # Always preserve first frame, the rest is determined by the shadow_diff_fidx_frac
                sd_img = cond[cond_img_name].clone()
                sd_img = th.cat((sd_img[0:1], tmp), dim=0)
            else:
                sd_img = cond[cond_img_name].clone()
            
            if args.relight_with_shadow_diff:
                # First frame (0 is non-shadow, > 0 is shadow)   
                sd_shadow = sd_img[0:1] > 0.
                shadow_area = sd_shadow * (1-weight_src)     # Shadow area assigned weight
                shadow_ff = shadow_area
                # Rest of the frames
                if args.relight_with_dst_c:
                    shadow_rf = ((sd_img[1:] > 0.) * (1-weight_dst))
                elif args.relight_with_rand_max_c:
                    rand_c = th.tensor(np.random.uniform(0.5, 1.0, size=weight_src.shape)).type_as(weight_src)
                    print(f"[#] Relight with random c_val [0.5 to 1] => {rand_c}")
                    shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + rand_c))
                else:
                    shadow_rf = ((sd_img[1:] > 0.) * (1-weight_src))    # Shadow area assigned weight

                # Final frames
                shadow = th.cat((shadow_ff, shadow_rf), dim=0)
                out_sd = shadow
                cond[cond_img_name] = out_sd
            else:
                if args.postproc_shadow_mask_smooth_keep_shadow_shading:
                    #NOTE: Keep the shadow shading from perturbed light

                    # First frame (0 is non-shadow, > 0 is shadow)   
                    sd_shadow = sd_img[0:1] > 0.
                    shadow_area = sd_shadow * (1-weight_src)     # Shadow area assigned weight
                    shadow_ff = shadow_area
                    # Rest of the frames
                    if args.relight_with_dst_c:
                        shadow_rf = (sd_img[1:] * (1-weight_dst))
                    elif args.relight_with_rand_max_c:
                        rand_c = th.tensor(np.random.uniform(0.5, 1.0, size=weight_src.shape)).type_as(weight_src)
                        print(f"[#] Relight with random c_val [0.5 to 1] => {rand_c}")
                        shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + rand_c))
                    else:
                        shadow_rf = (sd_img[1:] * (1-weight_src))    # Shadow area assigned weight

                    # Final frames
                    shadow = th.cat((shadow_ff, shadow_rf), dim=0)
                    out_sd = shadow
                    cond[cond_img_name] = out_sd
                    print("Value: ", th.unique(out_sd))
                elif args.postproc_shadow_mask_smooth:
                    #NOTE: Do not keep the shadow shading from perturbed light
                    # First frame
                    sd_shadow = sd_img[0:1] > 0.
                    shadow_area = sd_shadow * (1-weight_src_inversion)     # Shadow area assigned weight
                    shadow_ff = shadow_area
                    # Rest of the frames
                    if args.relight_with_dst_c:
                        print(f"[#] Relight with the dst c_val = {weight_dst.flatten()}")
                        shadow_rf = (sd_img[1:] * (1-weight_dst))
                    elif args.relight_with_rand_max_c:
                        rand_c = th.tensor(np.random.uniform(0.7, 1.0, size=weight_src.shape)).type_as(weight_src)
                        print(f"[#] Relight with random c_val [0.7 to 1] => {rand_c}")
                        shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + rand_c))
                    elif args.relight_with_given_c is not None:
                        print(f"[#] Relight with the given c_val = {args.relight_with_given_c}")
                        shadow_rf = ((sd_img[1:] > 0.) * ((weight_dst * 0.) + args.relight_with_given_c))
                    else:
                        print(f"[#] Relight with the src c_val = {weight_src.flatten()}")
                        shadow_rf = (sd_img[1:] * (1-weight_src))    # Shadow area assigned weight
                    # Final frames
                    shadow = th.cat((shadow_ff, shadow_rf), dim=0)
                    out_sd = shadow
                    cond[cond_img_name] = out_sd
                    print("[#] Unique of out_sd: ", th.unique(out_sd))
                    print("[#] Unique of shadow_ff: ", th.unique(shadow_ff))
                    print("[#] Unique of shadow_rf: ", th.unique(shadow_rf))
                else:
                    raise NotImplementedError

    return cond, {'src':weight_src, 'dst':weight_dst}

def shadow_diff_final_postproc(cond, misc):
    # This function for post-processing the shadow_diff mask 
    # 1. Smoothing the first and second frame
    import torchvision
    condition_img = misc['condition_img']
    n_step = misc['n_step']
    src_idx = misc['src_idx']
    args = misc['args']

    for _, cond_img_name in enumerate(condition_img):
        if (cond_img_name == 'shadow_diff_with_weight_simplified') or (cond_img_name == 'shadow_diff_with_weight_simplified_inverse'):
            if args.smooth_SD_to_SM:
                sm = cond[cond_img_name][0:1].clone()
                sdiff = cond[cond_img_name][1:2].clone()
                sm = sm.squeeze(0).permute(1, 2, 0).cpu().numpy()
                sdiff = sdiff.squeeze(0).permute(1, 2, 0).cpu().numpy()
                proc_set = {'sm':sm, 'sdiff':sdiff}
                out = {'sm': [sm], 'sdiff': [sdiff]}

                for k in proc_set.keys():
                    tmp = proc_set[k].copy()
                    erosion_size = 2
                    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))
                    org = tmp.copy()


                    break_cond = True
                    breaking_repeated = []
                    while break_cond:
                        # tmp = ndimage.binary_erosion(tmp, mask=~intr, iterations=1)
                        # print("Before: ", np.max(tmp), np.min(tmp))
                        tmp = cv2.erode(tmp, kernel=element, iterations=1)  # with cv2.erode the last dim is squeezed
                        tmp = tmp[..., None]
                        # print("After: ", np.max(tmp), np.min(tmp))
                        tmp = (tmp * ~intr) + (intr * org) # Modified only the area inside intr

                        out[k].append(tmp)
                        break_cond = (np.logical_and(tmp, intr).sum() / np.logical_or(tmp, intr).sum()) < 0.99
                        breaking_repeated.append(break_cond)
                        if len(breaking_repeated) > 20:
                            break_cond = False if np.mean(breaking_repeated[-10:]) == breaking_repeated[-1] else True

                print(len(out))
                vids = {}
                for k in out.keys():
                    print("[#] Length of out[{}]: {}".format(k, len(out[k])))
                    vid = np.stack(out[k])
                    vid = np.repeat(vid, 3, axis=-1)
                    vid = vid * 255
                    print(vid.shape)
                    vids[k] = vid
                    torchvision.io.write_video('{}.mp4'.format(k), th.tensor(vid), 20)

                final_vid = np.concatenate([vids['sdiff'], vids['sm'][::-1, ...]], axis=0)
                torchvision.io.write_video('final.mp4', th.tensor(final_vid), 20)

                # subsampling the video to have the equal number of frames for both sdiff and sm (shorter one)
                shorter = min(len(vids['sdiff']), len(vids['sm']))
                longer = max(len(vids['sdiff']), len(vids['sm']))
                idx = np.linspace(0, longer-1, shorter).astype(int)

                if len(vids['sdiff']) < len(vids['sm']):
                    vids['sm'] = vids['sm'][::-1, ...][idx]
                else:
                    vids['sdiff'] = vids['sdiff'][idx]

                # union version
                final_union_vid = np.logical_or(vids['sdiff'][1:], vids['sm'][:-1])
                final_union_vid = np.concatenate([vids['sdiff'][0:1] > 0, final_union_vid, vids['sm'][-1:] > 0], axis=0)


                final_out = th.tensor(final_out).cuda()
                cond[cond_img_name] = final_out

                # Append shading referece & face structure to have same length with the morph shadow_diff
                sr_0 = cond['deca_masked_face_images_woclip'][0:1].repeat_interleave(repeats=sdiff_erd.shape[0], dim=0)
                sr_1 = cond['deca_masked_face_images_woclip'][1:2].repeat_interleave(repeats=sm_erd.shape[0], dim=0)
                sr = th.cat((sr_0, sr_1, cond['deca_masked_face_images_woclip'][2:]), dim=0)
                cond['deca_masked_face_images_woclip'] = sr
                assert cond[cond_img_name].shape[0] == cond['deca_masked_face_images_woclip'].shape[0]
                misc['n_step'] = final_out.shape[0]
                cond['face_structure'] = cond['face_structure'][0:1].repeat_interleave(repeats=final_out.shape[0], dim=0)

    return cond, misc

def build_condition_image(cond, misc):
    src_idx = misc['src_idx']
    dst_idx = misc['dst_idx']
    n_step = misc['n_step']
    batch_size = misc['batch_size']
    render_batch_size = misc['render_batch_size']
    avg_dict = misc['avg_dict']
    dataset = misc['dataset']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    itp_func = misc['itp_func']
    deca_obj = misc['deca_obj']
    clip_ren = None
    
    # Handling the render face
    if np.any(['deca' in i for i in condition_img]) or np.any(['shadow_mask' in i for i in condition_img]) or np.any(['shadow_diff' in i for i in condition_img]):
        # Render the face
        if args.sh_grid_size is not None:
            #NOTE: Render w/ grid light 
            cond['light'] = params_utils.grid_sh(sh=cond['light'][src_idx], n_grid=args.sh_grid_size, sx=args.sh_span_x, sy=args.sh_span_y, sh_scale=args.sh_scale, use_sh=args.use_sh).reshape(-1, 27)
        elif 'render_face' in args.interpolate:
            #NOTE: Render w/ interpolated light (Mainly use this)
            if args.spiral_sh:
                print("[#] Spiral SH mode of src light...")
                interp_cond = mani_utils.spiral_sh(cond, src_idx=src_idx, n_step=n_step)
            elif args.rotate_sh:
                print("[#] Rotate SH mode of src light...")
                interp_cond = mani_utils.rotate_sh(cond, src_idx=src_idx, n_step=n_step, axis=args.rotate_sh_axis)
            else:
                print("[#] Interpolating SH mode from src->dst light...")
                interp_cond = mani_utils.iter_interp_cond(cond, interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
            cond.update(interp_cond)
        else:
            #NOTE: Render w/ same light
            repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light'])
            cond.update(repeated_cond)
        
        start_t = time.time()
        if np.any(['deca_masked' in n for n in condition_img]) or np.any(['shadow_mask' in n for n in condition_img]) or np.any(['shadow_diff' in n for n in condition_img]):
            mask = params_utils.load_flame_mask()
        else: mask=None
        
        #TODO: Render DECA in minibatch
        # sub_step = mani_utils.ext_sub_step(n_step, batch_size)
        sub_step = mani_utils.ext_sub_step(n_step, render_batch_size)
        all_render = []
        load_deca_time = time.time() - start_t
        render_time = []
        all_shadow_mask = []
        all_shadow_kk = []
        all_render_ld = []
        for i in range(len(sub_step)-1):
            print(f"[#] Sub step rendering : {sub_step[i]} to {sub_step[i+1]}")
            start = sub_step[i]
            end = sub_step[i+1]
            sub_cond = cond.copy()
            sub_cond['light'] = sub_cond['light'][start:end, :]
            # Deca rendered : B x 3 x H x W
            deca_rendered, orig_visdict = params_utils.render_deca(deca_params=sub_cond, 
                                                                idx=src_idx, n=end-start, 
                                                                avg_dict=avg_dict, 
                                                                render_mode=args.render_mode, 
                                                                rotate_normals=args.rotate_normals, 
                                                                mask=mask,
                                                                deca_obj=deca_obj,
                                                                repeat=True)
            # Shadow_mask : B x H x W
            if args.render_same_mask:
                print("[#] Rendering with the shadow mask from same render face...")
                shadow_mask, shadow_kk = params_utils.render_shadow_mask_with_smooth(
                                                sh_light=sub_cond['light'], 
                                                cam=sub_cond['cam'][src_idx],
                                                verts=orig_visdict['trans_verts_orig'], 
                                                deca=deca_obj)
            else:
                print("[#] Rendering with the shadow mask from face + scalp of render face...")
                if i == 0:
                    flame_face_scalp = params_utils.load_flame_mask(['face', 'scalp', 'left_eyeball', 'right_eyeball'])
                    deca_obj_face_scalp = params_utils.init_deca(mask=flame_face_scalp, rasterize_type=args.rasterize_type) # Init DECA with mask only once
                if args.rotate_sh_axis == 1:
                    print("[#] Fixing the axis 1...")
                shadow_mask, shadow_kk, render_ld = params_utils.render_shadow_mask_with_smooth(
                                                sh_light=sub_cond['light'], 
                                                cam=sub_cond['cam'][src_idx],
                                                verts=orig_visdict['trans_verts_orig'], 
                                                use_sh_to_ld_region=args.use_sh_to_ld_region,
                                                deca={'face_scalp':deca_obj_face_scalp}, 
                                                axis_1=args.rotate_sh_axis==1,
                                                device='cpu',   # Prevent OOM
                                                up_rate=args.up_rate_for_AA,
                                                org_h=img_size, org_w=img_size,
                                                rt_dict={'pt_round':args.pt_round, 'pt_radius':args.pt_radius, 'rt_regionG_scale':args.rt_regionG_scale, 'scale_depth':args.scale_depth}
                                            )
                # shadow_mask, shadow_kk, render_ld = params_utils.render_shadow_mask_with_smooth_nopt(
                #                                 sh_light=sub_cond['light'], 
                #                                 cam=sub_cond['cam'][src_idx],
                #                                 verts=orig_visdict['trans_verts_orig'], 
                #                                 use_sh_to_ld_region=args.use_sh_to_ld_region,
                #                                 deca={'face_scalp':deca_obj_face_scalp}, 
                #                                 axis_1=args.rotate_sh_axis==1,
                #                                 device='cpu',   # Prevent OOM
                #                                 up_rate=args.up_rate_for_AA,
                #                                 org_h=img_size, org_w=img_size,
                #                                 rt_dict={'pt_round':args.pt_round, 'pt_radius':args.pt_radius, 'rt_regionG_scale':args.rt_regionG_scale}
                #                             )
                if i == len(sub_step)-2:
                    del deca_obj_face_scalp
            all_render.append(deca_rendered)
            render_time.append(time.time() - start_t)
            
            all_shadow_mask.append(shadow_mask[:, None, ...])
            all_shadow_kk.append(shadow_kk[:, None, ...])
            all_render_ld.append(render_ld)
            
        
        if args.use_sh_to_ld_region:
            all_render_ld = th.cat(all_render_ld, dim=0)
            cond['render_ld'] = all_render_ld
        else:
            all_render_ld = None
            cond['render_ld'] = None

        render_time = np.mean(render_time) + load_deca_time
        cond['render_time'] = render_time
        print("Rendering time : ", time.time() - start_t)
        
        if args.fixed_render:
            print("[#] Fixed the Deca renderer")
            print(all_render[0].shape) # List of  [B x 3 x H x W, ...]
            deca_rendered = all_render[0][0:1].repeat_interleave(repeats=n_step, dim=0)
            print(deca_rendered.shape)
        else:
            deca_rendered = th.cat(all_render, dim=0)
            
        if args.fixed_shadow:
            print("[#] Fixed the Shadow mask")
            print(all_shadow_mask[0].shape) # List of  [B x 1 x H x W, ...]
            shadow_mask = all_shadow_mask[0][0:1].repeat_interleave(repeats=n_step, dim=0)
            print(shadow_mask.shape)
        else:
            shadow_mask = th.cat(all_shadow_mask, dim=0)
            shadow_kk = th.cat(all_shadow_kk, dim=0)

            
    print("Conditoning with image : ", condition_img)
    for i, cond_img_name in enumerate(condition_img):
        if ('faceseg' in cond_img_name) or ('laplacian' in cond_img_name) or ('sobel' in cond_img_name) or ('face_structure' in cond_img_name) or ('canny_edge_bg' in cond_img_name):
            bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * n_step
            if th.is_tensor(cond[f"{cond_img_name}_img"][src_idx]):
                bg_tmp = th.stack(bg_tmp, axis=0)
            else:
                bg_tmp = np.stack(bg_tmp, axis=0)
            cond[f"{cond_img_name}"] = th.tensor(bg_tmp)
            
        elif 'deca' in cond_img_name:
            rendered_tmp = []
            for j in range(n_step):
                if 'woclip' in cond_img_name:
                    #NOTE: Input is the npy array -> Used cv2.resize() to handle
                    r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
                    r_tmp = cv2.resize(r_tmp, (img_size, img_size), cv2.INTER_AREA)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    clip_ren = False
                else:
                    r_tmp = deca_rendered[j].mul(255).add_(0.5).clamp_(0, 255)
                    r_tmp = np.transpose(r_tmp.cpu().numpy(), (1, 2, 0))
                    r_tmp = r_tmp.astype(np.uint8)
                    r_tmp = dataset.augmentation(PIL.Image.fromarray(r_tmp))
                    r_tmp = dataset.prep_cond_img(r_tmp, cond_img_name, i)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    r_tmp = (r_tmp / 127.5) - 1
                    clip_ren = True
                rendered_tmp.append(r_tmp)
            rendered_tmp = np.stack(rendered_tmp, axis=0)
            cond[cond_img_name] = th.tensor(rendered_tmp).cuda()
        elif 'shadow_mask' in cond_img_name:
            shadow_mask_tmp = []
            for j in range(n_step):
                sm_tmp = shadow_mask[j].mul(255).add_(0.5).clamp_(0, 255)
                sm_tmp = sm_tmp.repeat_interleave(repeats=3, dim=0)
                sm_tmp = np.transpose(sm_tmp.cpu().numpy(), (1, 2, 0))  # HxWxC
                sm_tmp = sm_tmp.astype(np.uint8)
                sm_tmp = dataset.augmentation(PIL.Image.fromarray(sm_tmp))
                sm_tmp = dataset.prep_cond_img(sm_tmp, cond_img_name, i)
                sm_tmp = np.transpose(sm_tmp, (2, 0, 1))    # CxHxW
                sm_tmp = (sm_tmp / 127.5) - 1
                shadow_mask_tmp.append(sm_tmp[[0], ...])
            shadow_mask_tmp = np.stack(shadow_mask_tmp, axis=0)
            cond[cond_img_name] = th.tensor(shadow_mask_tmp).cuda()
        elif 'shadow_diff' in cond_img_name:
            shadow_diff_tmp = []
            m_face_tmp = []
            m_face_sd_tmp = []
            for j in range(n_step):
                sd_tmp = shadow_mask[j]
                sdkk_tmp = shadow_kk[j]
                if args.postproc_shadow_mask:
                    # Thresholding & Masking & Fill bg with 0.5
                    m_glasses_and_eyes = cond[f'{cond_img_name}_meg_mask'][src_idx].cpu().numpy()

                    # Masking out the bg area
                    m_face_parsing = cond[f'{cond_img_name}_mface_mask'][src_idx].cpu().numpy()
                    m_face_sd = sd_tmp > 0.01
                    if args.use_ray_mask:
                        m_face = m_face_parsing * m_face_sd
                    else:
                        m_face = m_face_parsing

                    m_face_tmp.append(m_face)   # Mask from face parsing
                    m_face_sd_tmp.append(m_face_sd) # Mask from Ray-tracing

                    sd_tmp = (sd_tmp < 0.5) * 1.0   # Bg
                    sd_tmp = ((sd_tmp * np.abs(1-m_glasses_and_eyes)) + (1.0 * m_glasses_and_eyes))
                    sd_tmp = np.abs(1 - sd_tmp) # Inverse => Shadow = 0, Non-shadow = 1
                    sd_tmp = (((sd_tmp * np.abs(1-m_glasses_and_eyes)) + (1.0 * m_glasses_and_eyes)) * m_face) + (0.5 * np.abs(1-m_face))
                
                elif args.postproc_shadow_mask_smooth_keep_shadow_shading:
                    #NOTE: Keep the shadow shading from perturbed light
                    m_glasses_and_eyes = cond[f'{cond_img_name}_meg_mask'][src_idx].cpu().numpy()
                    # Masking out the bg area
                    m_face_parsing = cond[f'{cond_img_name}_mface_mask'][src_idx].cpu().numpy()
                    m_face = m_face_parsing

                    sd_tmp_proc = ((1 - sd_tmp)) * (m_face * (1-m_glasses_and_eyes)) * ((1 - sdkk_tmp) > 0)
                    sd_tmp = sd_tmp_proc

                    # sd_tmp_proc = (sd_tmp)
                    # sd_tmp_proc = ((sd_tmp_proc * np.abs(1-m_glasses_and_eyes)) + (1.0 * m_glasses_and_eyes))
                    # sd_tmp_proc = np.abs(1 - sd_tmp_proc) # Inverse => Shadow = 0, Non-shadow = 1
                    # sd_tmp_proc = (((sd_tmp_proc * np.abs(1-m_glasses_and_eyes)) + (1.0 * m_glasses_and_eyes)) * m_face) + (1 * np.abs(1-m_face))
                    # mask_for_sd = th.tensor(sd_tmp_proc >= 0.5) * th.tensor(sd_tmp_proc < 1.0) * 1.0
                    # sd_tmp_proc = mask_for_sd * sd_tmp_proc
                    # sd_tmp = sd_tmp_proc

                elif args.postproc_shadow_mask_smooth:
                    #NOTE: Do not keep the shading of shadows from perturbed light
                    m_glasses_and_eyes = cond[f'{cond_img_name}_meg_mask'][src_idx].cpu().numpy()
                    # Masking out the bg area
                    m_face_parsing = cond[f'{cond_img_name}_mface_mask'][src_idx].cpu().numpy()
                    m_face = m_face_parsing
                    
                    sd_tmp_proc = (((1 - sd_tmp) > 0) * 1.0) * (m_face * (1-m_glasses_and_eyes)) * ((1 - sdkk_tmp) > 0.0)
                    sd_tmp = sd_tmp_proc


                shadow_diff_tmp.append(sd_tmp)
                
            if args.inverse_with_shadow_diff:
                print("[#] Inverse with shadow_diff (Replacing frame-0th)...")
                shadow_diff_tmp[0] = cond['shadow_diff_img'][src_idx]
                if args.fixed_shadow:
                    shadow_diff_tmp = [cond['shadow_diff_img'][src_idx] for _ in range(len(shadow_diff_tmp))]
            if args.relight_with_shadow_diff:
                # Relighting with the target shadow diff (For MultiPIE dataset only while we have same face structure for input-groundtruth)
                # NOTE: Work with n-frames == 2 only
                shadow_diff_tmp[-1] = cond['shadow_diff_img'][dst_idx]
                print("[#] Relight with shadow_diff...")

            shadow_diff_tmp = np.stack(shadow_diff_tmp, axis=0)
            cond[cond_img_name] = th.tensor(shadow_diff_tmp).cuda()

    
    return cond, clip_ren


def build_condition_image_for_vids(cond, misc):
    batch_size = misc['batch_size']
    avg_dict = misc['avg_dict']
    dataset = misc['dataset']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    deca_obj = misc['deca_obj']
    clip_ren = None
    
    # Handling the render face
    if np.any(['deca' in i for i in condition_img]):
        
        start_t = time.time()
        if np.any(['deca_masked' in n for n in condition_img]):
            mask = params_utils.load_flame_mask()
        else: mask=None
        
        #TODO: Render DECA in minibatch
        all_render = []
        print(f"[#] Rendering...")
        # Deca rendered : B x 3 x H x W
        deca_rendered, orig_visdict = params_utils.render_deca_videos(
                                                            deca_params=cond, 
                                                            avg_dict=avg_dict, 
                                                            render_mode=args.render_mode, 
                                                            rotate_normals=args.rotate_normals, 
                                                            mask=mask,
                                                            deca_obj=deca_obj)
        all_render.append(deca_rendered)
        print("Rendering time : ", time.time() - start_t)
        deca_rendered = th.cat(all_render, dim=0)
        
    #TODO: Make this applicable to either 'cond_img' or 'dpm_cond_img'
    print("Conditoning with image : ", condition_img)
    for i, cond_img_name in enumerate(condition_img):
        if ('faceseg' in cond_img_name) or ('laplacian' in cond_img_name) or ('canny_edge_bg' in cond_img_name):
            bg_tmp = cond[f"{cond_img_name}_img"]
            cond[f"{cond_img_name}"] = th.tensor(bg_tmp)
            
        elif 'deca' in cond_img_name:
            rendered_tmp = []
            for j in range(deca_rendered.shape[0]):
                if 'woclip' in cond_img_name:
                    #NOTE: Input is the npy array -> Used cv2.resize() to handle
                    r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
                    r_tmp = cv2.resize(r_tmp, (img_size, img_size), cv2.INTER_AREA)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    clip_ren = False
                else:
                    r_tmp = deca_rendered[j].mul(255).add_(0.5).clamp_(0, 255)
                    r_tmp = np.transpose(r_tmp.cpu().numpy(), (1, 2, 0))
                    r_tmp = r_tmp.astype(np.uint8)
                    r_tmp = dataset.augmentation(PIL.Image.fromarray(r_tmp))
                    r_tmp = dataset.prep_cond_img(r_tmp, cond_img_name, i)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    r_tmp = (r_tmp / 127.5) - 1
                    clip_ren = True
                rendered_tmp.append(r_tmp)
            rendered_tmp = np.stack(rendered_tmp, axis=0)
            cond[cond_img_name] = th.tensor(rendered_tmp).cuda()
        else: continue  # We don't process ['raw'] here
    print(cond.keys())
    exit()
    return cond, clip_ren

def build_condition_image_rotateSH(cond, misc):
    src_idx = misc['src_idx']
    sub_step = misc['sub_step']
    batch_size = misc['batch_size']
    avg_dict = misc['avg_dict']
    dataset = misc['dataset']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    deca_obj = misc['deca_obj']
    clip_ren = None
    
    # Handling the render face
    if np.any(['deca' in i for i in condition_img]):
        
        start_t = time.time()
        if np.any(['deca_masked' in n for n in condition_img]):
            mask = params_utils.load_flame_mask()
        else: mask=None
        
        all_render = []
        print(f"[#] Rendering ({sub_step} steps)...")
        # Deca rendered : B x 3 x H x W
        deca_rendered, orig_visdict = params_utils.render_deca_rotateSH(
                                                            deca_params=cond, 
                                                            avg_dict=avg_dict, 
                                                            render_mode=args.render_mode, 
                                                            rotate_normals=args.rotate_normals, 
                                                            mask=mask,
                                                            deca_obj=deca_obj)
        all_render.append(deca_rendered)
        print("Rendering time : ", time.time() - start_t)
        deca_rendered = th.cat(all_render, dim=0)
        
    #TODO: Make this applicable to either 'cond_img' or 'dpm_cond_img'
    print("Conditoning with image : ", condition_img)
    for i, cond_img_name in enumerate(condition_img):
        if ('faceseg' in cond_img_name) or ('laplacian' in cond_img_name) or ('canny_edge_bg' in cond_img_name):
            bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * sub_step
            if th.is_tensor(cond[f"{cond_img_name}_img"][src_idx]):
                bg_tmp = th.stack(bg_tmp, axis=0)
            else:
                bg_tmp = np.stack(bg_tmp, axis=0)
            cond[f"{cond_img_name}"] = th.tensor(bg_tmp)
            
            
            # bg_tmp = cond[f"{cond_img_name}_img"]
            # cond[f"{cond_img_name}"] = th.tensor(bg_tmp)
            
            # # faceseg_use = f"{cond_img_name}_img" if f"{cond_img_name}_img" in list(cond.keys()) else f"{cond_img_name}"
            
            # # bg_tmp = [cond[f"{faceseg_use}"][src_idx]] * sub_step
            # # if th.is_tensor(cond[f"{faceseg_use}"][src_idx]):
            # #     bg_tmp = th.stack(bg_tmp, axis=0)
            # # else:
            # #     bg_tmp = np.stack(bg_tmp, axis=0)
            # bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * sub_step
            # if th.is_tensor(cond[f"{cond_img_name}_img"][src_idx]):
            #     bg_tmp = th.stack(bg_tmp, axis=0)
            # else:
            #     bg_tmp = np.stack(bg_tmp, axis=0)
            
        elif 'deca' in cond_img_name:
            rendered_tmp = []
            for j in range(deca_rendered.shape[0]):
                if 'woclip' in cond_img_name:
                    #NOTE: Input is the npy array -> Used cv2.resize() to handle
                    r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
                    r_tmp = cv2.resize(r_tmp, (img_size, img_size), cv2.INTER_AREA)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    clip_ren = False
                else:
                    r_tmp = deca_rendered[j].mul(255).add_(0.5).clamp_(0, 255)
                    r_tmp = np.transpose(r_tmp.cpu().numpy(), (1, 2, 0))
                    r_tmp = r_tmp.astype(np.uint8)
                    r_tmp = dataset.augmentation(PIL.Image.fromarray(r_tmp))
                    r_tmp = dataset.prep_cond_img(r_tmp, cond_img_name, i)
                    r_tmp = np.transpose(r_tmp, (2, 0, 1))
                    r_tmp = (r_tmp / 127.5) - 1
                    clip_ren = True
                rendered_tmp.append(r_tmp)
            rendered_tmp = np.stack(rendered_tmp, axis=0)
            cond[cond_img_name] = th.tensor(rendered_tmp).cuda()
    return cond, clip_ren