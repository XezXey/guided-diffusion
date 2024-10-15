import numpy as np
import torch as th
import time, json, sys, os, tqdm
import argparse
import torchvision
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--set_', default='train')
parser.add_argument('--sample_json', required=True)
parser.add_argument('--rasterize_type', default='standard')
parser.add_argument('--n_frames', type=int, required=True)
parser.add_argument('--out_path', required=True)
args = parser.parse_args()

def render(img_name):
    # Load the DeCA model
    sj = img_name
    rotate_sh_axis = 2
    n_step = args.n_frames
    src_idx = 0
    cond = deca_params[sj].copy()
    render_batch_size = 20
    render_mode = 'shape'
    rotate_normals = False
    for k in cond.keys():
        cond[k] = th.tensor(cond[k][None, ...])
    cond['raw_image_path'] = ['/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/train/' + sj]
    cond['raw_image'] = th.tensor(np.array(Image.open(cond['raw_image_path'][0]))).permute(2, 0, 1)[None, ...]

    interp_cond = mani_utils.rotate_sh(cond, src_idx=src_idx, n_step=n_step, axis=rotate_sh_axis)
    sub_step = mani_utils.ext_sub_step(n_step, render_batch_size)
    cond['light'] = interp_cond['light']

    for i in range(len(sub_step)-1):
        start = sub_step[i]
        end = sub_step[i+1]
        sub_cond = cond.copy()
        sub_cond['light'] = sub_cond['light'][start:end, :]
        # Deca rendered : B x 3 x H x W
        deca_rendered, orig_visdict = params_utils.render_deca(deca_params=sub_cond, 
                                                            idx=0, n=end-start, 
                                                            avg_dict=avg_dict, 
                                                            render_mode=render_mode, 
                                                            rotate_normals=rotate_normals, 
                                                            mask=mask,
                                                            deca_obj=deca_obj,
                                                            repeat=True)
    return deca_rendered, orig_visdict
        
if __name__ == '__main__':
    import misc
    deca_params, avg_dict = misc.load_deca_params(deca_dir=f'/data/mint/DPM_Dataset/ffhq_256_with_anno/params/{args.set_}', cfg=None)

    sys.path.append("/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/sample_utils/")
    import mani_utils, params_utils
    start_t = time.time()
    mask = params_utils.load_flame_mask()
    deca_obj = params_utils.init_deca(mask=mask, rasterize_type=args.rasterize_type)
    print(f"[#] Time taken to load DECA: {time.time()-start_t}")

    rendered_img_path = f'{args.out_path}/deca_masked_face_images_wclip/{args.set_}/'
    os.makedirs(rendered_img_path, exist_ok=True)
    rendered_npy_path = f'{args.out_path}/deca_masked_face_images_woclip/{args.set_}/'
    os.makedirs(rendered_npy_path, exist_ok=True)


    with open(args.sample_json) as f:
        pairs = json.load(f)['pair']

    for k, v in tqdm.tqdm(pairs.items()):
        img_name = v['src']
        deca_rendered, orig_visdict = render(img_name)

        input_rendered = deca_rendered[0:1].cpu().numpy().copy()
        relit_rendered = deca_rendered[1:].cpu().numpy().copy()

        # Save the rendered images
        torchvision.utils.save_image(th.tensor(input_rendered), f"{rendered_img_path}/{img_name.split('.')[0]}_input.png")
        for i in range(relit_rendered.shape[0]):
            torchvision.utils.save_image(th.tensor(relit_rendered[i:i+1]), f"{rendered_img_path}/{img_name.split('.')[0]}_{v['dst'].split('.')[0]}_f{i+1}_relit.png")

        input_rendered = deca_rendered[0:1].cpu().numpy().transpose(0, 2, 3, 1).copy()
        relit_rendered = deca_rendered[1:].cpu().numpy().transpose(0, 2, 3, 1).copy()

        # Save the rendered npy
        np.save(f"{rendered_npy_path}/{img_name.split('.')[0]}_input.npy", input_rendered[0])
        for i in range(relit_rendered.shape[0]):
            np.save(f"{rendered_npy_path}/{img_name.split('.')[0]}_{v['dst'].split('.')[0]}_f{i+1}_relit.npy", relit_rendered[i])
