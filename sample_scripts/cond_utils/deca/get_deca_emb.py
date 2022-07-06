from genericpath import exists
import os, sys
from charset_normalizer import detect
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch as th
from PIL import Image
import blobfile as bf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def get_deca_emb(device, data, dat_type, vis=True):
    defaults = dict(
        input_path="",
        savefolder="./",
        device=device,
        iscrop=True,
        detector='fan',
        useTex=False,
        saveVis=True,
        saveKpt=True,
        saveDepth=False,
        saveObj=False,
        saveMat=False,
        saveImages=False,
        prefix=''
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args(args=[])
    
    # if dat_type == 'img':
    #     data = datasets.TestDataFromPath(img_path, iscrop=args.iscrop, face_detector=args.detector, device=device)
    # elif img_list is not None:
    data = datasets.TestData(data_list=data, dat_type=dat_type, iscrop=args.iscrop, face_detector=args.detector, device=device)



    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca = DECA(config = deca_cfg, device=device)

    deca_params = {}
    deca_images = {}

    for i in tqdm(range(len(data))):
        name = data[i]['imagename']
        images = data[i]['image'].to(device)[None,...]
        with th.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict) #tensor
            print(codedict.keys())
            print(opdict.keys())
            print(visdict.keys())
            if vis:
                plt.title(f"Imgage name : {name}")
                plt.imshow(deca.visualize(visdict)[..., ::-1])
                plt.show()

        '''
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                if vis_name not in visdict.keys():
                    continue
                image  =util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
        '''

        # print("OPDICT")
        # for k in opdict.keys():
        #   print(f'{k} : {opdict[k].shape}, max = {th.max(opdict[k])}, min = {th.min(opdict[k])}')
        # print("CODEDICT")
        # for k in codedict.keys():
        #   print(f'{k} : {codedict[k].shape}, max = {th.max(codedict[k])}, min = {th.min(codedict[k])}')
        # print("VISDICT")
        # for k in visdict.keys():
        #   print(f'{k} : {visdict[k].shape}, max = {th.max(visdict[k])}, min = {th.min(visdict[k])}')
        
        deca_params[name] = {
            'shape':codedict['shape'].flatten().detach().cpu().numpy(), 
            'pose':codedict['pose'].flatten().detach().cpu().numpy(), 
            'exp':codedict['exp'].flatten().detach().cpu().numpy(), 
            'cam':codedict['cam'].flatten().detach().cpu().numpy(),
            'light':codedict['light'].flatten().detach().cpu().numpy(),
        }

        deca_images[name] = {
            'rendered_images' : opdict['rendered_images'], 
            'alpha_images' : opdict['alpha_images'], 
            'normal_images' : opdict['normal_images'], 
            'normals' : opdict['normals']
        }
    return deca_params, deca_images