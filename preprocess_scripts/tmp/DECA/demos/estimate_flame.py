# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

from genericpath import exists
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    data = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca = DECA(config = deca_cfg, device=device)
    
    fo_shape = open(f"/{args.savefolder}/ffhq-{args.prefix}-shape-anno.txt", "w")
    fo_exp = open(f"/{args.savefolder}/ffhq-{args.prefix}-exp-anno.txt", "w")
    fo_pose = open(f"/{args.savefolder}/ffhq-{args.prefix}-pose-anno.txt", "w")
    fo_light = open(f"/{args.savefolder}/ffhq-{args.prefix}-light-anno.txt", "w")
    fo_cam = open(f"/{args.savefolder}/ffhq-{args.prefix}-cam-anno.txt", "w")
    fo_detail = open(f"/{args.savefolder}/ffhq-{args.prefix}-detail-anno.txt", "w")

    fo_dict = {'shape':fo_shape, 'exp':fo_exp, 'pose':fo_pose, 
            'light':fo_light, 'cam':fo_cam, 'detail':fo_detail}

    os.makedirs(f"{args.savefolder}/displacement_map/", exist_ok=True)
    os.makedirs(f"{args.savefolder}/uv_detail_normals/", exist_ok=True)
    os.makedirs(f"{args.savefolder}/uv_texture_gt/", exist_ok=True)


    for i in tqdm(range(len(data))):
        name = data[i]['imagename']
        images = data[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict) #tensor
            deca.visualize(visdict)

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
          # print(f'{k} : {opdict[k].shape}, max = {torch.max(opdict[k])}, min = {torch.min(opdict[k])}')
        # print("CODEDICT")
        # for k in codedict.keys():
          # print(f'{k} : {codedict[k].shape}, max = {torch.max(codedict[k])}, min = {torch.min(codedict[k])}')
        # print("VISDICT")
        # for k in visdict.keys():
          # print(f'{k} : {visdict[k].shape}, max = {torch.max(visdict[k])}, min = {torch.min(visdict[k])}')
        # exit()
        
        # Params according to fo_dict
        for k, fo in fo_dict.items():
            a = codedict[k].cpu().numpy().flatten()
            fo.write(name + ".jpg ")
            fo.write(" ".join([str(x) for x in a]) + "\n")

        # uv_detail_normal
        uv_detail_normal = opdict['uv_detail_normals'][0].permute((1, 2, 0)).cpu().numpy()
        # uv_detail_normal = (uv_detail_normal + 1) * 127.5
        # print(np.max(uv_detail_normal), np.min(uv_detail_normal))
        # uv_detail_normal = ((uv_detail_normal / 127.5) - 1)
        # print(np.max(uv_detail_normal), np.min(uv_detail_normal))
        # exit()
        cv2.imwrite(img=(uv_detail_normal + 1.0) * 127.5, filename=f"{args.savefolder}/uv_detail_normals/uv_detail_normals_{name}.png")
        np.save(arr=uv_detail_normal, file=f"{args.savefolder}/uv_detail_normals/uv_detail_normals_{name}.npy")
        # displacement map
        displacement_map = np.squeeze(opdict['displacement_map'][0].permute((1, 2, 0)).cpu().numpy(), axis=-1)
        np.save(arr=displacement_map, file=f"{args.savefolder}/displacement_map/displacement_map_{name}.npy")
        cv2.imwrite(img=(displacement_map + 1.0) * 127.5, filename=f"{args.savefolder}/displacement_map/displacement_map_{name}.png")
        # uv_texture_gt
        uv_texture_gt = np.squeeze(opdict['uv_texture_gt'][0].permute((1, 2, 0)).cpu().numpy())
        np.save(arr=uv_texture_gt, file=f"{args.savefolder}/uv_texture_gt/uv_texture_gt_{name}.npy")
        cv2.imwrite(img=uv_texture_gt, filename=f"{args.savefolder}/uv_texture_gt/uv_texture_gt_{name}.png")


    for k, fo in fo_dict.items():
        fo.close()
        
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    parser.add_argument('--prefix', type=str, required=True,
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())
