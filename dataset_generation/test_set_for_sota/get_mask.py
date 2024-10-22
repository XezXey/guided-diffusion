import numpy as np
import torch as th
import glob
import os, tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sota_data_path', type=str, default="/data/mint/DPM_Dataset/Dataset_For_Baseline/ffhq/")
parser.add_argument('--proc_dataset_path', type=str, default="/data/mint/DPM_Dataset/ffhq_256_with_anno/")
parser.add_argument('--dataset_name', type=str, default='ffhq')
parser.add_argument('--set_', type=str, default='valid')
parser.add_argument('--n_step', type=str, default=2)
args = parser.parse_args()

def face_segment(segment_part, img):
    
    if isinstance(img, Image.Image):
        face_segment_anno = np.array(img)
    else:
        face_segment_anno = img
        
    bg = (face_segment_anno == 0)
    skin = (face_segment_anno == 1)
    l_brow = (face_segment_anno == 2)
    r_brow = (face_segment_anno == 3)
    l_eye = (face_segment_anno == 4)
    r_eye = (face_segment_anno == 5)
    eye_g = (face_segment_anno == 6)
    l_ear = (face_segment_anno == 7)
    r_ear = (face_segment_anno == 8)
    ear_r = (face_segment_anno == 9)
    nose = (face_segment_anno == 10)
    mouth = (face_segment_anno == 11)
    u_lip = (face_segment_anno == 12)
    l_lip = (face_segment_anno == 13)
    neck = (face_segment_anno == 14)
    neck_l = (face_segment_anno == 15)
    cloth = (face_segment_anno == 16)
    hair = (face_segment_anno == 17)
    hat = (face_segment_anno == 18)
    face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip))
    foreground = face_segment_anno != 0

    if segment_part == 'faceseg_face':
        seg_m = face
    elif segment_part == 'faceseg_foreground':
        seg_m = foreground
    elif segment_part == 'faceseg_head':
        seg_m = (face | neck | hair)
    elif segment_part == 'faceseg_nohead':
        seg_m = ~(face | neck | hair)
    elif segment_part == 'faceseg_face&hair':
        seg_m = ~bg
    elif segment_part == 'faceseg_bg_noface&nohair':
        seg_m = (bg | hat | neck | neck_l | cloth) 
    elif segment_part == 'faceseg_bg&ears_noface&nohair':
        seg_m = (bg | hat | neck | neck_l | cloth) | (l_ear | r_ear | ear_r)
    elif segment_part == 'faceseg_bg':
        seg_m = bg
    elif segment_part == 'faceseg_bg&noface':
        seg_m = (bg | hair | hat | neck | neck_l | cloth)
    elif segment_part == 'faceseg_hair':
        seg_m = hair
    elif segment_part == 'faceseg_faceskin':
        seg_m = skin
    elif segment_part == 'faceseg_faceskin&nose':
        seg_m = (skin | nose)
    elif segment_part == 'faceseg_eyes&glasses&mouth&eyebrows':
        seg_m = (l_eye | r_eye | eye_g | l_brow | r_brow | mouth)
    elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows':
        seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye)
    elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses':
        seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye | eye_g)
    elif segment_part == 'faceseg_face_noglasses':
        seg_m = (~eye_g & face)
    elif segment_part == 'faceseg_face_noglasses_noeyes':
        seg_m = (~(l_eye | r_eye) & ~eye_g & face)
    elif segment_part == 'faceseg_eyes&glasses':
        seg_m = (l_eye | r_eye | eye_g)
    elif segment_part == 'glasses':
        seg_m = eye_g
    elif segment_part == 'faceseg_eyes':
        seg_m = (l_eye | r_eye)
    # elif (segment_part == 'sobel_bg_mask') or (segment_part == 'laplacian_bg_mask') or (segment_part == 'sobel_bin_bg_mask'):
    elif segment_part in ['sobel_bg_mask', 'laplacian_bg_mask', 'sobel_bin_bg_mask']:
        seg_m = ~(face | neck | hair)
    elif segment_part in ['canny_edge_bg_mask']:
        seg_m = ~(face | neck | hair) | (l_ear | r_ear)
    else: raise NotImplementedError(f"Segment part: {segment_part} is not found!")
    
    out = seg_m
    return out

if __name__ == '__main__':
    
    for p in tqdm.tqdm(glob.glob(f'{args.sota_data_path}/{args.set_}/*')):
        name = p.split('/')[-1]
        if args.dataset_name == 'ffhq':
            pair_id, src_id, dst_id = name.split('_')
        elif args.dataset_name in ['mp_test', 'mp_test2']:
            # pair1_src=001_01_01_051_06.png_dst=001_01_01_051_08.png to [pair_1, src=001_01_01_051_06.png, dst=001_01_01_051_08.png]
            pair_id = name.split('_src')[0]
            src_id = 'src=' + name.split('_src=')[-1].split('_dst')[0]
            dst_id = 'dst=' + name.split('_dst=')[-1].split('.png')[0] + '.png'
            
        src_sj_name = src_id.split('.')[0].split('=')[-1]
        faceseg_dir = f'{args.proc_dataset_path}/face_segment_with_pupil/{args.set_}/anno/anno_{src_sj_name}.png'
        faceseg = np.array(Image.open(faceseg_dir))
        foreground = face_segment('faceseg_foreground', faceseg)
        if os.path.exists(f'{p}/n_step={args.n_step}') == False:
            raise FileNotFoundError(f'{p}/n_step={args.n_step} is not found!')
        Image.fromarray(foreground.astype(np.uint8)*255).save(f'{p}/n_step={args.n_step}/{src_id}_mask.png')