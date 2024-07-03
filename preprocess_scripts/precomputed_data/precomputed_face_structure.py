import numpy as np
import torch as th
import argparse
import blobfile as bf
from PIL import Image
import cv2
import os, glob, sys, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data/mint/DPM_Dataset/ffhq_256_with_anno/face_segment_with_pupil/')
parser.add_argument('--set', type=str, default='train')
parser.add_argument('--segment_part', type=str, default=['hair', 'faceskin', 'eyes', 'pupils', 'glasses', 'ears', 'nose', 'inmouth', 'u_lip', 'l_lip', 'neck', 'cloth', 'hat'])
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--resolution', type=int, default=256)
args = parser.parse_args()

def load_image(path):
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    return pil_image

def face_segment_to_onehot(segment_part, img_path):
    
    seg_m = [face_segment(segment_part=f'faceseg_{p}', img_path=img_path) for p in segment_part]
    return seg_m

def face_segment(segment_part, img_path):
    face_segment_anno = load_image(img_path)
    
    face_segment_anno = np.array(face_segment_anno)
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
    l_pupil = (face_segment_anno == 19)
    r_pupil = (face_segment_anno == 20)
    face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, l_pupil, r_eye, r_pupil, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip))

    if segment_part == 'faceseg_face':
        seg_m = face
    elif segment_part == 'faceseg_head':
        seg_m = (face | neck | hair)
    elif segment_part == 'faceseg_nohead':
        seg_m = ~(face | neck | hair)
    elif segment_part == 'faceseg_hair':
        seg_m = hair
    elif segment_part == 'faceseg_eyes':
        seg_m = (l_eye | r_eye | l_pupil | r_pupil)
    elif segment_part == 'faceseg_pupils':
        seg_m = (l_pupil | r_pupil)
    elif segment_part == 'faceseg_ears':
        seg_m = (l_ear | r_ear | ear_r)
    elif segment_part == 'faceseg_nose':
        seg_m = nose
    elif segment_part == 'faceseg_mouth':
        seg_m = (mouth | u_lip | l_lip)
    elif segment_part == 'faceseg_u_lip':
        seg_m = u_lip
    elif segment_part == 'faceseg_l_lip':
        seg_m = l_lip
    elif segment_part == 'faceseg_inmouth':
        seg_m = mouth
    elif segment_part == 'faceseg_neck':
        seg_m = neck
    elif segment_part == 'faceseg_glasses':
        seg_m = eye_g
    elif segment_part == 'faceseg_cloth':
        seg_m = cloth
    elif segment_part == 'faceseg_hat':
        seg_m = hat
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
    elif segment_part == 'faceseg_faceskin':
        seg_m = skin
    elif segment_part == 'faceseg_faceskin&nose':
        seg_m = (skin | nose)
    elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows':
        seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye)    
    elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses':
        seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye | l_pupil | r_pupil | eye_g)
    elif segment_part == 'faceseg_eyes&glasses':
        seg_m = (l_eye | r_eye | eye_g | l_pupil | r_pupil)
    elif segment_part == 'faceseg_face_noglasses':
        seg_m = (~eye_g & face)
    elif segment_part == 'faceseg_face_noglasses_noeyes':
        seg_m = (~(l_eye | r_eye) & ~eye_g & face)
    elif segment_part in ['sobel_bg_mask', 'laplacian_bg_mask', 'sobel_bin_bg_mask']:
        seg_m = ~(face | neck | hair)
    elif segment_part in ['canny_edge_bg_mask']:
        seg_m = ~(face | neck | hair) | (l_ear | r_ear)
    else: raise NotImplementedError(f"Segment part: {segment_part} is not found!")
    
    out = seg_m
    return out

def get_face_structure(img_path, resolution):

    parts = face_segment_to_onehot(segment_part=args.segment_part, img_path=img_path)
    fs = []
    for pi in parts:
        tmp_fs = pi
        if resolution != 256:
            tmp_fs = cv2.resize(tmp_fs.astype(np.uint8), (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        else:
            tmp_fs = tmp_fs.astype(np.uint8)
        assert np.allclose(tmp_fs[..., 0], tmp_fs[..., 1]) and np.allclose(tmp_fs[..., 0], tmp_fs[..., 2])
        tmp_fs = tmp_fs[..., 0:1]
        fs.append(tmp_fs)
    fs = np.concatenate(fs, axis=-1)  # Concatenate the face structure parts into n-channels(parts)
    return fs


if __name__ == '__main__':
    imgs = glob.glob(f'{args.data_dir}/{args.set}/anno/anno_*.png')
    save_dir = f'{args.save_dir}/{args.set}/anno/'
    os.makedirs(save_dir, exist_ok=True)
    os.system(f"echo {' '.join(args.segment_part)} > {args.save_dir}/segment_parts_{args.set}.txt")
    for img in tqdm.tqdm(imgs):
        img_name = img.split('/')[-1].split('.')[0]
        out = get_face_structure(img, resolution=args.resolution)
        np.save(f'{save_dir}/{img_name}.npy', out)


