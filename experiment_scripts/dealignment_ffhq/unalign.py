import bz2
import os
import glob
import cv2
import copy
import os.path as osp
import sys
from multiprocessing import Pool

import dlib
import numpy as np
import PIL.Image
import PIL.ImageDraw
import requests
import scipy.ndimage
from tqdm import tqdm
from argparse import ArgumentParser

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

def create_gaussian_pyramid(img, level=5):
  pys = [img]
  for i in range(level-1):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_LINEAR)
    pys.append(img)
  return pys

def create_laplacian_pyramid(img, level=5):
  pys = create_gaussian_pyramid(img, level)
  for i in range(level-1):
    pys[i] = pys[i] - cv2.resize(pys[i+1], (pys[i].shape[1], pys[i].shape[0]))
  return pys

def laplacian_blending(imgs, masks, level):
  summask = masks[0] + masks[1] + 1e-10
  img_lp = [None] * 2
  mask_lp = [None] * 2

  for i in range(2):
    img_lp[i] = create_laplacian_pyramid(imgs[i], level)
    mask_lp[i] = create_gaussian_pyramid(masks[i] / summask, level)

  output_lp = []
  for i in range(len(img_lp[0])):
    output_lp.append((img_lp[0][i] * mask_lp[0][i] + img_lp[1][i] * mask_lp[1][i]))

  output_lp = output_lp[::-1]
  prev_lvl = output_lp[0]
  for idx in range(len(output_lp)-1):
    prev_lvl = cv2.resize(prev_lvl, dsize=(output_lp[idx+1].shape[1], output_lp[idx+1].shape[0]))
    prev_lvl += output_lp[idx+1]

  return prev_lvl


def image_align(src_file,
                dst_file,
                relit_file,
                composite_file,
                compare_file,
                face_landmarks,
                output_size=1024,
                transform_size=4096,
                enable_padding=True):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    lm = np.array(face_landmarks)
    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    # print("Eye left : ", eye_left)
    # print("Eye right : ", eye_right)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    do_pad = False
    do_crop = False
    do_shrink = False
    # Choose oriented crop rectangle.
    # print("Eye to mouth : ", eye_to_mouth)
    # print("Eye to mouth (np.flipud) : ", np.flipud(eye_to_mouth))
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)   # Unpacking x; e.g. x = [1, 2], *x = 1, 2
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print(
            '\nCannot find source image. Please run "--wilds" before "--align".'
        )
        return
    img = PIL.Image.open(src_file)
    img = img.convert('RGB')
    original_img = copy.deepcopy(img)
    original_img.save('./out/gg.png', 'PNG')
    print("Original image : ", original_img.size)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        do_shrink = True
        print("[#] Performing shrink...")
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        print("Shrinking : ", img.size)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    
    # print("Border : ", border)
    # crop is [x1, y1, x2, y2]
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    
    # print("Crop_1 : ", crop)
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    
    # print("Crop_2 : ", crop)
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        do_crop = True
        print("[#] Performing crop...")
        img = img.crop(crop)
        img.save('./out/after_crop.png', 'PNG')
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    # print("Pad : ", pad)
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        do_pad = True
        print("[#] Perform padding...")
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                  'RGB')
        quad += pad[:2]

    # Transform.
    # print(quad)
    # print(quad.shape)
    # print("ORIGINAL IMAGE SIZE : ", original_img.size)
    # print(np.array(quad).flatten())
    quad_tmp = np.concatenate((quad, quad[0:1, :]), axis=0)
    if do_pad:
        quad_tmp -= pad[:2]
    if do_crop:
        quad_tmp += crop[0:2]
    if do_shrink:
        quad_tmp *= shrink
    
    # Drawing
    # draw = PIL.ImageDraw.Draw(original_img)
    # draw.line([tuple(q) for q in quad_tmp], fill='red', width=10)
    # original_img.save('./out/apply_marker.png', 'PNG')
    
    # img.save('./out/before_transf.png', 'PNG')
    # print(np.array(img).shape)
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    # print("G", np.array(img).shape)
    
    # Reverse back
    r_img = PIL.Image.open(relit_file)
    r_img = r_img.convert('RGB')
    # print(r_img.size)
    r_img = r_img.resize((transform_size, transform_size), PIL.Image.ANTIALIAS)
    # r_img.save('./out/r_img_resize.png', 'PNG')
    # print(r_img.size)
    transf_coor = np.array([[0, 0], 
                            [0, transform_size], 
                            [transform_size, transform_size], 
                            [transform_size, 0]]).astype(np.float32)
    inv_quad = cv2.getPerspectiveTransform(np.array(transf_coor), 
                                           np.array(quad_tmp[:-1]).astype(np.float32), 
                                           )
    inv_transformed = cv2.warpPerspective(np.array(r_img), inv_quad, original_img.size)
    inv_transformed = PIL.Image.fromarray(inv_transformed)
    inv_transformed.save('./out/inverse_transf_2.png', 'PNG')
    
    mask = np.mean(np.array(inv_transformed), axis=-1)
    mask = (mask == 0)
    mask = cv2.dilate((mask).astype(np.uint8), np.ones((3, 3)).astype(np.uint8), iterations=10)
    center = (int(np.array(original_img).shape[1]/2), int(np.array(original_img).shape[0]/2))
    im_clone = cv2.seamlessClone(np.array(original_img), np.array(inv_transformed), mask*255, center, cv2.NORMAL_CLONE)
    
    im_clone = PIL.Image.fromarray(im_clone)
    im_clone.save('./im_clone.png', 'PNG')
    # exit()
    mask = np.stack([mask]*3, axis=-1)
    
    
    # mask_save = PIL.Image.fromarray(np.clip(mask*255, 0, 255).astype(np.uint8), 'RGB')
    # print(np.max(np.array(mask)), np.min(np.array(mask)), np.unique(mask))
    # mask_save.save('./out/masky_dilate2.png', 'PNG')
    
    # mask_dilate = cv2.dilate(np.array(mask).astype(np.uint8), kernel=(5, 5), iterations=10)
    # mask_dilate = cv2.GaussianBlur(mask_dilate, (3,3), 0)
    # ret, mask_dilate = cv2.threshold(mask_dilate, 127, 255, cv2.THRESH_BINARY)
    # PIL.Image.fromarray(mask_dilate).save('./out/mask_dilate.png', 'PNG')
    
    # inv_transformed = cv2.warpPerspective(np.array(r_img), inv_quad, (960, 960))
    # convert the resulting image back to a PIL image
    # inv_transformed = PIL.Image.fromarray(inv_transformed)
    # inv_transformed.save('./out/inverse_transf.png', 'PNG')
    
    # Place back to original image
    composite = np.array(im_clone)
    # composite = ((1-mask) * np.array(inv_transformed)) + (mask * np.array(original_img))
    
    PIL.Image.fromarray(composite).save('./composite_img.png', 'PNG')
    PIL.Image.fromarray(composite).save(composite_file, 'PNG')
    
    blended_out = laplacian_blending(imgs=[np.array(original_img)/255.0, composite/255.0], masks=[mask*1.0, 1.0-mask], level=9)
    PIL.Image.fromarray(np.clip(blended_out*255.0, 0, 255).astype(np.uint8)).save('./out/lap_blend_seam_Queen.png', 'PNG')
    
    compare = np.concatenate((original_img, np.clip(blended_out*255.0, 0, 255).astype(np.uint8)), axis=0)
    PIL.Image.fromarray(compare).save(compare_file, 'PNG')
    
    original_img.save('org.png', 'PNG')
    inv_transformed.save('inv_transf.png', 'PNG')
    
    
    # img.save('./out/after_transf.png', 'PNG')
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    img.save(dst_file, 'PNG')


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector(
        )  # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [
                (item.x, item.y)
                for item in self.shape_predictor(img, detection).parts()
            ]
            yield face_landmarks


def unpack_bz2(src_path):
    dst_path = src_path[:-4]
    if os.path.exists(dst_path):
        print(f'[#] File \"{dst_path}\" : cached')
        return dst_path
    data = bz2.BZ2File(src_path).read()
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def work_landmark(raw_img_path, img_name, face_landmarks):
    face_img_name = '%s.png' % (os.path.splitext(img_name)[0], )
    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
    composite_face_path = os.path.join(COMPOSITE_IMAGES_DIR, face_img_name)
    compare_face_path = os.path.join(COMPARE_IMAGES_DIR, face_img_name)
    relit_face_path = os.path.join(RELIT_IMAGES_DIR, f'res_{face_img_name}')
    
    print("Aligning : ", aligned_face_path)
    print("Relit : ", relit_face_path)
    # if os.path.exists(aligned_face_path):
    #     return
    image_align(raw_img_path,
                aligned_face_path,
                relit_face_path,
                composite_face_path,
                compare_face_path,
                face_landmarks,
                output_size=256)


def get_file(src, tgt):
    if os.path.exists(tgt):
        print(f'[#] File \"{tgt}\" : cached')
        return tgt
    tgt_dir = os.path.dirname(tgt)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    file = requests.get(src)
    open(tgt, 'wb').write(file.content)
    return tgt


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input_imgs_path",
                        type=str,
                        default="imgs",
                        help="input images directory path")
    parser.add_argument("-o",
                        "--output_imgs_path",
                        type=str,
                        default="imgs_align",
                        help="output images directory path")
    parser.add_argument("-c",
                        "--composite_imgs_path",
                        type=str,
                        default="imgs_align",
                        help="composite images directory path")
    parser.add_argument("-cmp",
                        "--compare_imgs_path",
                        type=str,
                        default="imgs_align",
                        help="compare images directory path")
    parser.add_argument("-r",
                        "--relit_imgs_path",
                        type=str,
                        help="relit images directory path")
    parser.add_argument("-o_ext",
                        "--output_ext",
                        type=str,
                        default=None,
                        help="output images extension")

    args = parser.parse_args()

    # takes very long time  ...
    landmarks_model_path = unpack_bz2(
        get_file(
            'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'temp/shape_predictor_68_face_landmarks.dat.bz2'))

    # RAW_IMAGES_DIR = sys.argv[1]
    # ALIGNED_IMAGES_DIR = sys.argv[2]
    RAW_IMAGES_DIR = args.input_imgs_path
    RELIT_IMAGES_DIR = args.relit_imgs_path
    ALIGNED_IMAGES_DIR = args.output_imgs_path
    COMPOSITE_IMAGES_DIR = args.composite_imgs_path
    COMPARE_IMAGES_DIR = args.compare_imgs_path

    if not osp.exists(ALIGNED_IMAGES_DIR): os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
    if not osp.exists(COMPOSITE_IMAGES_DIR): os.makedirs(COMPOSITE_IMAGES_DIR, exist_ok=True)
    if not osp.exists(COMPARE_IMAGES_DIR): os.makedirs(COMPARE_IMAGES_DIR, exist_ok=True)

    # files = sorted(os.listdir(RELIT_IMAGES_DIR))
    files = sorted(glob.glob(f'{RELIT_IMAGES_DIR}/res_f*'))
    files = [f.split('_')[-1] for f in files]
    print(f'[#] Total img files {len(files)}')
    
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in tqdm(files):
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        # raw_img_path = os.path.join(RELIT_IMAGES_DIR, img_name)
        # continue
        for i, face_landmarks in enumerate(
                landmarks_detector.get_landmarks(raw_img_path),
                start=1):
            # assert i == 1, f'{i}'
            # print(i, face_landmarks)
            # face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            # aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            # image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=256)

            work_landmark(raw_img_path, img_name, face_landmarks)
                # progress.update()

                    # job = pool.apply_async(
                    #     work_landmark,
                    #     (raw_img_path, img_name, face_landmarks),
                    #     callback=cb,
                    #     error_callback=err_cb,
                    # )
                    # res.append(job)

            # pool.close()
            # pool.join()
    print(f"output aligned images at: {ALIGNED_IMAGES_DIR}")
