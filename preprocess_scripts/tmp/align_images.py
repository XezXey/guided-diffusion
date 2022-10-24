import os
import sys
import bz2
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from tqdm import tqdm
from multiprocessing import Pool

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def work_landmark(raw_img_path, img_name, face_landmarks):
    face_img_name = '%s.png' % (os.path.splitext(img_name)[0], )
    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
    if os.path.exists(aligned_face_path):
        return
    image_align(raw_img_path,
                aligned_face_path,
                face_landmarks,
                output_size=256)


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    landmarks_model_path = unpack_bz2(
        get_file('shape_predictor_68_face_landmarks.dat.bz2',
                 LANDMARKS_MODEL_URL,
                 cache_subdir='temp'))

    RAW_IMAGES_DIR = sys.argv[1]
    ALIGNED_IMAGES_DIR = sys.argv[2]
    # RAW_IMAGES_DIR = '../manipulate_wavy_hair_girl/unaligned'
    # ALIGNED_IMAGES_DIR = '../manipulate_wavy_hair_girl'

    files = os.listdir(RAW_IMAGES_DIR)
    with tqdm(total=len(files)) as progress:

        def cb(*args):
            # print('update')
            progress.update()

        def err_cb(e):
            print('error:', e)

        with Pool(8) as pool:
            res = []
            landmarks_detector = LandmarksDetector(landmarks_model_path)
            for img_name in files:
                raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
                # print('img_name:', img_name)
                for i, face_landmarks in enumerate(
                        landmarks_detector.get_landmarks(raw_img_path),
                        start=1):
                    # assert i == 1, f'{i}'
                    # print(i, face_landmarks)
                    # face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                    # aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    # image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=256)

                    work_landmark(raw_img_path, img_name, face_landmarks)
                    progress.update()

                    # job = pool.apply_async(
                    #     work_landmark,
                    #     (raw_img_path, img_name, face_landmarks),
                    #     callback=cb,
                    #     error_callback=err_cb,
                    # )
                    # res.append(job)

            # pool.close()
            # pool.join()
