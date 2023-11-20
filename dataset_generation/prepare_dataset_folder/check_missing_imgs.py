import numpy as np
import glob, os

src = "/data/mint/DPM_Dataset/Generated_Dataset/"
cmp = "/data/mint/DPM_Dataset/Generated_Dataset_deca_estim/"
folder = "/rendered_images/deca_masked_face_images_woclip/train/"

src += folder
cmp += folder

src_img = glob.glob(src + "*")
cmp_img = glob.glob(cmp + "*")

# '\n'.join(src_img)
print("[#] Total images in src: ", len(src_img))
print("[#] Total images in cmp: ", len(cmp_img))