import numpy as np
import glob, os

src = "/data/mint/DPM_Dataset/Generated_Dataset/"
cmp = "/data/mint/DPM_Dataset/Generated_Dataset_deca_estim/"
fsrc = "/rendered_images/deca_masked_face_woclip/train/"
fcmp = "/rendered_images/deca_masked_face_images_woclip/train/"

src += fsrc
cmp += fcmp

os.chdir(src)
src_img = glob.glob("./*")
os.chdir(cmp)
cmp_img = glob.glob("./*")

print("[#] Total images in src: ", len(src_img))
print("[#] Total images in cmp: ", len(cmp_img))

out = set(src_img) - set(cmp_img)
print("[#] #N Missing files from source: ", len(out))
print("[#] Missing files from source: ", out)
print("[#] Index in source: ", sorted([src_img.index(x) for x in list(out)]))
