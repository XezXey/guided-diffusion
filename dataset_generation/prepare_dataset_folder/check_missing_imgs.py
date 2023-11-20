import numpy as np
import glob, os

gen = "/data/mint/DPM_Dataset/Generated_Dataset/" + "/images/train/"
src = "/data/mint/DPM_Dataset/Generated_Dataset/"
fsrc = "/rendered_images/deca_masked_face_woclip/train/"
cmp = "/data/mint/DPM_Dataset/Generated_Dataset_deca_estim/"
fcmp = "/rendered_images/deca_masked_face_images_woclip/train/"

src += fsrc
cmp += fcmp

os.chdir(gen)
gen_img = sorted(glob.glob("./*"))
os.chdir(src)
src_img = glob.glob("./*")
os.chdir(cmp)
cmp_img = glob.glob("./*")

print("[#] Total images in src: ", len(src_img))
print("[#] Total images in cmp: ", len(cmp_img))

out = list(set(src_img) - set(cmp_img))
print("[#] #N Missing files from source: ", len(out))
print("[#] Missing files from source: ", out)
print("[#] Index in source: ", [gen_img.index(x.replace('.npy', '.png')) for x in out])
print(f"[#] For running: {' '.join([str(gen_img.index(x.replace('.npy', '.png'))) for x in out])}")
