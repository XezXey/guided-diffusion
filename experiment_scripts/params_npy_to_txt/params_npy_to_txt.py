import numpy as np
import glob
import tqdm

out_txt = "/data/mint/Aom_Dataset/scene2/params/train/scene2-train-light-anno.txt"

fs = open(out_txt, "w")
for f in tqdm.tqdm(glob.glob("/data/mint/Aom_Dataset/scene2/params_npy/train/*.npy")):
    name = f.split("/")[-1].replace(".npy", ".png")
    params = np.load(f, allow_pickle=True)
    params = params.flatten()
    params_txt = " ".join([str(p) for p in params])
    fs.write(f"{name} {params_txt}" + "\n")
fs.close()