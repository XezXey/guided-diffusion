import numpy as np
import glob, os
import tqdm
import multiprocessing as mp

name = 'lisa_1'
alpha = 0.012
ratio = 0.75
minWidth = 20.0

def load_npy_file(filename):
    arr = np.load(filename, allow_pickle=True).item()
    print(f"[#] Done ===> Processed {filename} in process {mp.current_process().name}")
    for k in arr.keys():
        np.save(file=f'{out_dir}/{k}.npy', arr={k:arr[k]})
        
    del arr

flows_files = glob.glob(f'/data/mint/OptFlows/{name}/{alpha}alpha_{ratio}ratio_{minWidth}minWidth/all/*.npy')

out_dir = f"/data/mint/OptFlows/{name}/{alpha}alpha_{ratio}ratio_{minWidth}minWidth/sep/"
os.makedirs(out_dir, exist_ok=True)

flows_dict = {}


pool = mp.Pool(processes=len(flows_files))

# load the npy files in parallel using the worker processes
output = pool.starmap(load_npy_file, [(filename,) for filename in flows_files])
# close the worker pool
pool.close()
pool.join()
