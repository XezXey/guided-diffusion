import numpy as np
import glob
import tqdm
import multiprocessing as mp

name = 'joker_2'
alpha = 0.012
ratio = 0.75
minWidth = 20.0
flows_files = glob.glob(f'/data/mint/OptFlows/{name}/{alpha}alpha_{ratio}ratio_{minWidth}minWidth/all/*.npy')
flows_dict = {}

def load_npy_file(filename):
    arr = np.load(filename, allow_pickle=True).item()
    print(f"[#] Done ===> Processed {filename} in process {mp.current_process().name}")
    print(arr.keys())
    return arr

pool = mp.Pool(processes=len(flows_files))

# load the npy files in parallel using the worker processes
output = pool.starmap(load_npy_file, [(filename,) for filename in flows_files])
# close the worker pool
pool.close()
pool.join()

flow_dict = {}
for out in output:
    flow_dict.update(out)

print(len(flows_dict.keys()))
np.save(file=f"/data/mint/OptFlows/{name}/{alpha}alpha_{ratio}ratio_{minWidth}minWidth/all/{name}_flows.npy", arr=flows_dict)