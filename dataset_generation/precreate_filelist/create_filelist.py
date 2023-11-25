import numpy as np
import pickle
import blobfile as bf
import argparse
import time, glob

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_prefix_name', type=str, default='raw')
parser.add_argument('--use_recursive', action='store_true', default=False)
parser.add_argument('--file_ext', type=str, required=True)

def _list_image_files_recursively_separate(data_dir):
    input_results = []
    relit_results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            if 'input' in entry:
                input_results.append(full_path)
            elif 'relit' in entry:
                relit_results.append(full_path)
        elif bf.isdir(full_path):
            input_results_rec, relit_results_rec = _list_image_files_recursively_separate(full_path)
            input_results.extend(input_results_rec)
            relit_results.extend(relit_results_rec)
    return input_results, relit_results

if __name__ == '__main__':
    args = parser.parse_args()
    if args.file_ext[0] != '.':
        args.file_ext = '.' + args.file_ext
        
    tstart = time.time()
    if args.use_recursive:
        input_results, relit_results = _list_image_files_recursively_separate(args.data_dir)
    else: 
        # mint = time.time()
        input_results = [f for f in glob.iglob(f'{args.data_dir}/*{args.file_ext}') if f'input' in f]
        relit_results = [f for f in glob.iglob(f'{args.data_dir}/*{args.file_ext}') if f'relit' in f]
        # print(f"[#] Time taken to glob: {time.time() - mint:.2f} sec")
        # print(len(gg), gg[:10])
        # exit()
    # print("[#] Before save to .pkl: ", input_results, relit_results)
    
    with open(f'{args.data_dir}/{args.output_prefix_name}_input_results.pkl', 'wb') as f:
        pickle.dump(input_results, f)
    
    with open(f'{args.data_dir}/{args.output_prefix_name}_relit_results.pkl', 'wb') as f:
        pickle.dump(relit_results, f)
    
    print(f"[#] Time taken: {time.time() - tstart:.2f} sec")
    print(f"[#] Total input images: {len(input_results)}")
    print(f"[#] Total relit images: {len(relit_results)}")