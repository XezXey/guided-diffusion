import numpy as np
import glob, os, sys
import argparse, re

parser = argparse.ArgumentParser()
parser.add_argument('--genparams_path', required=True)
parser.add_argument('--out_path', required=True)
args = parser.parse_args()

params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'tform', 'albedo', 'detail', 'shadow']

all_files = glob.glob(os.path.join(args.genparams_path, '*'))

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path, exist_ok=True)
    
pattern = r'\b(\d+-\d+)\b'

for p in params_key:
    print("="*100)
    print("[#] Processing: ", p)
    files = [x for x in all_files if p in x]
    index_anno = [re.search(pattern, x).group(1) for x in files]
    print("[#] Indexing of: ", index_anno)
    out = f"ffhq-train-{p}-anno.txt"
    dat_lines = ''
    sj_dict = {}
    for f in files:
        with open(f, 'r') as fp:
            lines = fp.readlines()
        for l in lines:
            sj_k = l.split(' ')[0]
            if (sj_k in sj_dict.keys()) and (l != sj_dict[sj_k]['dat']):
                # print(l, sj_dict[sj_k], l == sj_dict[sj_k])
                assert False
            else:
                sj_dict[sj_k] = {}
            sj_dict[sj_k]['dat'] = l
            sj_dict[sj_k]['sep_dat'] = ' '.join(l.split(' ')[1:])
            if '_relit' in sj_k:
                sj_dict[sj_k]['sj_name'] = sj_k.replace('_relit', '')
            elif '_input' in sj_k:
                sj_dict[sj_k]['sj_name'] = sj_k.replace('_input', '')
                
    print(f"[#] #N of subjects: {len(sj_dict.keys())}")
    dat_lines = ''.join([f"{v['sj_name']} {v['sep_dat']}" for _, v in sj_dict.items()])
    with open(os.path.join(args.out_path, out), 'w') as fp:
        fp.write(dat_lines)
    print("="*100)
    # exit()