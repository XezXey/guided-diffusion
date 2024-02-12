import numpy as np
import glob, os
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, required=True)
args = parser.parse_args()

path = '/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/faster_inference_script/sample_json/paper_multipie/multipie_testset2.json'
with open(path, 'r') as f:
    data = json.load(f)['pair']
 
for p in glob.glob(args.pred_path + '/*'):   
    if os.path.isdir(p):
        os.makedirs(f"{p}/out_transf_eval/", exist_ok=True)
        for k, v in data.items():
            # Old filename: input=346_03_01_051_06.png#pred=090_01_01_051_03.png.png
            
            print(k, v)
            pid = k
            src = v['src']
            ref = v['dst']
            gt = v['gt']
            
            
            fn = f"input={src}#pred={ref}.png"
            fn_fp = f"{p}/out/{fn}"
            if not os.path.exists(fn_fp):
                print(f"Missing: {fn}, ID: {pid}, path: {fn_fp}")
                continue
            else:
                # new_fn = f'input={src}#ref={ref}#gt={gt}.png'
                new_fn = f'input={src}#ref={ref}#pred={gt}.png'
                new_fn_fp = f"{p}/out_transf_eval/{new_fn}"
                os.system(f"cp {fn_fp} {new_fn_fp}")
                
                
                
            
            
            