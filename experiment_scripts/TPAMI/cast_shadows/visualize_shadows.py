import numpy as np
from flask import Flask, request, send_file, send_from_directory
import glob, os, re
import numpy as np
import json
import sys
import argparse
import pandas as pd
    
def progress(vid, mothership=False):
    model = 'log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml_tomin_steps=50'
    n_frames = 3
    ckpt = 'ema_085000'
    
    if mothership:
        if argsp.set_ == 'train':
            sampling_path = f'/data/mint/sampling/TPAMI/FFHQ_diffuse_face/{model}/ema_085000/train_sub'
        else:
            sampling_path = f'/data/mint/sampling/TPAMI/FFHQ_diffuse_face/{model}/ema_085000/{argsp.set_}'
        progress_path = sampling_path
    else:
        if argsp.set_ == 'train':
            progress_path = f'/data/mint/sampling/TPAMI/FFHQ_diffuse_face/mount_sampling/v{vid}/{model}/{ckpt}/train_sub/'
        else: 
            progress_path = f'/data/mint/sampling/TPAMI/FFHQ_diffuse_face/mount_sampling/v{vid}/{model}/{ckpt}/{argsp.set_}'
    if os.path.exists(progress_path):
        print(os.listdir(progress_path))
    else: 
        print(f"[#] {progress_path} not found!")
        return []
    
    img_path = []
    for p in sorted(os.listdir(progress_path)):
        count = 0
        if argsp.set_ == 'train':
            start = int(p.split('_')[1])
            end = int(p.split('_')[3])
            n = end - start
            tail = f'{progress_path}/{p}/shadow/reverse_sampling/'
        elif argsp.set_ == 'valid':
            assert p == 'shadow'
            n = 1
            tail = f'{progress_path}/shadow/reverse_sampling/'
            start = '60000'
        else: raise NotImplementedError(f"Set: {argsp.set_} is not found!")
        for t in sorted(os.listdir(tail)):
            tmp = f'{tail}/{t}/dst={start}.jpg/Lerp_1000/n_frames={n_frames}/'
            assert len(os.listdir(tmp)) == n_frames * 2 + 1
            count += 1
            img_path.append(tmp)
            
        if argsp.set_ == 'train':
            print(f'[#] {p} => {count}/{n} => {count * 100/n:.2f}%')
        else:
            print(f'[#] {p} => {count}')
    return img_path

parser = argparse.ArgumentParser()
parser.add_argument('--set_', default='train')
parser.add_argument('--port', required=True)
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--rank_shadow_c', action='store_true', default=False)
parser.add_argument('--rank_shadow_iou', action='store_true', default=False)
argsp = parser.parse_args()
# Run only once
run = True
if run:
    
    if argsp.rank_shadow_c:
        c = pd.read_csv(f'/data/mint/DPM_Dataset/ffhq_256_with_anno/params/{argsp.set_}/ffhq-{argsp.set_}-shadow-anno.txt', sep=' ', header=None, names=['image_name', 'c_val'])
        c_sorted = c.sort_values(by=['c_val'], ascending=False)
    if argsp.rank_shadow_iou:
        c = pd.read_csv('./iou.csv', sep=',', header=None, skiprows=1, names=['image_name', 'IOU'])
        c_sorted = c.sort_values(by=['IOU'], ascending=False)
            
    diffuse_img_path = progress(vid='11') + progress(vid='10', mothership=True) + progress(vid='9') + progress(vid='8')
    diffuse_img_path = {re.findall(r"src=\d+\.jpg", p)[0].split('=')[1].split('.')[0]: p for p in diffuse_img_path}
    print(f"[#] Total images ({argsp.set_}): ", len(diffuse_img_path))
    run = False
    
else: pass

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        sort = request.args.get('sort', 'asc')
        s = request.args.get('start', 0)
        e = request.args.get('end', 10)

        set_ = argsp.set_
        out = ""
        
        # for img_path in glob.glob(f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff/thres_1e-1/{set_}/*.png')[:500]:
            # img_name = img_path.split('/')[-1].split('.')[0]
        all_path = c_sorted['image_name'].values if sort == 'asc' else c_sorted['image_name'].values[::-1]
        for img_path in all_path[int(s):int(e)]:
            print(img_path)
            img_name = img_path.split('.')[0]
            
            path = '/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff/'
            to_show_path = []
            # for to_show in ['median5_5e-2', 'thres_1e-1', 'thres_5e-2', 'futschik', 'futschik_1e-1', 'futschik_2e-1']:
            for to_show in ['thres_5e-2', 'median5_5e-2', 'futschik', 'futschik_2e-1']:
                to_show_path.append(f'{path}/{to_show}/{set_}/{img_name}.png')
            
            value = c_sorted[c_sorted['image_name'] == f'{img_name}.jpg'].values[0][1]
            out += f"<h1> Image: {img_name} => {value} </h1>"
            # Raw image
            tp = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/{set_}/{img_name}.jpg'
            out += f"<img src=/files/{tp} width=\"256\" height=\"256\">"
            
            # Reshadow
            rs_path = diffuse_img_path[img_name]
            tp = f"{rs_path}/res_frame2.png"
            out += f"<img src=/files/{tp} width=\"256\" height=\"256\">"
            
            # Show the images in the same rows
            for tp in to_show_path:
                out += f"<img src=/files/{tp} width=\"256\" height=\"256\">"
                
            # Shadow masks
            tp = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_masks/{set_}/{img_name}.png'
            out += f"<img src=/files/{tp} width=\"256\" height=\"256\">"
            
            # Shadow masks for vis
            tp = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_masks_t5e-1_forvis/{set_}/{img_name}.png'
            out += f"<img src=/files/{tp} width=\"256\" height=\"256\">"
            
            out += "<br>"
                
        return out
    
    return app
    
if __name__ == "__main__":
     
    app = create_app()
    app.run(host=argsp.host, port=argsp.port, debug=True, threaded=False)