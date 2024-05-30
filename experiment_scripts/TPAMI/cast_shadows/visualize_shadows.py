import numpy as np
from flask import Flask, request, send_file, send_from_directory
import glob, os, re
import numpy as np
import json
import sys
import argparse

def progress(vid, mothership=False):
    model = 'log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml_tomin_steps=50'
    n_frames = 3
    ckpt = 'ema_085000'
    if mothership:
        sampling_path = f'/data/mint/sampling/TPAMI/FFHQ_diffuse_face/{model}/ema_085000/train_sub'
        progress_path = sampling_path
    else:
        progress_path = f'/data/mint/sampling/TPAMI/FFHQ_diffuse_face/mount_sampling/v{vid}/{model}/{ckpt}/train_sub/'
    print(os.listdir(progress_path))
    
    img_path = []
    for p in sorted(os.listdir(progress_path)):
        start = int(p.split('_')[1])
        end = int(p.split('_')[3])
        n = end - start
        tail = f'{progress_path}/{p}/shadow/reverse_sampling/'
        count = 0
        for t in sorted(os.listdir(tail)):
            tmp = f'{tail}/{t}/dst={start}.jpg/Lerp_1000/n_frames={n_frames}/'
            assert len(os.listdir(tmp)) == n_frames * 2 + 1
            count += 1
            img_path.append(tmp)
            
        print(f'[#] {p} => {count}/{n} => {count * 100/n:.2f}%')
    return img_path

diffuse_img_path = progress(vid='11') + progress(vid='10', mothership=True) + progress(vid='9') + progress(vid='8')
diffuse_img_path = {re.findall(r"src=\d+\.jpg", p)[0].split('=')[1].split('.')[0]: p for p in diffuse_img_path}
print("[#] Total images: ", len(diffuse_img_path))


def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        set_ = args.set_
        out = ""
        # for img_path in glob.glob(f'/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/{set_}/*.jpg')[:100]:
        #     img_name = img_path.split('/')[-1].split('.')[0]
        
        for img_path in glob.glob(f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff/thres_1e-1/{set_}/*.png')[:500]:
            img_name = img_path.split('/')[-1].split('.')[0]
            
            path = '/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff/'
            to_show_path = []
            for to_show in ['median5_5e-2', 'thres_1e-1', 'thres_5e-2', 'futschik', 'futschik_1e-1', 'futschik_2e-1']:
                to_show_path.append(f'{path}/{to_show}/{set_}/{img_name}.png')
            
            out += f"<h1> Image: {img_name} </h1>"
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
            
            out += "<br>"
                
        return out
    
    return app
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_', default='train')
    parser.add_argument('--res', default=128)
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
     
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)