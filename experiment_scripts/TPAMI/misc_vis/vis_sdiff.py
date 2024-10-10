import numpy as np
from flask import Flask, request, send_file, send_from_directory
import glob, os, re
import numpy as np
import json
import sys
import argparse
import pandas as pd
    
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
    run = False

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        sort = request.args.get('sort', 'asc')
        s = request.args.get('s', 0)
        e = request.args.get('e', 10)

        set_ = argsp.set_
        out = ""
        if sort == 'asc':
            all_path = c_sorted['image_name'].values 
        elif sort == 'desc':
            all_path = c_sorted['image_name'].values[::-1]
        else:
            # Alphabetical
            all_path = sorted(glob.glob(f'/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/{set_}/*.jpg'))
            all_path = [os.path.basename(x) for x in all_path]
            
        for img_path in all_path[int(s):int(e)]:
            print(img_path)
            img_name = img_path.split('.')[0]
            
            value = c_sorted[c_sorted['image_name'] == f'{img_name}.jpg'].values[0][1]
            out += f"<h1> Image: {img_name} => {value} </h1>"
            # Raw image
            tp = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/{set_}/{img_name}.jpg'
            out += f"<img src=/files/{tp} width=\"256\" height=\"256\">"
            
            # Shadow masks from ray-tracing
            tp = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_masks/{set_}/{img_name}.png'
            out += f"<img src=/files/{tp} width=\"256\" height=\"256\">"
            tp = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/ray_masks/images/{set_}/{img_name}.png'
            out += f"<img src=/files/{tp} width=\"256\" height=\"256\">"
            out += "<br>"
            tp = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/ray_masks/overlays/{set_}/{img_name}.png'
            out += f"<img src=/files/{tp} width=\"768\" height=\"256\">"
            out += "<br>"
            
            # Shadow masks with smooth
            tp = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/vis/{set_}/{img_name}.png'
            out += f"<img src=/files/{tp} width=\"768\" height=\"256\">"
            out += "<br>"
            
            # Shadow masks without smooth
            tp = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_with_weight_simplified/vis/{set_}/{img_name}.png'
            out += f"<img src=/files/{tp} width=\"768\" height=\"256\">"
            out += "<br>"
            
            out += "<br><br>"
                
        return out
    
    return app
    
if __name__ == "__main__":
     
    app = create_app()
    app.run(host=argsp.host, port=argsp.port, debug=True, threaded=False)