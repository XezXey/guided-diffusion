from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--port', required=True)
parser.add_argument('--sampling_path', default="/data/mint/sampling/photometric/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml/ema_085000/valid/render_face/reverse_sampling/")
parser.add_argument('--host', default='0.0.0.0')
args = parser.parse_args()

def sort_by_frame(path_list):
    frame_anno = []
    for p in path_list:
        # frame_idx = os.path.splitext(p.split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_idx = os.path.splitext(p.split('/')[-1].split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_anno.append(int(frame_idx))
    sorted_idx = np.argsort(frame_anno)
    sorted_path_list = []
    for idx in sorted_idx:
      sorted_path_list.append(path_list[idx])
    return sorted_path_list

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        #NOTE: Root path
        out = ""
        path = glob.glob(f"/{args.sampling_path}/*")
        for p in path:
            sj = p.split('/')[-1]
            out += f"<a href=/show/sj={sj}&disp=res> {sj} </a> <br>"
        return out
    
    @app.route('/show/sj=<sj>&disp=<disp>')
    def show(sj, disp):
        out = ""
        out += f'{sj} <br> '
        imgs = glob.glob(f"/{args.sampling_path}/{sj}/**/{disp}_*.png", recursive=True)
        print(len(imgs))
        for img in imgs:
            img = img[1:]
            if 'res_frame0.png' in img: continue
            out += "<img src=/files/" + img + ">"
        
        return out
        

    return app

if __name__ == "__main__":
    # data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_{args.res}/{args.set_}/"
    # img_path = file_utils._list_image_files_recursively(data_path)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)