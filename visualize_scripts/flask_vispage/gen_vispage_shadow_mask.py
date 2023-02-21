from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/sample_utils/')
import mani_utils, file_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--port', required=True)
parser.add_argument('--res', default=256)
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--set_', default='train')
parser.add_argument('--n', default=1000)
args = parser.parse_args()


def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        #NOTE: Root path
        
        raw_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_{args.res}/{args.set_}/"
        img_path = file_utils._list_image_files_recursively(raw_path)
        
        render_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/rendered_images/deca_masked_face_images_wclip/{args.set_}/"
        shadow_mask_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_masks/{args.set_}/"
        
        out = "Visualization of Ray-casting for generating the shadow mask"
        idx = np.random.randint(0, len(img_path), args.n)
        img_path = [img_path[i] for i in idx]
        
        out += "<table>"
        out += "<tr> <th> Image </th> <th> Shadow mask </th> <th> Render </th> </tr>"
        for path in img_path:
            img_name = path.split('/')[-1]
            out += "<tr>"
            out += f"<td> <img src=/files/{raw_path + img_name}> </td>"
            out += f"<td> <img src=/files/{shadow_mask_path + img_name.replace('.jpg', '.png')}> </td>"
            out += f"<td> <img src=/files/{render_path + img_name.replace('.jpg', '.png')}> </td>"
            out += "</tr>"
        
        return out
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)