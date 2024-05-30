import numpy as np
from flask import Flask, request, send_file, send_from_directory
import glob, os, re
import numpy as np
import json
import sys
import argparse


def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        out = ""
        
        path = [
          f'/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/', 
          f'/data/mint/DPM_Dataset/ffhq_256_with_anno/rendered_images/deca_masked_face_images_wclip/', 
          f'/data/mint/DPM_Dataset/try_code/render_wclip/',
          f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_masks/',
          f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff//futschik_1e-1/',
          f'data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff/median5_5e-2/',
        ]
        ext = ['jpg', 'png', 'png', 'png', 'png', 'png']
        

        
        for f, s in zip(args.file, ['train', 'valid']):
            if not os.path.exists(f): continue
            # Show the images in the same rows
            with open(f, 'r') as file:
                dat = json.load(file)
                dat = [d['src'] for d in dat['pair'].values()]
            for sj_name in dat:
                for p, e in zip(path, ext):
                    out += f"<img src=/files/{p}/{s}/{sj_name.replace('jpg', e)} width=\"256\" height=\"256\">"
                out += "<br>"
                
        return out
    
    return app
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_', default='train')
    parser.add_argument('--res', default=128)
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--file', nargs='+', required=True)
    args = parser.parse_args()
     
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)