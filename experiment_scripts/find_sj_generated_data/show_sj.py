from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--set_', type=str, default='valid')
args = parser.parse_args()



def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        out = """
        <style>
            tr { display: block; float: left; }
            th, td { display: block; }
        </style>
        """
        out += "<table>"
        
        # out += "<tr>"
        # # for f in folders:
        # #     # out += f"<tr> {f} </tr>"
        # #     out += f"<p style=\"display: inline; margin:64px;\">{f}</p>"
        # out += "</tr>"
            
        path = f'/data/mint/DPM_Dataset/generated_dataset_80perc/gen_images/{args.set_}'

        imgs = glob.glob(f'{path}/*.png')
        sj_dict = {}
        for f in imgs:
            sj_dict['src=' + f.split('/')[-1].split('_')[0] + '.jpg'] = f
            
        
        for f in sj_dict.keys():
            f = f.split('=')[-1].replace('.jpg', '.png')
            print(f)
            img = '/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_128/valid/' + f
            out += "<tr>"
            # out += f"<td> <img src=\"/files/{vid}/path.png\" width=256px </td>"
            # for vid in vids:
            out += "<td>"
            out += f
            out += "<img src=/files/" + img + ">"
            out += "<td>"
            out += "</tr>"
        out += "</table>"
        return out
    return app
        
if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port='2841', debug=True, threaded=False)