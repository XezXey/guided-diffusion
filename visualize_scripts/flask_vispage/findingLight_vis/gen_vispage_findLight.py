from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/sample_utils/')
import mani_utils, file_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', required=True)
parser.add_argument('--sample_dir', default="/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate")
parser.add_argument('--set_', default='valid')
parser.add_argument('--port', required=True)
parser.add_argument('--host', default='0.0.0.0')
args = parser.parse_args()

# Edit global here
model = "log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml"
sj_path = f"{args.sample_dir}/{args.exp_dir}/{model}/ema_085000/valid/render_face/reverse_sampling/"
data_path = "/data/mint/DPM_Dataset/ITW_jr/"

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
        out += f"<a href=\"/subject_selector/\">Subject Selector (From dirs)</a> <br>"
        return out

    @app.route('/subject_selector/sj=<sj_name>/')
    def subject_selector_show(sj_name):
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                table {
                    width: 60%
                }
                table.fixed {
                    table-layout: fixed;
                }
                </style>"""
                
        light_from = glob.glob(f'{sj_path}/src={sj_name}/*')
        original = f"{data_path}/aligned_images/valid/{sj_name}" 
        for id, lf in enumerate(light_from):
            lf_name = lf.split('/')[-1]
            out += "<table>"
            out += f"<tr> <th> Model; <pre> Image : src({sj_name}), dst({lf_name}) </pre> </th>"
            out += f"<tr>"
            # print(lf)
            frames = glob.glob(f"{lf}/Lerp_1000/n_frames=2/res_f*")
            # print(frames)
            if len(frames) > 0:
                frames = sort_by_frame(frames)
                
                # Source
                out += "<tr> <td>"
                out += f"<img src=/files/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/{lf_name.split('=')[-1]}>"
                for f in frames:
                    f_tmp = f.split('/')[-1].split('_')[-1]
                    out += f"<img src=/files/{original}>"
                out += "</td> </tr>"
                
                out += "<tr> <td>"
                out += f"<img src=/files/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/{lf_name.split('=')[-1]}>"
                for f in frames:
                    out += f"<img src=/files/{f}>"
                out += "</td> </tr>"
                
            out += "</table>"
            out += "<br> <hr>"
        return out

    @app.route('/subject_selector/')
    def subject_selector():
        out = ""
        for sj in glob.glob(f'/{sj_path}/*'):
            sj_name = sj.split('/')[-1].split('=')[-1]
            out += f"Comparison file : <a href=\"sj={sj_name}\">{sj_name}</a> <br>"
        return out

    return app

if __name__ == "__main__":
    
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)