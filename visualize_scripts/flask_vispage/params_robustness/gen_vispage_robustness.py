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
parser.add_argument('--sample_pair_json', default="/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate")
parser.add_argument('--set_', default='valid')
parser.add_argument('--res', default=128)
parser.add_argument('--port', required=True)
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
        out += f"<a href=\"/model_comparison_from_json_lf/\">Model comparison Last frame (From json)</a> <br>"
        return out

    @app.route('/model_comparison_from_json_lf/jf=<jf>&itp_method=<itp_method>&diff_step=<diff_step>&sampling=<sampling>&ckpt=<ckpt>&show=<show>/')
    def model_comparison_from_json_lf(jf, itp_method, diff_step, sampling, ckpt, show):
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
        f = open(f'./json_comparison/experiment/{jf}')
        ckpt_dict = json.load(f)
        model = list(ckpt_dict.keys())
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path,
                                                       n_subject=-1)
        
        out += create_button_fn()
        
        out += "<table>"
        out += "<tr> <th> Model; <pre> Image : src(left), dst(right) </pre> </th>"
        # for s_id, src_dst in enumerate(subject_id):
        for m_id, m in enumerate(model):
            out += f"<th> {m_id+1}." + ckpt_dict[m]['alias'] + "</th>"
        # out += " <br> </tr>"
        
        for s_id, src_dst in enumerate(subject_id):
            # out += create_hide(m, m_id)
            out += f"<tr>"
            if int(args.res) == 256:
                out += f"<th style=\"font-size:10px;white-space: nowrap;\"> {s_id+1}.({src_dst[0].split('.')[0]},{src_dst[1].split('.')[0]})<img src=/files/{data_path}/{src_dst[0]} title=\"{src_dst[0]}\"><img src=/files/{data_path}/{src_dst[1]} title=\"{src_dst[1]}\"> </th>"
            else:
                out += f"<th style=\"font-size:10px;white-space: nowrap;\"> {s_id+1}.({src_dst[0].split('.')[0]},{src_dst[1].split('.')[0]})<img src=/files/{data_path}/{src_dst[0].replace('jpg', 'png')} title=\"{src_dst[0]}\"><img src=/files/{data_path}/{src_dst[1].replace('jpg', 'png')} title=\"{src_dst[1]}\"> </th>"
            for m_id, m in enumerate(model):
                if ckpt == 'json':
                    vis_ckpt = step = ckpt_dict[m]['step']
                else:
                    vis_ckpt = step = f'ema_{ckpt}'
                    
                itp = ckpt_dict[m]['itp']
                n_frames = ckpt_dict[m]['n_frames']
                if int(args.res) == 256:
                    # print("GGGGGGGGGGGGGGGGG")
                    # print(ckpt_dict[m]['step'])
                    # print(step)
                    each_model = f"{args.sample_dir}/{args.exp_dir}/{m}/{step}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0]}/dst={src_dst[1]}/"
                else:
                    each_model = f"{args.sample_dir}/{args.exp_dir}/{m}/{step}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0].replace('png', 'jpg')}/dst={src_dst[1].replace('png', 'jpg')}/"
                # print("GGODDL")
                # print(each_model)
                
                frames = glob.glob(f"{each_model}/{itp_method}_{diff_step}/n_frames={n_frames}/{show}_f*.png")
                # print("HELP")
                # print(f"{each_model}/{itp_method}_{diff_step}/n_frames={n_frames}/")
                # print(frames)
                # print("PLS HELP")
                
                
                out += "<td>"
                if len(frames) > 0:
                    frames = sort_by_frame(frames)
                    out += f"<img src=/files/{frames[-1]} title=\"{ckpt_dict[m]['alias']}\">"
                else:
                    out += "<p style=\"color:red\">Images not found!</p>"
                out += "</td>"
            out += "</tr>"
        out += "</table>"
        out += "<br> <hr>"
        return out

    @app.route('/model_comparison_from_json_lf/')
    def model_comparison_from_json_selector_lf():
        out = ""
        json_files = glob.glob('./json_comparison/experiment/*.json')
        link_based = "itp_method=Lerp&diff_step=1000&sampling=reverse&ckpt=json&show=res"
        for jf in json_files:
            jf = jf.split('/')[-1]
            out += f"Comparison file : <a href=\"jf={jf}&{link_based}\">{jf}</a> <br>"
        return out

    return app

if __name__ == "__main__":
    
    
    data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_{args.res}/{args.set_}/"
    img_path = file_utils._list_image_files_recursively(data_path)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)