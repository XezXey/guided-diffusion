from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/sample_utils/')
import mani_utils, file_utils
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', default="/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate")
parser.add_argument('--sample_pair_json', default="/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate")
parser.add_argument('--dataset_path', default="/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/")
parser.add_argument('--comparison_json', required=True)
parser.add_argument('--set_', default='valid')
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
        out += f"<a href=\"/sota_compare/\"> SOTA Comparison </a> <br>"
        return out

    @app.route('/sota_compare/')
    def sota_compare():
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        f = open(args.comparison_json, 'r')
        cmp_dict = json.load(f)
        model = list(cmp_dict.keys())
        
        f = open(f"{args.sample_pair_json}")
        sample_pairs = json.load(f)['pair']
        
        out += "<table>"
        out += "<tr> <th> Model; <pre> Image : src(left), dst(right) </pre> </th>"
        
        print(sample_pairs)
        for i in range(1, len(sample_pairs)+1):
            src_dst = [sample_pairs[f'pair{i}']['src'], sample_pairs[f'pair{i}']['dst']]
            out += "<tr>"
            out += f"<th style=\"font-size:10px;white-space: nowrap;\"> {i+1}.({src_dst[0].split('.')[0]} => {src_dst[1].split('.')[0]}) <br> <br> <br> <br> <br> <img src=/files/{data_path}/{src_dst[0].replace('jpg', 'jpg')} title=\"{src_dst[0]}\"> <img src=/files/{data_path}/{src_dst[1].replace('jpg', 'jpg')} title=\"{src_dst[1]}\"> </th>"
        
            # SOTA
            img_name = f"input={src_dst[0]}" + "%23" + f"pred={src_dst[1]}.png"
            for m_id, m in enumerate(model):
                img_path = f"{cmp_dict[m]['img_dir']}/{img_name}"
                out += f"<td>"
                out += f"<img src=/files/{img_path} title=\"{cmp_dict[m]['alias']}\">"
                out += "</td>"

            out += "</tr>"
        out += "</table>"
        out += "<br> <hr>"
        return out

    return app

if __name__ == "__main__":
    
    
    # f"/data/mint/DPM_Dataset/MultiPIE_testset/mp_aligned/{args.set_}/"
    data_path = args.dataset_path + args.set_
    print(data_path)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)
