from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys

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
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        f = open(args.comparison_json, 'r')
        cmp_dict = json.load(f)
        model = list(cmp_dict.keys())
        
        out += "<table>"
        out += "<tr> <th> Model; <pre> Image : src(left), dst(right) </pre> </th>"
        
        with open(args.sample_pair_json, 'r') as f:
            sample_pairs = json.load(f)['pair']
            
        data_path = "/data/mint/DPM_Dataset/MultiPIE/MultiPIE_testset/mp_aligned/valid/"
        for p_id, src_dst in sample_pairs.items():
            src = src_dst['src']
            dst = src_dst['dst']
            out += "<tr>"
            # out += f"<th style=\"font-size:10px;white-space: nowrap;\"> {p_id}: {src} => {dst} <br> <br> <br> <br> <br> <img style=\"width:128px;\" src=/files/{data_path}/{src.replace('jpg', 'png')} title=\"{src}\"><img style=\"width:128px;\" src=/files/{data_path}/{dst.replace('jpg', 'png')} title=\"{dst}\"> </th>"
            out += f"<th style=\"font-size:10px;white-space: nowrap;\"> {p_id}: {src} => {dst} <br> <br> <br> <br> <br> <img src=/files/{data_path}/{src.replace('jpg', 'png')} title=\"{src}\"><img src=/files/{data_path}/{dst.replace('jpg', 'png')} title=\"{dst}\"> </th>"
        
            # SOTA
            img_name = f"input={src}" + "%23" + f"pred={dst}.png"
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
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_pair_json', required=True)
    parser.add_argument('--comparison_json', required=True)
    parser.add_argument('--set_', default='valid')
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    
    # f"/data/mint/DPM_Dataset/MultiPIE_testset/mp_aligned/{args.set_}/"
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)
