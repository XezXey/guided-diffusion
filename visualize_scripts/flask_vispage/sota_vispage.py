from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/sample_utils/')
import mani_utils, file_utils

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

    @app.route('/sota_compare/jf=<jf>')
    def sota_compare(jf):
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        f = open(f'./json_comparison/sota/{jf}')
        cmp_dict = json.load(f)
        model = list(cmp_dict.keys())
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=dataset_img_path,
                                                       n_subject=-1)
        
        out += "<table>"
        out += "<tr> <th> Model; <pre> Image : src(left), dst(right) </pre> </th>"
        
        score_file = {}
        for m_id, m in enumerate(model):
            out += f"<th style=\"font-size:10px;white-space: nowrap;\"> {m_id+1}." + cmp_dict[m]['alias']
            try :
                if cmp_dict[m]['misc'] == 'sota':
                    with open(f'{args.sample_dir}/{m}/eval_score.json', 'r') as fp:
                        score_file[m] = json.load(fp)
                elif cmp_dict[m]['misc'] == 'ours':
                    with open(f"{args.sample_dir}/Ours/{m}/{cmp_dict[m]['step']}/eval_score.json", 'r') as fp:
                        score_file[m] = json.load(fp)
            except :
                out += " (No Eval File.)"
            out += "</th>"
            
        for s_id, src_dst in enumerate(subject_id):
            out += "<tr>"
            out += f"<th style=\"font-size:10px;white-space: nowrap;\"> {s_id+1}.({src_dst[0].split('.')[0]} => {src_dst[1].split('.')[0]}) <br> <br> <br> <br> <br> <img src=/files/{data_path}/{src_dst[0].replace('jpg', 'png')} title=\"{src_dst[0]}\"><img src=/files/{data_path}/{src_dst[1].replace('jpg', 'png')} title=\"{src_dst[1]}\"> </th>"
        
            # SOTA
            img_name = f"input={src_dst[0]}" + "%23" + f"pred={src_dst[1]}.png"
            for m_id, m in enumerate(model):
                try:
                    mse = "%0.5f" % float(score_file[m]['each_image'][f'{src_dst[1]}']['mse'])
                    dssim = "%0.5f" % float(score_file[m]['each_image'][f'{src_dst[1]}']['dssim'])
                    lpips = "%0.5f" % float(score_file[m]['each_image'][f'{src_dst[1]}']['lpips'])
                except: mse = dssim = lpips = 'NaN'
                if cmp_dict[m]['misc'] == "sota":
                    img_path = f"{args.sample_dir}/{m}/out/{img_name}"
                    out += f"<td>"
                    out += f"MSE = {mse} <br> DSSIM = {dssim} <br> LPIPS = {lpips} <br>"
                    out += f"<img src=/files/{img_path} title=\"{cmp_dict[m]['alias']}\">"
                    out += "</td>"
                elif cmp_dict[m]['misc'] == "ours": 
                    img_path = f"{args.sample_dir}/Ours/{m}/{cmp_dict[m]['step']}/upsample/{img_name}"
                    out += f"<td>"
                    out += f"MSE = {mse} <br> DSSIM = {dssim} <br> LPIPS = {lpips} <br>"
                    out += f"<img src=/files/{img_path} title=\"{cmp_dict[m]['alias']}\">"
                    out += "</td>"
                    
            out += "</tr>"
        out += "</table>"
        out += "<br> <hr>"
        return out

    @app.route('/sota_compare/')
    def model_comparison_from_json_selector():
        out = ""
        json_files = glob.glob('./json_comparison/sota/*.json')
        for jf in json_files:
            jf = jf.split('/')[-1]
            out += f"Comparison file : <a href=\"jf={jf}\">{jf}</a> <br>"
        return out
    
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', default="/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate")
    parser.add_argument('--sample_pair_json', default="/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate")
    parser.add_argument('--set_', default='valid')
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    
    
    data_path = f"/data/mint/DPM_Dataset/MultiPIE/mp_aligned/{args.set_}/"
    dataset_img_path = file_utils._list_image_files_recursively(data_path)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)