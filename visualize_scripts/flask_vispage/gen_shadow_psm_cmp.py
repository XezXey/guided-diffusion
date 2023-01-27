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
parser.add_argument('--addshadow_dir', default="/FFHQ_Hope_addshadow/")
parser.add_argument('--rmvshadow_dir', default="/FFHQ_Hope_rmvshadow/")
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
        out += f"<a href=\"/shadow/\"> Shadow (From json)</a> <br>"
        return out

    def create_hide(model, s_id): 
        tmp = ""
        for m_id, _ in enumerate(model):
            tmp += (
                f"<button id=but_{s_id}_{m_id} style='color:blue' onclick=\"toggle({s_id}, {m_id})\">Hide : {m_id+1}</button>"
        )
        return tmp
    
    def create_button_fn():
        tmp = (
            f"<script>"
                f"function toggle(s_id, m_id){{"
                    f"let element = document.getElementById(s_id + \"_\" + m_id);"
                    f"let button = document.getElementById(\"but_\" + s_id + \"_\" + m_id);"
                    f"let hidden = element.getAttribute(\"hidden\");"
                    f"if (hidden) {{"
                    f"    element.removeAttribute(\"hidden\");"
                    f"    button.innerText = \"Hide : \" + m_id;"
                    f"    button.style.color = 'blue';"
                    f"}} else {{"
                    f"    element.setAttribute(\"hidden\", \"hidden\");"
                    f"    button.innerText = \"Show : \" + m_id;"
                    f"    button.style.color = 'red';"
                    f"}}"
                f"}}"
                f"</script>"
        )
        return tmp

    @app.route('/shadow/jf=<jf>&itp_method=<itp_method>&diff_step=<diff_step>&sampling=<sampling>&ckpt=<ckpt>&show=<show>/')
    def shadow(jf, itp_method, diff_step, sampling, ckpt, show):
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                table {
                    width: 100%
                }
                table.fixed {
                    table-layout: fixed;
                }
                </style>"""
        f = open(f'./json_comparison/shadow/{jf}')
        ckpt_dict = json.load(f)
        model = list(ckpt_dict.keys())
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path,
                                                       n_subject=-1)
        print("GG", subject_id)
        out += create_button_fn()
        
        out += "<table>"
        out += "<tr> <th> Image name </th> <th> Potrait Shadow Manipulation (baseline) </th>"
        out += "<th> Input </th>"
        # for s_id, src_dst in enumerate(subject_id):
        for m_id, m in enumerate(model):
            out += f"<th> {m_id+1}." + ckpt_dict[m]['alias'] + "(Selected)" + "</th>"
            out += f"<th> {m_id+1}." + ckpt_dict[m]['alias'] + "(All)" + "</th>"
        # out += " <br> </tr>"
        
        cmp_path = f"/home/mint/guided-diffusion/preprocess_scripts/PSM_rebuttal/aligned/"
        
        for s_id, cmp_name in enumerate(sorted(glob.glob(f'{cmp_path}/*.png'))):
            cmp_name =  cmp_name.split('/')[-1]
            out += f"<tr>"
            out += f"<th> {s_id+1}.{cmp_name} </th>"
            # SOTA 
            out += f"<th style=\"font-size:10px;white-space: nowrap;vertical-align: bottom;\"> <img src=/files/{cmp_path}/{cmp_name} title=\"{cmp_name}\"> </th>"
 
            show_name = cmp_name.split('_')[0] + '.png'
            for _, src_dst in enumerate(subject_id):
                if show_name in src_dst[0]:
                    # Input image
                    out += f"<th style=\"font-size:10px;white-space: nowrap;vertical-align: bottom;\"> <img src=/files/{data_path}/{src_dst[0]} title=\"{src_dst[0]}\"> </th>"

                    # Ours
                    for m_id, m in enumerate(model):
                        if ckpt == 'json':
                            step = ckpt_dict[m]['step']
                        else:
                            step = f'ema_{ckpt}'
                        
                        itp = ckpt_dict[m]['itp']
                        n_frames = ckpt_dict[m]['n_frames']
                        each_model_rmv = f"{args.sample_dir}/{args.exp_dir}/{args.rmvshadow_dir}/{m}/{step}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0]}/dst={src_dst[1]}/"
                        frames_rmv = glob.glob(f"{each_model_rmv}/{itp_method}_{diff_step}/n_frames={n_frames}/{show}_f*.png")
                        frames = sort_by_frame(frames_rmv)
                        n_frames_sld = int(ckpt_dict[m]['n_frames']) - 1
                        out += "<td>"
                        out += f"""
                        <div class="slidecontainer">
                            <input type="range" min="0" max="{n_frames_sld}" value="0" class="slider" id="sel_{s_id}"
                            oninput="changeImage({s_id}, {frames})">
                        </div>
                        """
                        out += (
                            "<script>"
                            "function changeImage(id, frames) {"
                                "var slider = document.getElementById('sel_' + id);"
                                "var image = document.getElementById('ours_' + id);"
                                "var index = slider.value;"
                                "image.src = '/files/' + frames[index];"
                            "}"
                        "</script>"
                        )
                        out += f"<img id=\"ours_{s_id}\" src=/files/{frames[0]} title=\"{ckpt_dict[m]['alias']}\">"
                        out += "</td>"
                        out += "<td style=\"vertical-align: bottom;\">"
                        if len(frames) > 0:
                            for f in frames:
                                out += f"<img src=/files/{f} title=\"{ckpt_dict[m]['alias']}\">"
                            out += "<br>"
                        else:
                            out += "<p style=\"color:red\">Images not found!</p>"
                        out += "</td>"
                    out += "</tr>"
        out += "</table>"
        out += "<br> <hr>"
        return out

    @app.route('/shadow/')
    def model_comparison_from_json_selector_shadow():
        out = ""
        json_files = glob.glob('./json_comparison/shadow/*.json')
        link_based = "itp_method=Lerp&diff_step=1000&sampling=reverse&ckpt=json&show=res"
        for jf in json_files:
            jf = jf.split('/')[-1]
            out += f"Comparison file : <a href=\"jf={jf}&{link_based}\">{jf}</a> <br>"
        return out

    return app

if __name__ == "__main__":
    
    
    data_path = f"/data/mint/DPM_Dataset/ITW/itw_images_aligned/{args.set_}"
    # data_path = f"/home/mint/guided-diffusion/preprocess_scripts/PSM_rebuttal/aligned/"
    img_path = file_utils._list_image_files_recursively(data_path)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)