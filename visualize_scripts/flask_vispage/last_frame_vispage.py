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
        # out += f"<a href=\"/model_comparison_from_json/\">Model comparison (From json)</a> <br>"
        out += f"<a href=\"/model_comparison_from_json_lf/\">Model comparison Last frame (From json)</a> <br>"
        return out

    @app.route('/model_comparison_from_json_lf/jf=<jf>&&itp_method=<itp_method>&diff_step=<diff_step>&n_frame=<n_frame>&sampling=<sampling>&ckpt=<ckpt>&show=<show>/')
    def model_comparison_from_json_lf(jf, itp_method, n_frame, diff_step, sampling, ckpt, show):
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        f = open(f'./json_comparison/{jf}')
        ckpt_dict = json.load(f)
        model = list(ckpt_dict.keys())
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path,
                                                       n_subject=-1)
        
        out += create_button_fn()
        
        out += "<table>"
        out += "<tr> <th> Checkpoint </th>"
        for s_id, src_dst in enumerate(subject_id):
            out += f"<th> <img src=/files/{data_path}/{src_dst[0].replace('jpg', 'png')} width=\"64\" height=\"64\" title=\"{src_dst[0]}\">, <img src=/files/{data_path}/{src_dst[1].replace('jpg', 'png')} width=\"64\" height=\"64\" title=\"{src_dst[1]}\"> </th>"
        out += "</tr>"
        
        for m_id, m in enumerate(model):
            # out += create_hide(m, m_id)
            
            out += f"<tr>"
            out += f"<td> {m_id+1} : {ckpt_dict[m]['alias']} <br> </td>"
            for s_id, src_dst in enumerate(subject_id):
                if ckpt == 'json':
                    vis_ckpt = step = ckpt_dict[m]['step']
                else:
                    vis_ckpt = step = f'ema_{ckpt}'
                    
                itp = ckpt_dict[m]['itp']
                each_model = f"{args.sample_dir}/{args.exp_dir}/{m}/{step}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0].replace('png', 'jpg')}/dst={src_dst[1].replace('png', 'jpg')}/"
                frames = glob.glob(f"{each_model}/{itp_method}_{diff_step}/n_frames={n_frame}/{show}_f*.png")
                
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
    def model_comparison_from_json_selector():
        out = ""
        json_files = glob.glob('./json_comparison/*.json')
        link_based = "itp_method=Slerp&diff_step=1000&n_frame=5&sampling=reverse&ckpt=050000&show=res"
        for jf in json_files:
            jf = jf.split('/')[-1]
            out += f"Comparison file : <a href=\"jf={jf}&&{link_based}\">{jf}</a> <br>"
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
    return app

if __name__ == "__main__":
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
    
    
    data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_{args.res}/{args.set_}/"
    img_path = file_utils._list_image_files_recursively(data_path)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)