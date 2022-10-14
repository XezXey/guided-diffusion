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
        out += f"<a href=\"/best_checkpoint_vid\">Best checkpoint (Video)</a> <br>"
        out += f"<a href=\"/best_checkpoint_img\">Best checkpoint (Image)</a> <br>"
        out += f"<a href=\"/best_diff_step_img\">Best diffusion-step (Image)</a> <br>"
        out += f"<a href=\"/uncond_vs_reverse\"> Uncondition Vs. Reverse sampling </a> <br>"
        out += f"<a href=\"/intermediate_step\"> Visualize the intermediate step </a> <br>"
        # out += f"<a href=\"/model_comparison_from_json/itp_method=Slerp&diff_step=1000&n_frame=5&sampling=reverse&ckpt=050000/\">Model comparison (From json)</a> <br>"
        out += f"<a href=\"/model_comparison_from_json/\">Model comparison (From json)</a> <br>"
        return out

    @app.route('/best_checkpoint_vid/')
    def best_checkpoint_vid():
        out = ""
        folder = f"{args.sample_dir}/{args.exp_dir}/"
        model = glob.glob(f"{folder}/*")
        for m in model:
            m_name = m.split('/')[-1]
            out += f"<a href=\"show_m_name={m_name}&itp=render_face&itp_method=Slerp&diff_step=1000&n_frame=30&sampling=reverse\">{m_name}</a> <br>"
        return out

    @app.route('/best_checkpoint_vid/show_m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&diff_step=<diff_step>&n_frame=<n_frame>&sampling=<sampling>')
    def best_checkpoint_vid_show(m_name, itp, itp_method, sampling, diff_step, n_frame=30):
        out = """<style>
                th, tr, td {
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        out += f"<h2>[#] Model name : {m_name} </h2>"
        out += f"<h2>[#] Interpolate on : {itp} </h2>"
        out += f"<h2>[#] Interpolate method : {itp_method} </h2>"
        out += f"<h2>[#] #N-Frames (@fps=30) : {n_frame} </h2>"
        folder = f"{args.sample_dir}/{args.exp_dir}/{m_name}/"
        checkpoint = sorted(glob.glob(f"{folder}/*"))
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path, 
                                                       n_subject=-1)
        for idx, src_dst in enumerate(subject_id):
            out += f"[#{idx}] {src_dst[0]} : <img src=/files/{data_path}/{src_dst[0].split('=')[-1]} width=\"64\" height=\"64\">, {src_dst[1]} : <img src=/files/{data_path}/{src_dst[1].split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
            out += "<table>"
            for ckpt in checkpoint:
                out += f"<th>{ckpt.split('/')[-1]}</th>"
            out += "<tr>"
            for ckpt in checkpoint:
                vid_path = glob.glob(f"{ckpt}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0]}/dst={src_dst[1]}/{itp_method}_{diff_step}/n_frames={n_frame}/res*.mp4")
                if len(vid_path) == 0:
                    out += "<td> <p style=\"color:red\">Video not found!</p> </td>"
                    continue
                else:
                  out += f"""
                  <td>
                  <video width=\"256\" height=\"256\" autoplay muted controls loop> 
                      <source src=\"/files/{vid_path[0]}\" type=\"video/mp4\">
                      Your browser does not support the video tag.
                      </video>
                  </td>
                  """
            out += "</tr>"     
            out += "</table>"
            out += "<br> <hr>"
                
        return out

    @app.route('/best_checkpoint_img/')
    def best_checkpoint_img():
        out = ""
        folder = f"{args.sample_dir}/{args.exp_dir}/"
        model = glob.glob(f"{folder}/*")
        for m in model:
            m_name = m.split('/')[-1]
            out += f"<a href=\"show_m_name={m_name}&itp=render_face&itp_method=Slerp&diff_step=1000&n_frame=5&sampling=reverse\">{m_name}</a> <br>"
        return out

    @app.route('/best_checkpoint_img/show_m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&diff_step=<diff_step>&n_frame=<n_frame>&sampling=<sampling>')
    def best_checkpoint_img_show(m_name, itp, itp_method, sampling, diff_step, n_frame=5):
        out = """<style>
                th, tr, td {
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        out += f"<h2>[#] Model name : {m_name} </h2>"
        out += f"<h2>[#] Interpolate on : {itp} </h2>"
        out += f"<h2>[#] Interpolate method : {itp_method} </h2>"
        out += f"<h2>[#] #N-Frames (@fps=30) : {n_frame} </h2>"
        folder = f"{args.sample_dir}/{args.exp_dir}/{m_name}/"
        checkpoint = sorted(glob.glob(f"{folder}/*"))
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path, 
                                                       n_subject=-1)
        
        for idx, src_dst in enumerate(subject_id):
            out += "<table>"
            out += "<tr> <th> Checkpoint </th> <th> Image </th> </tr>"
            out += f"[#{idx}] {src_dst[0]} : <img src=/files/{data_path}/{src_dst[0].split('=')[-1]} width=\"64\" height=\"64\">, {src_dst[1]} : <img src=/files/{data_path}/{src_dst[1].split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
            for ckpt in checkpoint:
                out += "<tr>"
                out += f"<td>{ckpt.split('/')[-1]}</td> "
                frames = glob.glob(f"{ckpt}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0]}/dst={src_dst[1]}/{itp_method}_{diff_step}/n_frames={n_frame}/res_*.png")
                out += "<td>"
                if len(frames) > 0:
                    frames = sort_by_frame(frames)
                    for f in frames:
                        out += "<img src=/files/" + f + ">"
                else:
                    out += "<p style=\"color:red\">Images not found!</p>"
                out += "</td>"
                out += "</tr>"
            out += "</table>"
            out += "<br> <hr>"
            
        return out
    
    @app.route('/best_diff_step_img/')
    def best_diff_step_img():
        out = ""
        folder = f"{args.sample_dir}/{args.exp_dir}/"
        model = glob.glob(f"{folder}/*")
        for m in model:
            m_name = m.split('/')[-1]
            out += f"<a href=\"show_m_name={m_name}&itp=render_face&itp_method=Slerp&n_frame=5&sampling=reverse\">{m_name}</a> <br>"
        return out

    @app.route('/best_diff_step_img/show_m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&n_frame=<n_frame>&sampling=<sampling>')
    def best_diff_step_show(m_name, itp, itp_method, sampling, n_frame=5):
        out = """<style>
                th, tr, td {
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        out += f"<h2>[#] Model name : {m_name} </h2>"
        out += f"<h2>[#] Interpolate on : {itp} </h2>"
        out += f"<h2>[#] Interpolate method : {itp_method} </h2>"
        out += f"<h2>[#] #N-Frames (@fps=30) : {n_frame} </h2>"
        folder = f"{args.sample_dir}/{args.exp_dir}/{m_name}/"
        checkpoint = sorted(glob.glob(f"{folder}/*"))
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path, 
                                                       n_subject=-1)
        
        for idx, src_dst in enumerate(subject_id):
            out += "<table>"
            out += "<tr> <th> Checkpoint </th> <th> Diffusion step </th> <th> Image </th> </tr>"
            out += f"[#{idx}] {src_dst[0]} : <img src=/files/{data_path}/{src_dst[0].split('=')[-1]} width=\"64\" height=\"64\">, {src_dst[1]} : <img src=/files/{data_path}/{src_dst[1].split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
            for ckpt in checkpoint:
                diff_step = glob.glob(f"{ckpt}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0]}/dst={src_dst[1]}/{itp_method}_*")
                diff_step = sorted([int(ds.split('/')[-1].split('_')[-1]) for ds in diff_step])
                out += f"<td rowspan=\"{len(diff_step)+1}\"> {ckpt.split('/')[-1]}</td> "
                for ds in diff_step:
                    out += "<tr>"
                    out += f"<td>{ds}</td> "
                    frames = glob.glob(f"{ckpt}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0]}/dst={src_dst[1]}/{itp_method}_{ds}/n_frames={n_frame}/res_*.png")
                    out += "<td>"
                    if len(frames) > 0:
                        frames = sort_by_frame(frames)
                        for f in frames:
                            out += "<img src=/files/" + f + ">"
                    else:
                        out += "<p style=\"color:red\">Images not found!</p>"
                    out += "</td>"
                    out += "</tr>"
            out += "</table>"
            out += "<br> <hr>"
            
        return out
    
    @app.route('/intermediate_step/')
    def intermediate_step():
        out = ""
        folder = f"{args.sample_dir}/{args.exp_dir}/"
        model = glob.glob(f"{folder}/*")
        for m in model:
            m_name = m.split('/')[-1]
            out += f"<a href=\"show_m_name={m_name}&itp=render_face&itp_method=Slerp&show_frame=0,25,50,75,100,125,250,500,750,800,999\">{m_name}</a> <br>"
        return out

    @app.route('/intermediate_step/show_m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&show_frame=<show_frame>')
    def intermediate_step_show(m_name, itp, itp_method, show_frame):
        out = """<style>
        th, tr, td {
            border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
        }
        </style>"""
        
        out += f"<h2>[#] Model name : {m_name} </h2>"
        out += f"<h2>[#] Show Frames : {show_frame} </h2>"
        folder = f"{args.sample_dir}/{args.exp_dir}/{m_name}/"
        checkpoint = sorted(glob.glob(f"{folder}/*"))
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path, 
                                                       n_subject=-1)
        
        process = {'uncond_sampling_0':{'name':'Gaussian', 'alias':'Gaussian noise', 'dir':['forward']}, 
                   'reverse_sampling':{'name':'Reversed', 'alias':'DDIM reverse', 'dir':['forward', 'reverse']},
                }
        
        show_frame = show_frame.split(',')
        show_frame = [int(s) for s in show_frame]
        
        for idx, src_dst in enumerate(subject_id):
            out += "<table>"
            out += "<tr> <th> Checkpoint </th> <th> Diffusion step </th> <th> Diffuse Process </th> <th> Image </th> </tr>"
            out += f"[#{idx}] {src_dst[0]} : <img src=/files/{data_path}/{src_dst[0].split('=')[-1]} width=\"64\" height=\"64\">, {src_dst[1]} : <img src=/files/{data_path}/{src_dst[1].split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
            for ckpt in checkpoint:
                diff_step = glob.glob(f"{ckpt}/{args.set_}/{itp}/Intermediate/*")
                diff_step = sorted([int(ds.split('/')[-1].split('_')[-1]) for ds in diff_step])
                out += f"<td rowspan=\"{len(diff_step) + 3}\"> {ckpt.split('/')[-1]}</td> "
                for ds in diff_step:
                    out += "<tr>"
                    out += f"<td rowspan=\"{3}\"> {ds} </td> "
                    for sampling, proc in process.items():
                        for dir in proc['dir']:
                            out += f"<td>{proc['alias']} : {dir}</td> "
                            frames = glob.glob(f"{ckpt}/{args.set_}/{itp}/Intermediate/diffstep_{ds}/{sampling}/src={src_dst[0]}/dst={src_dst[1]}/{proc['name']}/{dir}/{src_dst[0]}/sample/*frame*.png")
                            out += "<td>"
                            if len(frames) > 0:
                                frames = sort_by_frame(frames)
                                if -1 in show_frame:
                                    for f in frames:
                                        out += "<img src=/files/" + f + ">"
                                else:
                                    for f in [frames[i] for i in show_frame]:
                                        out += "<img src=/files/" + f + ">"
                            else:
                                out += "<p style=\"color:red\">Images not found!</p>"
                            out += "</td>"
                            out += "</tr>"
            out += "</table>"
            out += "<br> <hr>"
            
        return out
        
    
    @app.route('/uncond_vs_reverse/')
    def uncond_vs_reverse_img():
        out = ""
        folder = f"{args.sample_dir}/{args.exp_dir}/"
        model = glob.glob(f"{folder}/*")
        for m in model:
            m_name = m.split('/')[-1]
            out += f"<a href=\"show_m_name={m_name}&itp=render_face&itp_method=Slerp&diff_step=1000&n_frame=5\">{m_name}</a> <br>"
        return out

    @app.route('/uncond_vs_reverse/show_m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&diff_step=<diff_step>&n_frame=<n_frame>')
    def uncond_vs_reverse_img_show(m_name, itp, itp_method, diff_step, n_frame=5):
        out = """<style>
                th, tr, td {
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        out += f"<h2>[#] Model name : {m_name} </h2>"
        out += f"<h2>[#] Interpolate on : {itp} </h2>"
        out += f"<h2>[#] Interpolate method : {itp_method} </h2>"
        out += f"<h2>[#] #N-Frames (@fps=30) : {n_frame} </h2>"
        
        folder = f"{args.sample_dir}/{args.exp_dir}/{m_name}/"
        checkpoint = sorted(glob.glob(f"{folder}/*"))
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path, 
                                                       n_subject=-1)
        
        for idx, src_dst in enumerate(subject_id):
            out += "<table>"
            out += "<tr> <th> Checkpoint </th>  <th> Sampling </th>  <th> Image </th> </tr>"
            out += f"[#{idx}] {src_dst[0]} : <img src=/files/{data_path}/{src_dst[0].split('=')[-1]} width=\"64\" height=\"64\">, {src_dst[1]} : <img src=/files/{data_path}/{src_dst[1].split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
            for ckpt in checkpoint:
                out += "<tr>"
                out += f"<td rowspan=\"2\">{ckpt.split('/')[-1]}</td> "
                for sampling in ['uncond', 'reverse']:
                    out += f"<td> {sampling}</td>"
                    frames = glob.glob(f"{ckpt}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0]}/dst={src_dst[1]}/{itp_method}_{diff_step}/n_frames={n_frame}/res_*.png")
                    out += "<td>"
                    if len(frames) > 0:
                        frames = sort_by_frame(frames)
                        for f in frames:
                            out += "<img src=/files/" + f + ">"
                    else:
                        out += "<p style=\"color:red\">Images not found!</p>"
                    out += "</td>"
                    out += "</tr>"
            out += "</table>"
            out += "<br> <hr>"
            
        return out


    @app.route('/model_comparison_from_json/')
    def model_comparison_from_json_selector():
        out = ""
        json_files = glob.glob('./json_comparison/*.json')
        link_based = "itp_method=<itp_method>&diff_step=<diff_step>&n_frame=<n_frame>&sampling=<sampling>&ckpt=<ckpt>"
        link_based = "itp_method=Slerp&diff_step=1000&n_frame=5&sampling=reverse&ckpt=050000"
        for jf in json_files:
            jf = jf.split('/')[-1]
            out += f"Comparison file : <a href=\"jf={jf}&&{link_based}\">{jf}</a> <br>"
        return out
        
    @app.route('/model_comparison_from_json/jf=<jf>&&itp_method=<itp_method>&diff_step=<diff_step>&n_frame=<n_frame>&sampling=<sampling>&ckpt=<ckpt>/')
    def model_comparison_from_json_show(jf, itp_method, n_frame, diff_step, sampling, ckpt):
        
        print(ckpt)
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        out+= ckpt
        f = open(f'./json_comparison/{jf}')
        ckpt_dict = json.load(f)
        model = list(ckpt_dict.keys())
        
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_pair_json}", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path,
                                                       n_subject=-1)
        print(subject_id)
        out += create_button_fn()
        for s_id, src_dst in enumerate(subject_id):
            print(src_dst)
            out += "<table>"
            out += "<tr> <th> Checkpoint </th> <th> Video </th> <th> Input </th> <th> Image </th> <th> Input </th> </tr>"
            out += f"[#{s_id}] {src_dst[0]} : <img src=/files/{data_path}/{src_dst[0].replace('jpg', 'png')} width=\"64\" height=\"64\">, {src_dst[1]} : <img src=/files/{data_path}/{src_dst[1].replace('jpg', 'png')} width=\"64\" height=\"64\">" + "<br>" + "<br>"
            out += create_hide(model, s_id)
            for m_id, m in enumerate(model):
                if ckpt == 'json':
                    vis_ckpt = step = ckpt_dict[m]['step']
                else:
                    vis_ckpt = step = f'ema_{ckpt}'
                    
                out += f"<tr id={s_id}_{m_id}>"
                out += f"<td> {m_id+1} : {ckpt_dict[m]['alias']} <br> ({vis_ckpt}) </td>"
                itp = ckpt_dict[m]['itp']
                each_model = f"{args.sample_dir}/{args.exp_dir}/{m}/{step}/{args.set_}/{itp}/{sampling}_sampling/src={src_dst[0].replace('png', 'jpg')}/dst={src_dst[1].replace('png', 'jpg')}/"
                frames = glob.glob(f"{each_model}/{itp_method}_{diff_step}/n_frames={n_frame}/res_*.png")
                vid_path = glob.glob(f"{each_model}/{itp_method}_{diff_step}/n_frames={n_frame}/res_*.mp4")
                    
                if len(vid_path) == 0:
                    out += "<td> <p style=\"color:red\">Video not found!</p> </td>"
                else:
                  out += f"""
                  <td>
                  <video width=\"64\" height=\"64\" autoplay muted controls loop> 
                      <source src=\"/files/{vid_path[0]}\" type=\"video/mp4\">
                      Your browser does not support the video tag.
                      </video>
                  </td>
                  """
                
                out += f"<td> <img src=/files/{data_path}/{src_dst[0].replace('jpg', 'png')}> </td>"
                out += "<td>"
                if len(frames) > 0:
                    frames = sort_by_frame(frames)
                    for f in frames:
                        out += "<img src=/files/" + f + ">"
                else:
                    out += "<p style=\"color:red\">Images not found!</p>"
                out += "</td>"
                out += f"<td> <img src=/files/{data_path}/{src_dst[0].replace('jpg', 'png')}> </td>"
                out += "</tr>"
                
            out += "</table>"
           
            out += "<br> <hr>"
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