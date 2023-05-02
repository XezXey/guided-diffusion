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
        out = "<h1> Experiment on Diffusion's sampling step </h1>"
        out += f"<a href=\"/fixed_trainingstep__varying_diffstep\">Fixed training step & Varying diffusion step</a> <br>"
        out += f"<a href=\"/varying_trainingstep__fixed_diffstep\">Varying training step & Fixed diffusion step</a> <br>"
        
        return out
        
    @app.route('/fixed_trainingstep__varying_diffstep/')
    def fixed_trainingstep__varying_diffstep_choose_model():
        out = ""
        folder = f"{args.sampling_dir}/{args.exp_dir}/"
        model = glob.glob(f"{folder}/*")
        n_frame = 5
        itp = 'render_face'
        itp_method = 'Lerp'
        diff_step = 1000
        ckpt = 'ema_040000'
        for m in model:
            if 'log' in m:
                m_name = m.split('/')[-1]
                out += f"<a href=\"show_m_name={m_name}&ckpt={ckpt}&itp={itp}&itp_method={itp_method}&diff_step={diff_step}&n_frame={n_frame}&sampling=reverse&show_itmd=True\">{m_name}</a> <br>"
        return out
    
    @app.route('/varying_trainingstep__fixed_diffstep/')
    def varying_trainingstep__fixed_diffstep_choose_model():
        out = ""
        folder = f"{args.sampling_dir}/{args.exp_dir}/"
        model = glob.glob(f"{folder}/*")
        n_frame = 5
        itp = 'render_face'
        itp_method = 'Lerp'
        diff_step = 1000
        m_type = 'model'
        time_respace = 250
        for m in model:
            if 'log' in m:
                m_name = m.split('/')[-1]
                out += f"<a href=\"show_m_name={m_name}&m_type={m_type}&itp={itp}&itp_method={itp_method}&diff_step={diff_step}&time_respace={time_respace}&n_frame={n_frame}&sampling=reverse&show_itmd=True\">{m_name}</a> <br>"
        return out
    
    
    
    @app.route("/fixed_trainingstep__varying_diffstep/show_m_name=<m_name>&ckpt=<ckpt>&itp=<itp>&itp_method=<itp_method>&diff_step=<diff_step>&n_frame=<n_frame>&sampling=<sampling>&show_itmd=<show_itmd>")
    def fixed_trainingstep__varying_diffstep(m_name, itp, itp_method, diff_step, ckpt, n_frame, sampling, show_itmd):
        
        # Fixed the training step and varying the diffusion step
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        out += f"<h2>[#] Model name : {m_name} </h2>"
        out += f"<h2>[#] Interpolate on : {itp} </h2>"
        out += f"<h2>[#] Interpolate method : {itp_method} </h2>"
        out += f"<h2>[#] #N-Frames (@fps=30) : {n_frame} </h2>"
        
        
        data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_{args.res}/{args.set_}/"
        assert os.path.isfile(args.sample_pair_json)
        f = open(args.sample_pair_json)
        sample_pairs = json.load(f)['pair']
        
        # path example : /data/mint/sampling/paired_training_target/log=paired+allunet_eps+ddst_128_cfg=paired+allunet_eps+ddst_128.yaml
        # /model_040000/valid/render_face/reverse_sampling/src=62385.jpg/dst=61432.jpg
        for k, v in sample_pairs.items():
            out += "<table>"
            out += "<tr> <th> #N diffusion step </th> <th> Input </th> <th> Image </th> <th> Input </th> </tr>"
            pid = k
            src = v['src']
            dst = v['dst']
            
            out += f"[#{k}] {src}=>{dst} : <img src=/files/{data_path}/{src.replace('jpg', 'png')}>, {dst} : <img src=/files/{data_path}/{dst.replace('jpg', 'png')}>" + "<br>" + "<br>"
            path = f"{args.sampling_dir}/{args.exp_dir}/{m_name}/{ckpt}/{args.set_}/{itp}/{sampling}_sampling/src={src}/dst={dst}/"
            for time_respace in [25, 50, 100, 250, 500, 750, 1000, ""]:
                out += "<tr>"
                out += f"<td> {time_respace} </td> "
                
                if args.res == 128:
                    out += f"<td> <img src=/files/{data_path}/{src.replace('jpg', 'png')}> </td>"
                else:
                    out += f"<td> <img src=/files/{data_path}/{src}> </td>"
                
                ###################################################
                # Show results
                frames = glob.glob(f"{path}/{itp_method}_diff={diff_step}_respace={time_respace}/n_frames={n_frame}/res_*.png")
                # out += str(show_itmd)
                    
                out += "<td>"
                if len(frames) > 0:
                    frames = sort_by_frame(frames)
                    if show_itmd == "False":
                        frames = [frames[0], frames[-1]]
                    for f in frames:
                        out += "<img src=/files/" + f + ">"
                else:
                    out += "<p style=\"color:red\">Images not found!</p>"
                out += "</td>"
                ###################################################
                
                if args.res == 128:
                    out += f"<td> <img src=/files/{data_path}/{src.replace('jpg', 'png')}> </td>"
                else:
                    out += f"<td> <img src=/files/{data_path}/{src}> </td>"
                
                out += "</tr>"
                
            out += "</table>"
            out += "<br> <hr>"
                    
        return out


    @app.route("/varying_trainingstep__fixed_diffstep/show_m_name=<m_name>&m_type=<m_type>&itp=<itp>&itp_method=<itp_method>&diff_step=<diff_step>&time_respace=<time_respace>&n_frame=<n_frame>&sampling=<sampling>&show_itmd=<show_itmd>")
    def varying_trainingstep__fixed_diffstep(m_name, m_type, itp, itp_method, diff_step, time_respace, n_frame, sampling, show_itmd):
        
        # Fixed the training step and varying the diffusion step
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        out += f"<h2>[#] Model name : {m_name} </h2>"
        out += f"<h2>[#] Saving Model type : {m_type} </h2>"
        out += f"<h2>[#] Interpolate on : {itp} </h2>"
        out += f"<h2>[#] Interpolate method : {itp_method} </h2>"
        out += f"<h2>[#] Diffusion step : {diff_step} </h2>"
        out += f"<h2>[#] Time-respacing : {time_respace} </h2>"
        out += f"<h2>[#] #N-Frames (@fps=30) : {n_frame} </h2>"
        
        
        data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_{args.res}/{args.set_}/"
        assert os.path.isfile(args.sample_pair_json)
        f = open(args.sample_pair_json)
        sample_pairs = json.load(f)['pair']
        
        # path example : /data/mint/sampling/paired_training_target/log=paired+allunet_eps+ddst_128_cfg=paired+allunet_eps+ddst_128.yaml
        # /model_040000/valid/render_face/reverse_sampling/src=62385.jpg/dst=61432.jpg
        for k, v in sample_pairs.items():
            out += "<table>"
            out += "<tr> <th> #N Training step </th> <th> Input </th> <th> Image </th> <th> Input </th> </tr>"
            src = v['src']
            dst = v['dst']
            
            ckpt_step = sorted([step.split('/')[-1] for step in glob.glob(f"{args.sampling_dir}/{args.exp_dir}/{m_name}/*")])
            # out += str(ckpt_step)
            out += f"[#{k}] {src}=>{dst} : <img src=/files/{data_path}/{src.replace('jpg', 'png')}>, {dst} : <img src=/files/{data_path}/{dst.replace('jpg', 'png')}>" + "<br>" + "<br>"
            for cs in ckpt_step:
                if m_type not in cs: continue
                path = f"{args.sampling_dir}/{args.exp_dir}/{m_name}/{cs}/{args.set_}/{itp}/{sampling}_sampling/src={src}/dst={dst}/"
                out += "<tr>"
                out += f"<td> {cs} </td> "
                
                if args.res == 128:
                    out += f"<td> <img src=/files/{data_path}/{src.replace('jpg', 'png')}> </td>"
                else:
                    out += f"<td> <img src=/files/{data_path}/{src}> </td>"
                
                ###################################################
                # Show results
                frames = glob.glob(f"{path}/{itp_method}_diff={diff_step}_respace={time_respace}/n_frames={n_frame}/res_*.png")
                # out += str(show_itmd)
                    
                out += "<td>"
                if len(frames) > 0:
                    frames = sort_by_frame(frames)
                    if show_itmd == "False":
                        frames = [frames[0], frames[-1]]
                    for f in frames:
                        out += "<img src=/files/" + f + ">"
                else:
                    out += "<p style=\"color:red\">Images not found!</p>"
                out += "</td>"
                ###################################################
                
                if args.res == 128:
                    out += f"<td> <img src=/files/{data_path}/{src.replace('jpg', 'png')}> </td>"
                else:
                    out += f"<td> <img src=/files/{data_path}/{src}> </td>"
                
                out += "</tr>"
                
            out += "</table>"
            out += "<br> <hr>"
                    
        return out
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_dir', default='/data/mint/sampling')
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--sample_pair_json', required=True)
    parser.add_argument('--set_', default='valid')
    parser.add_argument('--res', default=128)
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()  
    
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)