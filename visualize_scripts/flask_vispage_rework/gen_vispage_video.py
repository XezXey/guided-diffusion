from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/sample_utils/')
import mani_utils, file_utils


data_path = "/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/"
img_path = file_utils._list_image_files_recursively(data_path)

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
        out += f"<a href=\"/model_comparison_from_json/itp_method=Slerp&n_frame=5/\">Model comparison (From json)</a> <br>"
        return out

    @app.route('/best_checkpoint_vid/')
    def best_checkpoint_vid():
        out = ""
        folder = f"{args.sample_dir}/{args.exp_dir}/"
        model = glob.glob(f"{folder}/*")
        for m in model:
            m_name = m.split('/')[-1]
            out += f"<a href=\"show_m_name={m_name}&itp=spatial_latent&itp_method=Slerp&n_frame=30\">{m_name}</a> <br>"
        return out

    @app.route('/best_checkpoint_vid/show_m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&n_frame=<n_frame>')
    def best_checkpoint_vid_show(m_name, itp, itp_method, n_frame=30):
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
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_dir}/hard_samples.json", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path, 
                                                       n_subject=-1)
        for idx, src_dst in enumerate(subject_id):
            out += f"[#{idx}] {src_dst[0]} : <img src=/files/{data_path}/{src_dst[0].split('=')[-1]} width=\"64\" height=\"64\">, {src_dst[1]} : <img src=/files/{data_path}/{src_dst[1].split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
            out += "<table>"
            for ckpt in checkpoint:
                out += f"<th> Checkpoint at : {ckpt.split('/')[-1]} </th>"
            out += "<tr>"
            for ckpt in checkpoint:
                vid_path = glob.glob(f"{ckpt}/valid/{itp}/src={src_dst[0]}/dst={src_dst[1]}/{itp_method}_1000/n_frames={n_frame}/*.mp4")
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
            out += f"<a href=\"show_m_name={m_name}&itp=spatial_latent&itp_method=Slerp&n_frame=30\">{m_name}</a> <br>"
        return out

    @app.route('/best_checkpoint_img/show_m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&n_frame=<n_frame>')
    def best_checkpoint_img_show(m_name, itp, itp_method, n_frame=30):
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
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_dir}/hard_samples.json", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path, 
                                                       n_subject=-1)
        
        for idx, src_dst in enumerate(subject_id):
            out += "<table>"
            out += f"[#{idx}] {src_dst[0]} : <img src=/files/{data_path}/{src_dst[0].split('=')[-1]} width=\"64\" height=\"64\">, {src_dst[1]} : <img src=/files/{data_path}/{src_dst[1].split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
            for ckpt in checkpoint:
                out += "<tr>"
                out += f"<td> Checkpoint at : {ckpt.split('/')[-1]} </td> "
                frames = glob.glob(f"{ckpt}/valid/{itp}/src={src_dst[0]}/dst={src_dst[1]}/{itp_method}_1000/n_frames={n_frame}/*.png")
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

    @app.route('/model_comparison_from_json/itp_method=<itp_method>&n_frame=<n_frame>/')
    def model_comparison_from_json(itp_method, n_frame):
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        f = open('./model_comparison_hard_AdaGN.json')
        ckpt_dict = json.load(f)
        model = list(ckpt_dict.keys())
        
        _, subject_id, _ = mani_utils.get_samples_list(sample_pair_json=f"{args.sample_dir}/hard_samples.json", 
                                                       sample_pair_mode='pair', 
                                                       src_dst=None, 
                                                       img_path=img_path, 
                                                       n_subject=-1)
        
        for idx, src_dst in enumerate(subject_id):
            out += "<table>"
            out += f"[#{idx}] {src_dst[0]} : <img src=/files/{data_path}/{src_dst[0].split('=')[-1]} width=\"64\" height=\"64\">, {src_dst[1]} : <img src=/files/{data_path}/{src_dst[1].split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
            for m_index, m in enumerate(model):
                out += "<tr>"
                out += f"<td> <br> {m_index} : {ckpt_dict[m]['alias']} <br> </td>"
                step = ckpt_dict[m]['step']
                itp = ckpt_dict[m]['itp']
                each_model = f"{args.sample_dir}/{args.exp_dir}/{m}/ema_{step}/valid/{itp}/src={src_dst[0]}/dst={src_dst[1]}/"
                if m in ["log=UNetCond_Spatial_Hadamart_Tanh_Shape_cfg=UNetCond_Spatial_Hadamart_Tanh_Shape.yaml", "log=UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_cfg=UNetCond_Spatial_Hadamart_Tanh_Shape+Bg.yaml"]:
                    frames = glob.glob(f"{each_model}/{itp_method}_1000/*.png")
                    vid_path = glob.glob(f"{each_model}/{itp_method}_1000/*.mp4")
                else:
                    frames = glob.glob(f"{each_model}/{itp_method}_1000/n_frames={n_frame}/*.png")
                    vid_path = glob.glob(f"{each_model}/{itp_method}_1000/n_frames={n_frame}/*.mp4")
                    
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
            
        # for i, src_path in enumerate(glob.glob(f"{folder}/src=*")):
        #     src_id = src_path.split('/')[-1]
        #     for d in glob.glob(f"{folder}/{src_id}/dst=*"):
        #         if not os.path.isdir(d): continue
        #         src_id = d.split('/')[-2]
        #         dst_id = d.split('/')[-1]
        #         out += f"[#{i}] {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
        #         for m_index, m in enumerate(model):
        #             out += f"<br> {m_index} : {m} <br>"
        #             if m == "log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml":
        #                 each_model = f"{args.sample_dir}/{args.exp_dir}/{m}/ema_{ckpt_dict[m]}/valid/light/"
        #             else:
        #                 each_model = f"/{args.sample_dir}/{args.exp_dir}/{m}/ema_{ckpt_dict[m]}/valid/spatial_latent/"
        #             for d in glob.glob(f"{each_model}/{src_id}/dst=*"):
        #                 if m in ["log=UNetCond_Spatial_Hadamart_Tanh_Shape_cfg=UNetCond_Spatial_Hadamart_Tanh_Shape.yaml", "log=UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_cfg=UNetCond_Spatial_Hadamart_Tanh_Shape+Bg.yaml"]:
        #                     img_path = glob.glob(d + f"/{itp_method}_1000/*.png")
        #                 else:
        #                     img_path = glob.glob(d + f"/{itp_method}_1000/n_frames={n_frame}/*.png")
        #                 img_path = sort_by_frame(img_path)
        #                 for f in img_path:
        #                        out += "<img src=/files/" + f + ">"

        #     out += "<br> <hr>"
        # return out
    return app



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--sample_dir', default="/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate")
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)