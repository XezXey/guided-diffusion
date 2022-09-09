from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--folder', require=True)
# args = parser.parse_args()

data_path = "/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/"
# sampling_folder = 'hard_samples'
# sampling_folder = 'rotated_normals'
sampling_folder = 'test_new'

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

app = Flask(__name__)
@app.route('/files/<path:path>')
def servefile(path):
  return send_from_directory('/', path)
  
@app.route('/')
def root():
  out = ""
  out += f"<a href=\"/best_checkpoint_vid\">Best checkpoint (Video)</a> <br>"
  out += f"<a href=\"/best_checkpoint_img\">Best checkpoint (Image)</a> <br>"
  # out += f"<a href=\"/best_itp_method\">Best interpolation method</a> <br>"
  # out += "<a href=\"/model_comparison/\">Model comparison</a> <br>"
  out += "<a href=\"/model_comparison_from_json/itp_method=Slerp&n_frame=5/\">Model comparison (From json)</a> <br>"
  
  return out

@app.route('/best_checkpoint_vid/')
def best_checkpoint_vid():
  out = ""
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{sampling_folder}/"
  model = glob.glob(f"{folder}/*")
  for m in model:
    m_name = m.split('/')[-1]
    out += f"<a href=\"show_m_name={m_name}&itp=spatial_latent&itp_method=Slerp&n_frame=30\">{m_name}</a> <br>"
  return out

@app.route('/best_checkpoint_vid/show_m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&n_frame=<n_frame>')
def best_checkpoint_vid_show(m_name, itp, itp_method, n_frame=30):
  out = ""
  out += f"<h1>[#] Model name : {m_name} </h1>"
  out += f"<h1>[#] Interpolate on : {itp} </h1>"
  out += f"<h1>[#] Interpolate method : {itp_method} </h1>"
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{sampling_folder}/{m_name}/"
  checkpoint = sorted(glob.glob(f"{folder}/*"))
  for i, src_path in enumerate(glob.glob(f"{checkpoint[0]}/valid/{itp}/src=*")):
      src_id = src_path.split('/')[-1]
      for dst_path in glob.glob(f"{checkpoint[0]}/valid/{itp}/{src_id}/dst=*"):
        if not os.path.isdir(dst_path): continue
        src_id = dst_path.split('/')[-2]
        dst_id = dst_path.split('/')[-1]
        out += f"[#{i}] {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
        out += "<table>"
        out += "<tr>"
        for ckpt in checkpoint:
          out += f"<td> Checkpoint at : {ckpt.split('/')[-1]} </td>"
        out += "</tr> <tr>"
        for ckpt in checkpoint:
          vid_path = glob.glob(f"{ckpt}/valid/{itp}/{src_id}/{dst_id}/{itp_method}_1000/n_frames={n_frame}/*.mp4")
          if len(vid_path) == 0:
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
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{sampling_folder}/"
  model = glob.glob(f"{folder}/*")
  for m in model:
    m_name = m.split('/')[-1]
    out += f"<a href=\"show_m_name={m_name}&itp=spatial_latent&itp_method=Slerp&n_frame=30\">{m_name}</a> <br>"
  return out

@app.route('/best_checkpoint_img/show_m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&n_frame=<n_frame>')
def best_checkpoint_img_show(m_name, itp, itp_method, n_frame=30):
  out = ""
  out += f"<h1>[#] Model name : {m_name} </h1>"
  out += f"<h1>[#] Interpolate on : {itp} </h1>"
  out += f"<h1>[#] Interpolate method : {itp_method} </h1>"
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{sampling_folder}/{m_name}/"
  checkpoint = sorted(glob.glob(f"{folder}/*"))
  for i, src_path in enumerate(glob.glob(f"{checkpoint[0]}/valid/{itp}/src=*")):
      src_id = src_path.split('/')[-1]
      for dst_path in glob.glob(f"{checkpoint[0]}/valid/{itp}/{src_id}/dst=*"):
        if not os.path.isdir(dst_path): continue
        src_id = dst_path.split('/')[-2]
        dst_id = dst_path.split('/')[-1]
        out += f"[#{i}] {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
        for ckpt in checkpoint:
          out += f"Checkpoint at : {ckpt.split('/')[-1]}"
          img_path = glob.glob(f"{ckpt}/valid/{itp}/{src_id}/{dst_id}/{itp_method}_1000/n_frames={n_frame}/*.png")
          img_path = sort_by_frame(img_path)
          for f in img_path:
            out += "<img src=/files/" + f + ">"
          out += "<br>"
        out += "<br> <hr>"
  return out

@app.route('/model_comparison_from_json/itp_method=<itp_method>&n_frame=<n_frame>/')
def model_comparison_from_json(itp_method, n_frame):
  out = ""
  # model = glob.glob("/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/*")
  # model = [m.split('/')[-1] for m in model]
  
  # folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/{model[0]}/ema_{ema}/valid/{itp}/"
  f = open('./model_comparison_hard_AdaGN.json')
  ckpt_dict = json.load(f)['model_comparison']
  model = list(ckpt_dict.keys())
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{sampling_folder}/log=Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml_cfg=Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml/ema_{ckpt_dict['log=Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml_cfg=Spatial_Hadamart_AdaGN_ReduceCh-4g_Shape.yaml']}/valid/spatial_latent/"
  for i, src_path in enumerate(glob.glob(f"{folder}/src=*")):
    src_id = src_path.split('/')[-1]
    for d in glob.glob(f"{folder}/{src_id}/dst=*"):
      if not os.path.isdir(d): continue
      src_id = d.split('/')[-2]
      dst_id = d.split('/')[-1]
      out += f"[#{i}] {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
      for m_index, m in enumerate(model):
        out += f"<br> {m_index} : {m} <br>"
        if m == "log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml":
          each_model = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{sampling_folder}/{m}/ema_{ckpt_dict[m]}/valid/light/"
        else:
          each_model = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{sampling_folder}/{m}/ema_{ckpt_dict[m]}/valid/spatial_latent/"
        for d in glob.glob(f"{each_model}/{src_id}/dst=*"):
          if m in ["log=UNetCond_Spatial_Hadamart_Tanh_Shape_cfg=UNetCond_Spatial_Hadamart_Tanh_Shape.yaml", "log=UNetCond_Spatial_Hadamart_Tanh_Shape+Bg_cfg=UNetCond_Spatial_Hadamart_Tanh_Shape+Bg.yaml"]:
            img_path = glob.glob(d + f"/{itp_method}_1000/*.png")
          else:
            img_path = glob.glob(d + f"/{itp_method}_1000/n_frames={n_frame}/*.png")
          img_path = sort_by_frame(img_path)
          for f in img_path:
            out += "<img src=/files/" + f + ">"

      out += "<br> <hr>"
  return out



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4750, debug=True, threaded=False)