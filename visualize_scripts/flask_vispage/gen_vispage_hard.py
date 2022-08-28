from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', require=True)
# args = parser.parse_args()

data_path = "/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/"

def sort_by_frame(path_list):
    frame_anno = []
    for p in path_list:
        frame_idx = os.path.splitext(p.split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
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
  hostname = request.headers.get('Host')
  out = ""
  out += f"<a href=\"/best_checkpoint\">Best checkpoint</a> <br>"
  out += f"<a href=\"/best_itp_method\">Best interpolation method</a> <br>"
  out += "<a href=\"/model_comparison/\">Model comparison</a> <br>"
  out += "<a href=\"/model_comparison_from_json/itp_method=Slerp/\">Model comparison (From json)</a> <br>"
  
  return out

@app.route('/best_checkpoint/')
def best_checkpoint():
  out = ""
  folder = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/"
  model = glob.glob(f"{folder}/*")
  for m in model:
    m_name = m.split('/')[-1]
    out += f"<a href=\"m_name={m_name}&itp=light&itp_method=Lerp\">{m_name}</a> <br>"
  return out

@app.route('/best_checkpoint/m_name=<m_name>&itp=<itp>&itp_method=<itp_method>')
def best_checkpoint_vis(m_name, itp, itp_method):
  out = ""
  out += f"<h1>[#] Model name : {m_name} </h1>"
  out += f"<h1>[#] Interpolate on : {itp} </h1>"
  out += f"<h1>[#] Interpolate method : {itp_method} </h1>"
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/{m_name}/"
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
          img_path = glob.glob(f"{ckpt}/valid/{itp}/{src_id}/{dst_id}/{itp_method}_1000/*.png")
          img_path = sort_by_frame(img_path)
          for f in img_path:
            out += "<img src=/files/" + f + ">"
          out += "<br>"
        out += "<br> <hr>"
  return out

@app.route('/best_itp_method/')
def best_itp_method():
  out = ""
  folder = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/"
  model = glob.glob(f"{folder}/*")
  for m in model:
    m_name = m.split('/')[-1]
    out += f"<h2> [#] Model name : {m_name} </h2>"
    out += "<ul>"
    for ckpt in sorted(glob.glob(f"{m}/*")):
      ckpt_name = ckpt.split('/')[-1]
      out += f"<li> <h3> <a href=\"m_name={m_name}&itp=light&ckpt={ckpt_name}\"> checkpoint at : {ckpt_name} </a> </li> </h3>"
    out += "</ul>"
    out += "<br>"
  return out

@app.route('/best_itp_method/m_name=<m_name>&itp=<itp>&ckpt=<ckpt>')
def best_itp_method_vis(m_name, itp, ckpt):
  out = ""
  out += f"<h1>[#] Model name : {m_name} </h1>"
  out += f"<h1>[#] Checkpoint at : {ckpt} </h1>"
  out += f"<h1>[#] Interpolate on : {itp} </h1>"
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/{m_name}/{ckpt}/"
  for i, src_path in enumerate(glob.glob(f"{folder}/valid/{itp}/src=*")):
      src_id = src_path.split('/')[-1]
      for dst_path in glob.glob(f"{src_path}/dst=*"):
        if not os.path.isdir(dst_path): continue
        src_id = dst_path.split('/')[-2]
        dst_id = dst_path.split('/')[-1]
        out += f"[#{i}] {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
        itp_method = glob.glob(f"{dst_path}/*")
        for itp_m in itp_method:
          if itp_m.split('/')[-1].lower() == 'combined': continue           
          out += f"{itp_m.split('/')[-1]} ===> "
          img_path = glob.glob(f"{itp_m}/*.png")
          img_path = sort_by_frame(img_path)
          for f in img_path:
            out += "<img src=/files/" + f + ">"
          out += "<br>"
        out += "<br> <hr>"
  return out

@app.route('/model_comparison/ema=<ema>&itp_method=<itp_method>/')
def model_comparison(ema, itp_method):
  out = ""
  model = glob.glob("/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/*")
  model = [m.split('/')[-1] for m in model]
  
  # folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/{model[0]}/ema_{ema}/valid/{itp}/"
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_{ema}/valid/light/"
  for i, src_path in enumerate(glob.glob(f"{folder}/src=*")):
    src_id = src_path.split('/')[-1]
    for d in glob.glob(f"{folder}/{src_id}/dst=*"):
      if not os.path.isdir(d): continue
      src_id = d.split('/')[-2]
      dst_id = d.split('/')[-1]
      out += f"[#{i}] {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
      for m in model:
        out += f"<br> {m} <br>"
        if m == "log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml":
          each_model = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/{m}/ema_{ema}/valid/light/"
        else:
          each_model = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/{m}/ema_{ema}/valid/spatial_latent/"
        for d in glob.glob(f"{each_model}/{src_id}/dst=*"):
          img_path = glob.glob(d + f"/{itp_method}_1000/*.png")
          img_path = sort_by_frame(img_path)
          for f in img_path:
            out += "<img src=/files/" + f + ">"

      out += "<br> <hr>"
  return out

@app.route('/model_comparison_from_json/itp_method=<itp_method>/')
def model_comparison_from_json(itp_method):
  out = ""
  # model = glob.glob("/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/*")
  # model = [m.split('/')[-1] for m in model]
  
  # folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/{model[0]}/ema_{ema}/valid/{itp}/"
  f = open('./model_comparison_hard.json')
  ckpt_dict = json.load(f)['model_comparison']
  model = list(ckpt_dict.keys())
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_{ckpt_dict['log=UNetCond_Spatial_Concat_Shape_cfg=UNetCond_Spatial_Concat_Shape.yaml']}/valid/light/"
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
          each_model = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/{m}/ema_{ckpt_dict[m]}/valid/light/"
        else:
          each_model = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples/{m}/ema_{ckpt_dict[m]}/valid/spatial_latent/"
        for d in glob.glob(f"{each_model}/{src_id}/dst=*"):
          img_path = glob.glob(d + f"/{itp_method}_1000/*.png")
          img_path = sort_by_frame(img_path)
          for f in img_path:
            out += "<img src=/files/" + f + ">"

      out += "<br> <hr>"
  return out

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4751, debug=True, threaded=False)