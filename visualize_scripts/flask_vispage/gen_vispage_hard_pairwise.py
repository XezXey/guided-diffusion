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
  out += "<a href=\"/pairwise/itp_method=Slerp/\"> Hard samples - Pairwise</a> <br>"
  
  return out

@app.route('/pairwise/itp_method=<itp_method>/src-dst/')
def model_comparison_from_json(itp_method):
  diffusion_steps = 1000
  out = """<style>
    table, th, tr, td {
      border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
    }
  </style>"""
  # model = glob.glob("/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/*")
  # model = [m.split('/')[-1] for m in model]
  
  # folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/{model[0]}/ema_{ema}/valid/{itp}/"
  samples_folder = "hard_samples_pairwise"
  f = open('./model_comparison_hard_pairwise.json')
  ckpt_dict = json.load(f)['model_comparison']
  
  f = open('/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples_pairwise.json')
  sample_pairs = json.load(f)['hard_samples_pairwise']
  sj_src_list = sample_pairs['src']
  sj_dst_list = sample_pairs['dst']
  
  
  # folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{samples_folder}/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_{ckpt_dict['log=UNetCond_Spatial_Concat_Shape_cfg=UNetCond_Spatial_Concat_Shape.yaml']}/valid/light/"
  model_name = "log=UNetCond_Spatial_Hadamart_Tanh_Shape_cfg=UNetCond_Spatial_Hadamart_Tanh_Shape.yaml"
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{samples_folder}/{model_name}/ema_{ckpt_dict[model_name]}/valid/spatial_latent/"
  
  out += f"<h2> {model_name} <h2>"
  out += "<table>"
  out += "<tr> <td> Source\Destination <td>"
  for i, sj_dst in enumerate(sj_dst_list):
      out += f"<td>[#{i}] <img src=/files/{data_path}/{sj_dst} width=\"64\" height=\"64\"> <td>"
  out += "<tr>"
  
  for i, sj_src in enumerate(sj_src_list):
    
    out += "<tr>"
    out += f"<td> [#{i}] <img src=/files/{data_path}/{sj_src} width=\"64\" height=\"64\"> <td>"
    for sj_dst in sj_dst_list:
      src_id = sj_src
      dst_id = sj_dst
      each_model = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{samples_folder}/{model_name}/ema_{ckpt_dict[model_name]}/valid/spatial_latent/"
      
      img_path = glob.glob(f"{each_model}/src={src_id}/dst={dst_id}/{itp_method}_{diffusion_steps}/*.png")
      out += "<td>"
      if len(img_path) > 0:
        img_path = sort_by_frame(img_path)
        out += "<img src=/files/" + img_path[-1] + ">"
      
      img_path = glob.glob(f"{each_model}/src={dst_id}/dst={src_id}/{itp_method}_{diffusion_steps}/*.png")
      if len(img_path) > 0:
        img_path = sort_by_frame(img_path)
        out += "<img src=/files/" + img_path[-1] + ">"
      out += "<td>"
    out += "<tr>"
  out += "</table>"
  return out

@app.route('/pairwise/itp_method=<itp_method>/dst-src/')
def model_comparison_from_json(itp_method):
  diffusion_steps = 1000
  out = """<style>
    table, th, tr, td {
      border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
    }
  </style>"""
  # model = glob.glob("/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/*")
  # model = [m.split('/')[-1] for m in model]
  
  # folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/{model[0]}/ema_{ema}/valid/{itp}/"
  samples_folder = "hard_samples_pairwise"
  f = open('./model_comparison_hard_pairwise.json')
  ckpt_dict = json.load(f)['model_comparison']
  
  f = open('/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/hard_samples_pairwise.json')
  sample_pairs = json.load(f)['hard_samples_pairwise']
  sj_src_list = sample_pairs['src']
  sj_dst_list = sample_pairs['dst']
  
  
  # folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{samples_folder}/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_{ckpt_dict['log=UNetCond_Spatial_Concat_Shape_cfg=UNetCond_Spatial_Concat_Shape.yaml']}/valid/light/"
  model_name = "log=UNetCond_Spatial_Hadamart_Tanh_Shape_cfg=UNetCond_Spatial_Hadamart_Tanh_Shape.yaml"
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{samples_folder}/{model_name}/ema_{ckpt_dict[model_name]}/valid/spatial_latent/"
  
  out += f"<h2> {model_name} <h2>"
  out += "<table>"
  out += "<tr> <td> Source\Destination <td>"
  for i, sj_dst in enumerate(sj_dst_list):
      out += f"<td>[#{i}] <img src=/files/{data_path}/{sj_dst} width=\"64\" height=\"64\"> <td>"
  out += "<tr>"
  
  for i, sj_src in enumerate(sj_src_list):
    
    out += "<tr>"
    out += f"<td> [#{i}] <img src=/files/{data_path}/{sj_src} width=\"64\" height=\"64\"> <td>"
    for sj_dst in sj_dst_list:
      src_id = sj_src
      dst_id = sj_dst
      each_model = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{samples_folder}/{model_name}/ema_{ckpt_dict[model_name]}/valid/spatial_latent/"
      
      img_path = glob.glob(f"{each_model}/src={src_id}/dst={dst_id}/{itp_method}_{diffusion_steps}/*.png")
      out += "<td>"
      if len(img_path) > 0:
        img_path = sort_by_frame(img_path)
        out += "<img src=/files/" + img_path[-1] + ">"
      
      img_path = glob.glob(f"{each_model}/src={dst_id}/dst={src_id}/{itp_method}_{diffusion_steps}/*.png")
      if len(img_path) > 0:
        img_path = sort_by_frame(img_path)
        out += "<img src=/files/" + img_path[-1] + ">"
      out += "<td>"
    out += "<tr>"
  out += "</table>"
  return out

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4753, debug=True, threaded=False)