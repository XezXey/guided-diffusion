from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', require=True)
# args = parser.parse_args()

data_path = "/data/mint/ffhq_256_with_anno/ffhq_256/valid/"

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
  out = ""
  out += "<a href=\"http://10.204.100.126:4747/Lerp/\">Lerp</a> <br>"
  out += "<a href=\"http://10.204.100.126:4747/CLS/sigma=1/\">Linear Classifier (Sigma=1)</a> <br>"
  out += "<a href=\"http://10.204.100.126:4747/CLS/sigma=10/\">Linear Classifier (Sigma=10)</a> <br>"
  return out

@app.route('/Lerp/')
def lerp():
  out = ""
  folder = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_500000/valid/light/"
  for src_path in glob.glob(f"{folder}/src=*"):
    src_id = src_path.split('/')[-1]
    for d in glob.glob(f"{folder}/{src_id}/dst=*"):
      if not os.path.isdir(d): continue
      src_id = d.split('/')[-2]
      dst_id = d.split('/')[-1]
      out += f"{src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"
      img_path = glob.glob(d + "/Lerp/*.png")
      img_path = sort_by_frame(img_path)
      for f in img_path:
        out += "<img src=/files/" + f + ">"
      out += "<br> <hr>"
  return out

@app.route('/CLS/sigma=<sigma>/')
def cls(sigma):
  out = ""
  folder = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_300000/valid/light/"
  for i, src_path in enumerate(glob.glob(f"{folder}/src=*")):
    src_id = src_path.split('/')[-1]
    for d in glob.glob(f"{folder}/{src_id}/dst=*"):
      if not os.path.isdir(d): continue
      src_id = d.split('/')[-2]
      dst_id = d.split('/')[-1]
      out += f"{i} {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"

      img_path = glob.glob(d + "/Lerp/*.png")
      img_path = sort_by_frame(img_path)
      out += "LERP <br>"
      for f in img_path:
        out += "<img src=/files/" + f + ">"
      # out += "<br>"

      out += "<br> CLS-SIGMA1 <br>"
      for sub in glob.glob(d + f"/LinearClassifier/sigma=1"):
        img_path = glob.glob(sub + "/*.png")
        img_path = sort_by_frame(img_path)
        for f in img_path:
          out += "<img src=/files/" + f + ">"

      # out += "<br>"

      out += "<br> CLS-SIGMA10 <br>"
      for sub in glob.glob(d + f"/LinearClassifier/sigma=10"):
        img_path = glob.glob(sub + "/*.png")
        img_path = sort_by_frame(img_path)
        for f in img_path:
          out += "<img src=/files/" + f + ">"


      out += "<br> <hr>"
  return out