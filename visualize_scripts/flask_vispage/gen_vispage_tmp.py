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
  out += "<a href=\"http://10.204.100.126:4747/best_diffusion_steps/\">Best diffusion steps</a> <br>"
  out += "<a href=\"http://10.204.100.126:4747/model_comparison/\">Model comparison</a> <br>"
  
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

@app.route('/best_diffusion_steps/')
def best_diff_steps():
  out = ""
  folder = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/best_num_step_diffusion/samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_300000/valid/light/"
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

      diffusion_steps = [250, 300, 400, 500, 600, 700]
      for step in diffusion_steps:
        img_path = glob.glob(d + f"/Lerp_{step}/*.png")
        img_path = sort_by_frame(img_path)
        out += f"<br> LERP {step}<br>"
        for f in img_path:
          out += "<img src=/files/" + f + ">"

      out += "<br> CLS-SIGMA1 <br>"
      for sub in glob.glob(d + f"/LinearClassifier/sigma=1"):
        img_path = glob.glob(sub + "/*.png")
        img_path = sort_by_frame(img_path)
        for f in img_path:
          out += "<img src=/files/" + f + ">"

      out += "<br> CLS-SIGMA10 <br>"
      for sub in glob.glob(d + f"/LinearClassifier/sigma=10"):
        img_path = glob.glob(sub + "/*.png")
        img_path = sort_by_frame(img_path)
        for f in img_path:
          out += "<img src=/files/" + f + ">"

      out += "<br> <hr>"
  return out

@app.route('/interpolate_all/')
def itp_all():
  out = ""
  # folder = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolation/samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_300000/valid/all/"
  folder = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_300000/valid/all"
  folder2 = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_300000/valid/all_itp_noise"
  # folder3 = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_300000/valid/light"
  folder3 = "/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/best_num_step_diffusion/samples/log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml/ema_300000/valid/light/"
  for i, src_path in enumerate(glob.glob(f"{folder}/src=*")):
    src_id = src_path.split('/')[-1]
    for d in glob.glob(f"{folder}/{src_id}/dst=*"):
      if not os.path.isdir(d): continue
      src_id = d.split('/')[-2]
      dst_id = d.split('/')[-1]
      out += f"{i} {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"

      diffusion_steps = [250]
      for step in diffusion_steps:
        img_path = glob.glob(d + f"/Lerp_{step}/*.png")
        img_path = sort_by_frame(img_path)
        out += f"<br> LERP {step} - w/o itp_noise (all)<br>"
        for f in img_path:
          out += "<img src=/files/" + f + ">"

        img_path = glob.glob(f"{folder2}/{src_id}/{dst_id}/" + f"/Lerp_{step}/*.png")
        img_path = sort_by_frame(img_path)
        out += f"<br> LERP {step} - w/ itp_noise (all)<br>"
        for f in img_path:
          out += "<img src=/files/" + f + ">"

        diffusion_steps = [250]
        for step in diffusion_steps:
          img_path = glob.glob(f"{folder3}/{src_id}/{dst_id}/" + f"/Lerp_{step}/*.png")
          img_path = sort_by_frame(img_path)
          out += f"<br> LERP {step} (only light)<br>"
          for f in img_path:
            out += "<img src=/files/" + f + ">"


        img_path = glob.glob(f"{folder3}/{src_id}/{dst_id}/" + "/Lerp/*.png")
        img_path = sort_by_frame(img_path)
        out += "<br> LERP 1000 (only light)<br>"
        for f in img_path:
          out += "<img src=/files/" + f + ">"

      out += "<br> <hr>"
  return out


# @app.route('/model_comparison/')
# def model_comparison():
#   out = ""
#   model_folder = ['log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml', 'log=DuplicateUNetCond_lastconv=False_cfg=DuplicateUNetCond_lastconv=False.yaml', 'log=UNetCond_blur1_dim256_cfg=UNetCond_blur1_dim128.yaml', 'log=UNetCond_blur1_dim256_cfg=UNetCond_blur1_dim256.yaml']
#   for model in model_folder:
#     folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/{model}/ema_300000/valid/light/"
#     out += f"{model} <br>"
#     for i, src_path in enumerate(glob.glob(f"{folder}/src=*")):
#       src_id = src_path.split('/')[-1]
#       for d in glob.glob(f"{folder}/{src_id}/dst=*"):
#         if not os.path.isdir(d): continue
#         src_id = d.split('/')[-2]
#         dst_id = d.split('/')[-1]
#         out += f"{i} {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"

#         img_path = glob.glob(d + "/Lerp/*.png")
#         img_path = sort_by_frame(img_path)
#         out += "LERP <br>"
#         for f in img_path:
#           out += "<img src=/files/" + f + ">"

#         diffusion_steps = [450]
#         for step in diffusion_steps:
#           img_path = glob.glob(d + f"/Lerp_{step}/*.png")
#           img_path = sort_by_frame(img_path)
#           out += f"<br> LERP {step}<br>"
#           for f in img_path:
#             out += "<img src=/files/" + f + ">"

#         out += "<br> <hr>"
#   return out

@app.route('/model_comparison/')
def model_comparison():
  out = ""
  model = ['log=cond_img64_by_deca_arcface_cfg=cond_img64_by_deca_arcface.yaml', 'log=DuplicateUNetCond_lastconv=False_cfg=DuplicateUNetCond_lastconv=False.yaml', 'log=UNetCond_blur1_dim27_cfg=UNetCond_blur1_dim27.yaml', 'log=UNetCond_blur1_dim128_cfg=UNetCond_blur1_dim128.yaml', 'log=UNetCond_blur1_dim256_cfg=UNetCond_blur1_dim256.yaml']
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/{model[0]}/ema_300000/valid/light/"
  for i, src_path in enumerate(glob.glob(f"{folder}/src=*")):
    src_id = src_path.split('/')[-1]
    for d in glob.glob(f"{folder}/{src_id}/dst=*"):
      if not os.path.isdir(d): continue
      src_id = d.split('/')[-2]
      dst_id = d.split('/')[-1]
      out += f"{i} {src_id} : <img src=/files/{data_path}/{src_id.split('=')[-1]} width=\"64\" height=\"64\">, {dst_id} : <img src=/files/{data_path}/{dst_id.split('=')[-1]} width=\"64\" height=\"64\">" + "<br>" + "<br>"

      for m in model:

        out += f"<br> {m} <br>"
        if m == 'log=UNetCond_blur1_dim27_cfg=UNetCond_blur1_dim27.yaml':
          folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/{m}/ema_150000/valid/light/"
        else:
          folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/samples/{m}/ema_300000/valid/light/"
        for d in glob.glob(f"{folder}/{src_id}/dst=*"):

          img_path = glob.glob(d + "/Lerp/*.png")
          img_path = sort_by_frame(img_path)
          for f in img_path:
            out += "<img src=/files/" + f + ">"

          diffusion_steps = [450]
          for step in diffusion_steps:
            img_path = glob.glob(d + f"/Lerp_{step}/*.png")
            img_path = sort_by_frame(img_path)
            for f in img_path:
              out += "<img src=/files/" + f + ">"

      out += "<br> <hr>"
  return out

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4750, debug=True, threaded=False)