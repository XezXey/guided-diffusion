from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--folder', require=True)
# args = parser.parse_args()

data_path = "/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/"
sampling_folder = 'hard_samples'
# sampling_folder = 'rotated_normals'

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
  out += f"<a href=\"/best_checkpoint\">Best checkpoint</a> <br>"
  out += f"<a href=\"/best_itp_method\">Best interpolation method</a> <br>"
  out += "<a href=\"/model_comparison/\">Model comparison</a> <br>"
  out += "<a href=\"/model_comparison_from_json/itp_method=Slerp/\">Model comparison (From json)</a> <br>"
  
  return out

@app.route('/best_checkpoint/')
def best_checkpoint():
  out = ""
  folder = f"/home/mint/guided-diffusion/sample_scripts/py/relighting_sample_id/ddim_reverse_interpolate/{sampling_folder}/"
  model = glob.glob(f"{folder}/*")
  for m in model:
    m_name = m.split('/')[-1]
    out += f"<a href=\"m_name={m_name}&itp=spatial_latent&itp_method=Slerp&n_frame=30\">{m_name}</a> <br>"
  return out

@app.route('/best_checkpoint/m_name=<m_name>&itp=<itp>&itp_method=<itp_method>&n_frame=<n_frame>')
def best_checkpoint_vis(m_name, itp, itp_method, n_frame=30):
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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4750, debug=True, threaded=False)