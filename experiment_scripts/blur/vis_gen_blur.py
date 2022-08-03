from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', require=True)
# args = parser.parse_args()

data_path = "/data/mint/ffhq_256_with_anno/ffhq_256/valid/"

app = Flask(__name__)
@app.route('/files/<path:path>')
def servefile(path):
  return send_from_directory('/', path)


@app.route('/outer/')
def outer():
  out = ""
  path = "/home/mint/guided-diffusion/experiment_scripts/blur/gen/"
  image_name = sorted([ip.split('/')[-1] for ip in glob.glob(f'{path}/*')])
  segment = [seg.split('/')[-1] for seg in glob.glob(f'/{path}/{image_name[0]}/*')]

  for seg in segment:
    out += f"{seg} <br>"
    for img_name in image_name:
      color = glob.glob(f"{path}/{img_name}/{seg}/*")
      for c in color:
        pic = sorted(glob.glob(f"{path}/{img_name}/{seg}/{c.split('/')[-1]}/*.png"))
        for p in pic:
          out += "<img src=/files/" + f"{p}" + ">"
        out += "<br>"
      out += "<br>"
    out += "<hr>"
  return out

@app.route('/inner/')
def inner():
  out = ""
  path = "/home/mint/guided-diffusion/experiment_scripts/blur/gen/"
  image_name = sorted([ip.split('/')[-1] for ip in glob.glob(f'{path}/*')])
  segment = [seg.split('/')[-1] for seg in glob.glob(f'/{path}/{image_name[0]}/*')]

  for seg in segment:
    out += f"{seg} <br>"
    color = glob.glob(f"{path}/{image_name[0]}/{seg}/*")
    for c in color:
      out += f"{c.split('/')[-1]} <br>"
      for img_name in image_name:
        pic = sorted(glob.glob(f"{path}/{img_name}/{seg}/{c.split('/')[-1]}/*.png"))
        for p in pic:
          out += "<img src=/files/" + f"{p}" + ">"
        out += "<br>"
      out += "<br>"
    out += "<hr>"
  return out

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4748, debug=True, threaded=True)