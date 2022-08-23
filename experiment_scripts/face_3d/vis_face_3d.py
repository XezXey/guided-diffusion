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


@app.route('/')
def outer():
    out = ""
    path = "/home/mint/mothership_v16/experiment_scripts/face_3d/render_face/valid/"
    image_name = sorted([ip.split('/')[-1] for ip in glob.glob(f'{path}/*')])
    render_type = sorted([ren.split('/')[-1] for ren in glob.glob(f'/{path}/{image_name[0]}/*')])

    for img_name in image_name:
        out += "<table>"
        out += "<tr>"
        out += f"<th> raw </th>"
        for ren in render_type:
            out += f"<th> {ren} </th>"
        out += "</tr>"
        out += "<tr>"
        
        out += f"<th> <img src=/files/{data_path}/{img_name} width=\"256\" height=\"256\"> </th>"
        for ren in render_type:
            # out += f"{ren} <br>"
            pic = sorted(glob.glob(f"{path}/{img_name}/{ren}/*.png"))
            for p in pic:
                out += f"<th> <img src=/files/{p}> </th>"

        out += "</tr>"
        out += "</table>"
        out += "<hr>"
    return out


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4748, debug=True, threaded=True)