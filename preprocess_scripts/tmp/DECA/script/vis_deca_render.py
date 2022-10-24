from flask import Flask, request, send_file, send_from_directory
import glob, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/mint/guided_diffusion')
from sample_scripts.sample_utils.file_utils import list_path_to_dict
import matplotlib.pyplot as plt
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', require=True)
# args = parser.parse_args()


app = Flask(__name__)
@app.route('/files/<path:path>')
def servefile(path):
  return send_from_directory('/', path)

@app.route('/valid')
def valid():
  out = """
    <style>
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
    }
    </style>
  """

  data_path = "/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/"
  src_image = sorted(glob.glob(f'/{data_path}*.jpg'))
  image_name = [x.split('/')[-1] for x in src_image]
  
  folder_pred = "/data/mint/DPM_Dataset/ffhq_256_with_anno/rendered_images/deca_shape_images/valid/"
  folder_template = "/data/mint/DPM_Dataset/ffhq_256_with_anno/rendered_images/deca_template_shape_images/valid/"
  folder_pred_verifying = "/data/mint/ffhq_256_with_anno_dev/for_verifying_shape_images/valid/"
  folder_template_verifying = "/data/mint/ffhq_256_with_anno_dev/for_verifying_template_shape_images/valid/"

  img_pred = list_path_to_dict(glob.glob(f'{folder_pred}/*.png'), force_type='.jpg')
  img_template = list_path_to_dict(glob.glob(f'{folder_template}/*.png'), force_type='.jpg')
  img_pred_verifying = list_path_to_dict(glob.glob(f'{folder_pred_verifying}/*.png'), force_type='.jpg')
  img_template_verifying = list_path_to_dict(glob.glob(f'{folder_template_verifying}/*.png'), force_type='.jpg')
  anno = ["Raw", "Predicted Face", "Predicted Face (Verifying sample)", "Template Face", "Template Face (Verifying sample)", "Predicted Face (Difference)", "Template Face (Difference)"]
  
  for i, name in enumerate(image_name):
    if i == 300: break
        
    out += f"{name} <br>"
    out += "<table>"
    out += "<tr>"
    for a in anno:
      out += f"<th> {a} </th>"
    out += "<tr>"
    out += "<tr>"
    out += f"<td> <img src=/files/{data_path}/{name} width=\"256\" height=\"256\"> </td>"
    if name in img_pred.keys():
      pred_image = np.array(Image.open(img_pred[name]))
      max_min =  f"<br> {np.min(pred_image, axis=(0, 1))} - {np.max(pred_image, axis=(0, 1))}"
      out += f"<td> <img src=/files/{img_pred[name]}> {max_min} </td>"
    if name in img_pred_verifying.keys():
      verifying_pred_image = np.array(Image.open(img_pred_verifying[name]))
      max_min = f"<br> {np.min(verifying_pred_image, axis=(0, 1))} - {np.max(verifying_pred_image, axis=(0, 1))}"
      out += f"<td> <img src=/files/{img_pred_verifying[name]}> {max_min} </td>"
    if name in img_template.keys():
      template_image = np.array(Image.open(img_template[name]))
      max_min = f"<br> {np.min(template_image, axis=(0, 1))} - {np.max(template_image, axis=(0, 1))}"
      out += f"<td> <img src=/files/{img_template[name]}> {max_min} </td>"
    if name in img_template_verifying.keys():
      verifying_template_image = np.array(Image.open(img_template_verifying[name]))
      max_min = f"<br> {np.min(verifying_template_image, axis=(0, 1))} - {np.max(verifying_template_image, axis=(0, 1))}"
      out += f"<td> <img src=/files/{img_template_verifying[name]}> {max_min} </td>"
      
    out += "<td>"
    if name in img_pred.keys() and name in img_pred_verifying.keys():
      status = np.allclose(verifying_pred_image, pred_image, rtol=1e-3, atol=1e-3)
      Image.fromarray(verifying_pred_image - pred_image).save(f'/data/mint/difference/valid/pred_{name}')
      out += f"<br> <img src=/files/data/mint/difference/valid/pred_{name}><br> Allclose : {status} "
    out += "</td>"
    out += "<td>"
    if name in img_template.keys() and name in img_template_verifying.keys():
      status = np.allclose(verifying_template_image, template_image, rtol=1e-3, atol=1e-3)
      Image.fromarray(verifying_template_image - template_image).save(f'/data/mint/difference/valid/template_{name}')
      out += f"<br> <img src=/files/data/mint/difference/valid/template_{name}><br> Allclose : {status} "
    out += "</td>"
    
    out += "</tr>"
    out += "</table>"
    out += "<hr>"
    
  return out


@app.route('/train')
def train():
  out = """
    <style>
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
    }
    </style>
  """
  data_path = "/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/train/"
  src_image = sorted(glob.glob(f'/{data_path}*.jpg'))
  image_name = [x.split('/')[-1] for x in src_image]
  
  # folder_pred = "/data/mint/ffhq_256_with_anno/rendered_images/shape_images/valid/"
  # folder_template = "/data/mint/ffhq_256_with_anno/rendered_images/template_shape_images/valid/"
  folder_pred = "/data/mint/DPM_Dataset/ffhq_256_with_anno/rendered_images/deca_shape_images/train/"
  folder_template = "/data/mint/DPM_Dataset/ffhq_256_with_anno/rendered_images/deca_template_shape_images/train/"
  folder_pred_verifying = "/data/mint/ffhq_256_with_anno_dev/for_verifying_shape_images/train/"
  folder_template_verifying = "/data/mint/ffhq_256_with_anno_dev/for_verifying_template_shape_images/train/"

  img_pred = list_path_to_dict(glob.glob(f'{folder_pred}/*.png'), force_type='.jpg')
  img_template = list_path_to_dict(glob.glob(f'{folder_template}/*.png'), force_type='.jpg')
  img_pred_verifying = list_path_to_dict(glob.glob(f'{folder_pred_verifying}/*.png'), force_type='.jpg')
  img_template_verifying = list_path_to_dict(glob.glob(f'{folder_template_verifying}/*.png'), force_type='.jpg')
  anno = ["Raw", "Predicted Face", "Predicted Face (Verifying sample)", "Template Face", "Template Face (Verifying sample)", "Predicted Face (Difference)", "Template Face (Difference)"]
  
  for i, name in enumerate(image_name):
    if i == 300: break
        
    out += f"{name} <br>"
    out += "<table>"
    out += "<tr>"
    for a in anno:
      out += f"<th> {a} </th>"
    out += "<tr>"
    out += "<tr>"
    out += f"<td> <img src=/files/{data_path}/{name} width=\"256\" height=\"256\"> </td>"
    if name in img_pred.keys():
      pred_image = np.array(Image.open(img_pred[name]))
      max_min =  f"<br> {np.min(pred_image, axis=(0, 1))} - {np.max(pred_image, axis=(0, 1))}"
      out += f"<td> <img src=/files/{img_pred[name]}> {max_min} </td>"
    if name in img_pred_verifying.keys():
      verifying_pred_image = np.array(Image.open(img_pred_verifying[name]))
      max_min = f"<br> {np.min(verifying_pred_image, axis=(0, 1))} - {np.max(verifying_pred_image, axis=(0, 1))}"
      out += f"<td> <img src=/files/{img_pred_verifying[name]}> {max_min} </td>"
    if name in img_template.keys():
      template_image = np.array(Image.open(img_template[name]))
      max_min = f"<br> {np.min(template_image, axis=(0, 1))} - {np.max(template_image, axis=(0, 1))}"
      out += f"<td> <img src=/files/{img_template[name]}> {max_min} </td>"
    if name in img_template_verifying.keys():
      verifying_template_image = np.array(Image.open(img_template_verifying[name]))
      max_min = f"<br> {np.min(verifying_template_image, axis=(0, 1))} - {np.max(verifying_template_image, axis=(0, 1))}"
      out += f"<td> <img src=/files/{img_template_verifying[name]}> {max_min} </td>"
      
    out += "<td>"
    if name in img_pred.keys() and name in img_pred_verifying.keys():
      status = np.allclose(verifying_pred_image, pred_image, rtol=1e-1, atol=1e-1)
      Image.fromarray(verifying_pred_image - pred_image).save(f'/data/mint/difference/train/pred_{name}')
      out += f"<br> <img src=/files/data/mint/difference/train/pred_{name}><br> Allclose : {status} "
    out += "</td>"
    out += "<td>"
    if name in img_template.keys() and name in img_template_verifying.keys():
      status = np.allclose(verifying_template_image, template_image, rtol=1e-1, atol=1e-1)
      Image.fromarray(verifying_template_image - template_image).save(f'/data/mint/difference/train/template_{name}')
      out += f"<br> <img src=/files/data/mint/difference/train/template_{name}><br> Allclose : {status} "
    out += "</td>"
    
    out += "</tr>"
    out += "</table>"
    out += "<hr>"
    
  return out

if __name__ == "__main__":
    os.makedirs('/data/mint/difference/train/', exist_ok=True)
    os.makedirs('/data/mint/difference/valid/', exist_ok=True)
    app.run(host='0.0.0.0', port=4751, debug=True, threaded=False)