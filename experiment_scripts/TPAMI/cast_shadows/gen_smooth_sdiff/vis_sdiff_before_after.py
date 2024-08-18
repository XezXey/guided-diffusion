import numpy as np
from flask import Flask, request, send_file, send_from_directory
import glob, os, re
import numpy as np
import json
import sys
import argparse
import pandas as pd
    
parser = argparse.ArgumentParser()
parser.add_argument('--set_', default='train')
parser.add_argument('--port', required=True)
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--rank_shadow_c', action='store_true', default=False)
parser.add_argument('--rank_shadow_iou', action='store_true', default=False)
argsp = parser.parse_args()


def sort_by_frame(path_list):
    frame_anno = []
    for p in path_list:
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
        sort = request.args.get('sort', 'desc')
        s = request.args.get('s', 0)
        e = request.args.get('e', 10)
        print(sort, s, e)

        set_ = argsp.set_
        out = ""
        
        before_pth = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_with_weight_simplified/vis/{set_}/'
        after_pth = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/vis/{set_}/'

        stregthen_pth = f'/data/mint/DPM_Dataset/Soften_Strengthen_Shadows/TPAMI/FFHQ_shadow_face/log=difareli_canny=153to204bg_256_vll_cfg=difareli_canny=153to204bg_256_vll.yaml_tomax_steps=50/ema_085000/train_sub/'
        soften_pth = f'/data/mint/DPM_Dataset/Soften_Strengthen_Shadows/TPAMI/FFHQ_diffuse_face/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml_tomin_steps=50/ema_085000/train_sub/'


        all_path = sorted(glob.glob(f'{before_pth}/*.png'))
        min_c = -4.989461
        max_c =  8.481700

        if argsp.rank_shadow_c:
            c = pd.read_csv(f'/data/mint/DPM_Dataset/ffhq_256_with_anno/params/{argsp.set_}/ffhq-{argsp.set_}-shadow-anno.txt', sep=' ', header=None, names=['image_name', 'c_val'])
            c_sorted = c.sort_values(by=['c_val'], ascending=False if sort == 'desc' else True)
            c_sorted_norm = c_sorted.copy()
            c_sorted_norm['c_val'] = (c_sorted['c_val'] - min_c) / (max_c - min_c)
            # Filter out images that do not exist
            c_sorted = c_sorted[c_sorted['image_name'].isin([img_path.split('/')[-1].split('.')[0] + '.jpg' for img_path in all_path])]
            all_path = [f'{before_pth}/{img_name}.png' for img_name in c_sorted['image_name']]
            print(c_sorted)
        # if argsp.rank_shadow_iou:
        #     c = pd.read_csv('./iou.csv', sep=',', header=None, skiprows=1, names=['image_name', 'IOU'])
        #     c_sorted = c.sort_values(by=['IOU'], ascending=False)

        
        for img_path in all_path[int(s):int(e)]:
            img_name = img_path.split('/')[-1].split('.')[0]
            if not os.path.exists(f'{before_pth}/{img_name}.png'):
                continue
            
            value = c_sorted[c_sorted['image_name'] == f'{img_name}.jpg'].values[0][1]
            out += f"<h1> Image: {img_name} => {value} ({(value - min_c) / (max_c - min_c)}) </h1>"

            # Shadow diff
            before_tp = f'{before_pth}/{img_name}.png'
            out += f"<img src=/files/{before_tp} width=\"768\" height=\"256\"><br>"

            after_tp = f'{after_pth}/{img_name}.png'
            out += f"<img src=/files/{after_tp} width=\"768\" height=\"256\"><br>"

            # Strengthen shadow
            # Find subfolder which has 5000 images interval and has name in train_{start}_to_{end}/
            # Where (start, end) can be (0, 5000), (5000, 10000), (10000, 15000), (15000, 20000), (20000, 25000), ...
            idx = int(img_name) // 5000
            tmp = [[0, 5000], [5000, 10000], [10000, 15000], [15000, 20000], [20000, 25000], [25000, 30000], [30000, 35000], [35000, 40000], [40000, 45000], [45000, 50000], [50000, 55000], [55000, 60000]]
            # print(idx, img_name)
            x = tmp[idx]
            # print(x)
            sub_f = f'train_{x[0]}_to_{x[1]}'

            strengthen_pth_tp = sort_by_frame(glob.glob(f'{stregthen_pth}/{sub_f}/shadow/reverse_sampling/src={img_name}.jpg/dst={x[0]}.jpg/Lerp_1000/n_frames=5/res_frame*.png'))
            # print(f'{stregthen_pth}/{sub_f}/shadow/reverse_sampling/src={img_name}.jpg/dst={x[0]}.jpg/Lerp_1000/n_frames=5/res_frame*.png')
            out += f"<h2> Strengthen Shadow ===> </h2>"
            if len(strengthen_pth_tp) > 0:
                for tp in strengthen_pth_tp:
                    out += f"<img src=/files/{tp} width=\"128\" height=\"128\">"
            out += "<br>"

            # print(f'{soften_pth}/{sub_f}/shadow/reverse_sampling/src={img_name}.jpg/dst={x[0]}.jpg/Lerp_1000/n_frames=3/res_frame*.png')
            soften_pth_tp = sort_by_frame(glob.glob(f'{soften_pth}/{sub_f}/shadow/reverse_sampling/src={img_name}.jpg/dst={x[0]}.jpg/Lerp_1000/n_frames=3/res_frame*.png'))
            out += f"<h2> Soften Shadow ===> </h2>"
            if len(soften_pth_tp) > 0:
                for tp in soften_pth_tp:
                    out += f"<img src=/files/{tp} width=\"128\" height=\"128\">"
            out += "<br>"
                
        return out
    
    return app
    
if __name__ == "__main__":
     
    app = create_app()
    app.run(host=argsp.host, port=argsp.port, debug=True, threaded=False)