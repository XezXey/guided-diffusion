from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
sys.path.insert(0, '/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/sample_utils/')
import mani_utils, file_utils
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', required=True)
parser.add_argument('--dataset_name', default='ffhq_rotate')
parser.add_argument('--sampling_dir', default='/data/mint/sampling')
parser.add_argument('--exp_dir', default='')
parser.add_argument('--sample_pair_json', required=True)
parser.add_argument('--comparison_candidate', required=True)
parser.add_argument('--set_', default='valid')
parser.add_argument('--res', default=256)
parser.add_argument('--port', required=True)
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--n_frames', default=60)
args = parser.parse_args()

def sort_by_frame(path_list):
    frame_anno = []
    for p in path_list:
        frame_idx = os.path.splitext(p.split('/')[-1].split('_')[-1])[0]  # file format is m_000.png, ...
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
        out = f"<h1> Comparison: {args.comparison_candidate} </h1>"
        out += f"<a href=\"/model_compare/\"> Model Comparison </a> <br>"
        
        return out
        
    @app.route("/model_compare/")
    def model_compare():
        # Fixed the training step and varying the diffusion step

        # border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;
                }
                .file-name {
                    font-size: 16px;
                    color: gray;
                }
                </style>"""
        
        out += "<script>"
        out += """
        function transposeTable(table) {
            var transposedTable = document.createElement("table");

            for (var i = 0; i < table.rows[0].cells.length; i++) {
                var newRow = transposedTable.insertRow(i);

                for (var j = 0; j < table.rows.length; j++) {
                var newCell = newRow.insertCell(j);
                newCell.innerHTML = table.rows[j].cells[i].innerHTML;
                }
            }

            table.parentNode.replaceChild(transposedTable, table);
        }

        function transposeAllTables() {
            var tables = document.getElementsByTagName("table");

            for (var i = 0; i < tables.length; i++) {
                transposeTable(tables[i]);
            }
        }

        """
        out += "</script>"
        
        show_vid = request.args.get('show_vid', "True")
        show_map_centered = request.args.get('show_map_centered', "False")
        show_map_clean = request.args.get('show_map_clean', "False")
        show_map_ball = request.args.get('show_ball', "False")
        show_map_ball_transp = request.args.get('show_ball_transp', "False")
        show_all_frames = request.args.get('show_all_frames', "False")
        n_frame = request.args.get('n_frame', None)
        s = request.args.get('s', 0)
        e = request.args.get('e', 100)
        ds = int(request.args.get('ds', 5))
        sample_json = str(request.args.get('sample_json', args.sample_pair_json))
        model_json = str(request.args.get('model_json', args.comparison_candidate))
        
        data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_{args.res}/{args.set_}/"
        try:
            os.path.isfile(sample_json)
            f = open(sample_json)
            sample_pairs = json.load(f)['pair']
        except:
            raise ValueError(f"Sample json file not found: {sample_json}")
        
        out += f"<h2> Sample json file: {sample_json} {n_frame} </h2>"
        out += "Transpose : <button onclick='transposeAllTables()'>Transpose</button>"
        
        # path example : /data/mint/sampling/FFHQ_Reshadow_mintomax/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml_steps50/ema_085000/valid/shadow/reverse_sampling/src=60000.jpg/dst=60000.jpg 
        f = open(model_json)
        candidates = json.load(f)
        print(candidates)
        
        count = 0
        to_show = list(sample_pairs.items())[int(s):int(e)]
        # for k, v in sample_pairs.items():
        for ts in to_show:
            k, v = ts
            count += 1
            if count > 100: break
            out += "<table>"
            out += "<tr> <th> #N diffusion step </th> <th> Input </th> <th> Image </th> <th> Input </th> </tr>"
            pair_id = k
            src = v['src']
            dst = v['dst']
            show_frames = v['frames'] if 'frames' in v else None
            
            if args.res == 128:
                shadow_area_pth = '/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/vis/'
                out += f"[#{k}] {src}=>{dst} : <img src=/files/{data_path}/{src.replace('jpg', 'png')}>, {dst} : <img src=/files/{data_path}/{dst.replace('jpg', 'png')}>" + ", Shadow area = " + f"<img height=\"128\" src=/files/{shadow_area_pth}/{args.set_}/{src.replace('jpg', 'png')}>" + "<br>" + "<br>"
            else:
                shadow_area_pth = '/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/vis/'
                out += f"[#{k}] {src}=>{dst} : <img src=/files/{data_path}/{src}>, {dst} : <img src=/files/{data_path}/{dst}>" + ", Shadow area = " + f"<img height=\"256\" src=/files/{shadow_area_pth}/{args.set_}/{src.replace('jpg', 'png')}>" + "<br>" + "<br>"
            # Model 
            light_path = f'/data/mint/DPM_Dataset/Dataset_For_Baseline/{args.dataset_name}/{args.set_}/{pair_id}_src={src}_dst={dst}/n_step={n_frame}/'
            light_path_transp = f'/data/mint/DPM_Dataset/Dataset_For_Baseline/for_vis/{args.dataset_name}_vis_ball/{pair_id}_src={src}_dst={dst}/n_step={n_frame}/'

            out += "<tr>"
            
            out += f"<td> <img src=/files/{data_path}/{src}> </td>"
            
            ###################################################
            # Show results
            if show_map_centered == "True":
                frames = glob.glob(f"{light_path}/map_centered/m_*.png")
            elif show_map_clean == "True":
                frames = glob.glob(f"{light_path}/map_clean/m_*.png")
            elif show_map_ball_transp == "True":
                frames = glob.glob(f"{light_path_transp}/ball/m_*.png")
            elif show_map_ball == "True":
                frames = glob.glob(f"{light_path}/ball/m_*.png")
            else:
                frames = []

            vid_file = 'map_centered.mp4' if show_map_centered == "True" else 'map_clean.mp4'
            if os.path.exists(f"{light_path}/{vid_file}") and show_vid == "True":
                out += f"""
                    <td>  
                    <video controls autoplay muted loop>
                        <source src=/files/{light_path}/{vid_file} type="video/mp4">
                    </video>
                    </td>
                """
            else: 
                out += "<td> <p style=\"color:red\">Video not found!</p> </td>"
            out += f"<td>"
            frame_id = []
            if len(frames) > 1:
                if ds > 0:
                    tmp_ds = [0] + list(range(1, len(frames)-1, int(len(frames)/ds))) + [len(frames)-1]
                else:
                    tmp_ds = list(range(len(frames)))

                if show_frames is not None:
                    tmp_ds = [i for i in range(len(frames)) if i in show_frames]
                if show_all_frames == "True":
                    tmp_ds = list(range(len(frames)))

                frames = sort_by_frame(frames)
                for idx, f in enumerate(frames):
                    if idx not in tmp_ds: continue
                    out += "<img src=/files/" + f + ">"
                    frame_id.append(f.split('/')[-1])
                
                # Write all frame id within oneline right below the images, each text away by space of 128px (since images are 256px)
                out += "<br>"
                for f in frame_id:
                    out += f'<span style="display: inline-block; width:256px;">{f}</span>'
                out += "<br>"
            else:
                out += "<p style=\"color:red\">Images not found!</p>"
            out += "</td>"
            ###################################################
            
            out += "</tr>"
                
            out += "</table>"
            out += "<br> <hr>"
                    
        return out

    return app

if __name__ == "__main__":
    
    # f"/data/mint/DPM_Dataset/MultiPIE_testset/mp_aligned/{args.set_}/"
    data_path = args.dataset_path
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=True)
