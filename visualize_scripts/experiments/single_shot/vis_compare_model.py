from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/sample_utils/')
import mani_utils, file_utils

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
        out = f"<h1> Comparison: {args.comparison_candidate} </h1>"
        out += f"<a href=\"/model_compare/sampling=reverse&show_itmd=True&show_recon=True&show_relit=True\"> Model Comparison </a> <br>"
        
        return out
        
    @app.route("/model_compare/sampling=<sampling>&show_itmd=<show_itmd>&show_recon=<show_recon>&show_relit=<show_relit>")
    def model_compare(sampling, show_itmd, show_recon, show_relit):
        # Fixed the training step and varying the diffusion step
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
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
        out += "<button onclick='transposeAllTables()'>Transpose</button>"
        data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_{args.res}/{args.set_}/"
        assert os.path.isfile(args.sample_pair_json)
        f = open(args.sample_pair_json)
        sample_pairs = json.load(f)['pair']
        
        # path example : /data/mint/sampling/paired_training_target/log=paired+allunet_eps+ddst_128_cfg=paired+allunet_eps+ddst_128.yaml
        # /model_040000/valid/render_face/reverse_sampling/src=62385.jpg/dst=61432.jpg
        f = open(args.comparison_candidate)
        candidates = json.load(f)
        print(candidates)
        
        for k, v in sample_pairs.items():
            out += "<table>"
            out += "<tr> <th> #N diffusion step </th> <th> Input </th> <th> Image </th> <th> Input </th> </tr>"
            src = v['src']
            dst = v['dst']
            
            if args.res == 128:
                out += f"[#{k}] {src}=>{dst} : <img src=/files/{data_path}/{src.replace('jpg', 'png')}>, {dst} : <img src=/files/{data_path}/{dst.replace('jpg', 'png')}>" + "<br>" + "<br>"
            else:
                out += f"[#{k}] {src}=>{dst} : <img src=/files/{data_path}/{src}>, {dst} : <img src=/files/{data_path}/{dst}>" + "<br>" + "<br>"
            # Model 
            for m_name, metadat in candidates.items():
                # Model's metadata
                ckpt = metadat['step']
                alias = metadat['alias']
                itp = metadat['itp']
                itp_method = metadat['itp_method']
                diff_step = metadat['diff_step']
                time_respace = metadat['time_respace']
                n_frame= metadat['n_frame']
                
                path = f"{args.sampling_dir}/{args.exp_dir}/{m_name}/{ckpt}/{args.set_}/{itp}/{sampling}_sampling/src={src}/dst={dst}/"
            
                out += "<tr>"
                alias_str = alias.split('_')
                # alias example: dpmsolver++_3_singlestep_15_dynamic_thresholding
                out += f"<td> {alias} <br> {ckpt} </td> "
                
                if args.res == 128:
                    out += f"<td> <img src=/files/{data_path}/{src.replace('jpg', 'png')}> </td>"
                else:
                    out += f"<td> <img src=/files/{data_path}/{src}> </td>"
                
                ###################################################
                # Show results
                # if 'baseline' in alias:
                if 'SP' in alias:
                    frames = glob.glob(f"{path}/{itp_method}_diff={diff_step}_respace={time_respace}/n_frames={n_frame}/res_*.png")
                    # out += f"{path}/{itp_method}_{diff_step}/n_frames={n_frame}/res_*.png"
                else:
                    frames = glob.glob(f"{path}/{itp_method}_{diff_step}/n_frames={n_frame}/res_*.png")
                    # out += f"{path}/{itp_method}_{diff_step}/n_frames={n_frame}/res_*.png"
                # out += str(show_itmd)
                print(frames)
                out += "<td>"
                if len(frames) > 0:
                    frames = sort_by_frame(frames)
                    if show_itmd == "False":
                        frames = [frames[0], frames[-1]]
                    if show_recon == "False":
                        frames = frames[1:]
                    if show_relit == "False":
                        frames = frames[:-1]
                    for f in frames:
                        if 'baseline' in alias:
                            out += "<img width=\"128\" height=\"128\" src=/files/" + f + ">"
                        else:
                            out += "<img src=/files/" + f + ">"
                else:
                    out += "<p style=\"color:red\">Images not found!</p>"
                out += "</td>"
                ###################################################
                
                if args.res == 128:
                    out += f"<td> <img src=/files/{data_path}/{src.replace('jpg', 'png')}> </td>"
                else:
                    out += f"<td> <img src=/files/{data_path}/{src}> </td>"
                
                out += "</tr>"
                
            out += "</table>"
            out += "<br> <hr>"
                    
        return out


    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_dir', default='/data/mint/sampling')
    parser.add_argument('--exp_dir', default='')
    parser.add_argument('--sample_pair_json', required=True)
    parser.add_argument('--comparison_candidate', required=True)
    parser.add_argument('--set_', default='valid')
    parser.add_argument('--res', default=128)
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()  
    
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)