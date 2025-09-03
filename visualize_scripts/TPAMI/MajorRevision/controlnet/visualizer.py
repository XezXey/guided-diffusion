from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument('--sample_pair_json', required=True)
parser.add_argument('--comparison_json', required=True)
parser.add_argument('--set_', default='valid')
parser.add_argument('--res', default=128)
parser.add_argument('--port', required=True)
parser.add_argument('--host', default='0.0.0.0')
args = parser.parse_args()

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
    
    @app.route("/")
    def root():
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
        
        show_vid = request.args.get('show_vid', "True")
        show_img = request.args.get('show_img', "True")
        show_shadm = request.args.get('show_shadm', "False")
        show_itmd = request.args.get('show_itmd', "True")
        show_recon = request.args.get('show_recon', "True")
        show_relit = request.args.get('show_relit', "True")
        n_frame = request.args.get('n_frame', None)
        s = request.args.get('s', 0)
        e = request.args.get('e', 100)
        ds = int(request.args.get('ds', 5))
        sample_json = str(request.args.get('sample_json', args.sample_pair_json))
        model_json = str(request.args.get('model_json', args.comparison_json))
        
        data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/{args.set_}/"
        try:
            os.path.isfile(sample_json)
            f = open(sample_json)
            sample_pairs = json.load(f)['pair']
        except:
            raise ValueError(f"Sample json file not found: {sample_json}")
        
        out += f"<h2> Sample json file: {sample_json} {n_frame} </h2>"
        out += "Transpose : <button onclick='transposeAllTables()'>Transpose</button>"
        
        f = open(model_json)
        candidates = json.load(f)
        print(candidates)
        
        count = 0
        to_show = list(sample_pairs.items())[int(s):int(e)]
        # for k, v in sample_pairs.items():
        for ts in to_show:
            k, v = ts
            count += 1
            out += "<table>"
            out += "<tr> <th> Alias </th> <th> Input </th> <th> Image </th> <th> Input </th> </tr>"
            src = v['src']
            dst = v['dst']
            
            shadow_area_pth = '/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff_SS_with_c_simplified/vis/'
            out += f"[#{k}] {src}=>{dst} : <img width=\"{args.res}\" height=\"{args.res}\" src=/files/{data_path}/{src}>, {dst} : <img width=\"{args.res}\" height=\"{args.res}\" src=/files/{data_path}/{dst}>" + ", Shadow area = " + f"<img width=\"{args.res * 3}\" height=\"{args.res}\" src=/files/{shadow_area_pth}/{args.set_}/{src.replace('jpg', 'png')}>" + "<br>" + "<br>"
            # Model 
            for m_idx, metadat in candidates.items():
                # Model's metadata
                ckpt = metadat['step']
                alias = metadat['alias']
                img_path = metadat['res_dir']
                n_frames = metadat['n_frames']

                path = f"{img_path}/src={src}/dst={dst}/Lerp_1000/n_frames={n_frames}"
            
                out += "<tr>"
                alias_str = alias.split('_')
                out += f"<td> {alias_str} <br> {ckpt} </td> "
                
                out += f"<td> <img width=\"{args.res}\" height=\"{args.res}\" src=/files/{data_path}/{src}> </td>"
                
                ###################################################
                # Show results
                if show_shadm == "True":
                    frames = glob.glob(f"{path}/shadm_*.png")
                elif show_img == "True":
                    frames = glob.glob(f"{path}/res_frame*.png")
                else:
                    frames = []

                if os.path.exists(f"{path}/out_rt.mp4") and show_vid == "True":
                    out += f"""
                        <td>  
                        <video controls autoplay muted loop>
                            <source src=/files/{path}/out_rt.mp4 type="video/mp4">
                        </video>
                        </td>
                    """
                else: 
                    out += "<td> <p style=\"color:red\">Video not found!</p> </td>"
                out += f"<td>"
                if len(frames) > 1:
                    if ds > 0:
                        tmp_ds = [0] + list(range(1, len(frames)-1, int(len(frames)/ds))) + [len(frames)-1]
                    else:
                        tmp_ds = list(range(len(frames)))
                    frames = sort_by_frame(frames)
                    if show_itmd == "False":
                        frames = [frames[0], frames[-1]]
                    if show_recon == "False":
                        frames = frames[1:]
                    if show_relit == "False":
                        frames = frames[:-1]
                    for idx, f in enumerate(frames):
                        if idx not in tmp_ds: continue
                            
                        out += f"<img width=\"{args.res}\" height=\"{args.res}\" src=/files/" + f + ">"
                else:
                    out += "<p style=\"color:red\">Images not found!</p>"
                out += "</td>"
                ###################################################
                
                out += f"<td> <img width=\"{args.res}\" height=\"{args.res}\" src=/files/{data_path}/{src}> </td>"
                
                out += "</tr>"
                
            out += "</table>"
            out += "<br> <hr>"
                    
        return out

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=True)
