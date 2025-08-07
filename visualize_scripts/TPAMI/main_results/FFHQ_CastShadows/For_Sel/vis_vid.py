from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)
parser.add_argument('--port', required=True)
parser.add_argument('--idx_file', default=None)
parser.add_argument('--sort_by_c', default=False, action='store_true')
args = parser.parse_args()

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('./', path)
    
    @app.route('/')
    def root():
        # Query string
        s = request.args.get('s', 0)
        e = request.args.get('e', 9999)

        
        # get a list of comma separated id from the url
        idx_str = request.args.get('idx', None)

        out = """
        <style>
            tr { display: block; float: left; }
            th, td { display: block; }
        </style>
        """
        out += "<table>"
        if idx_str:
            idx_to_show = []
            for id in idx_str.split(","):
                idx_to_show += glob.glob(f'./{args.path}/pair{id}_*.mp4')
        elif args.idx_file:
            # Read .txt file containing the list of indices to show
            idx = []
            with open(args.idx_file, 'r') as f:
                for line in f:
                    idx.append(line.strip())
            # Get path from glob.glob(f'./{args.path}/*.mp4') since full fn is pair{id}_src={src}_dst={dst}.mp4
            idx_to_show = []
            for id in idx:
                idx_to_show += glob.glob(f'./{args.path}/pair{id}_*.mp4')
        elif args.sort_by_c:
            idx_to_show = []
            with open("/home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/sample_json/DiFaReli++/top50perc_shadow_for_rotate.json", 'r') as f:
                data = json.load(f)['pair']
            for k, v in data.items():
                idx_to_show.append(f'./{args.path}/{k}_src={v["src"]}_dst={v["dst"]}.mp4')
        else:
            idx_to_show = glob.glob(f'./{args.path}/*.mp4')
            
        for vid in idx_to_show[int(s):int(e)]:
            out += "<tr>"
            out += "<td>"
            src = vid.split('/')[-1].split('_')[1]
            src = src.replace('src=', '')
            p_src = f'./Out/inp/{src}'
            out += vid.split('/')[-1].split('_')[0]
            out += f"<img src=\"/files/{p_src}\" width=256px> </img>"
            out += f"""
                <video width=\"256\" height=\"256\" autoplay muted controls loop> 
                    <source src=\"/files/{vid}\" type=\"video/mp4\">
                    Your browser does not support the video tag.
                    </video>
            """ 
            out += f"""
                <video width=\"256\" height=\"256\" autoplay muted controls loop> 
                    <source src=\"/files/{vid.replace('srcC', 'maxC')}\" type=\"video/mp4\">
                    Your browser does not support the video tag.
                    </video>
            """ 
            out += "<td>"
            out += "</tr>"
        out += "</table>"
        return out
    return app
        
if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=args.port, debug=True, threaded=False)