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
parser.add_argument('--sample_pair_json', default='/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/default/sample_json/sj_in_paper.json')
parser.add_argument('--res', default='256')
args = parser.parse_args()

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)

    @app.route("/")
    def root():
        # Fixed the training step and varying the diffusion step
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        
        sample_json = str(request.args.get('sample_json', args.sample_pair_json))
        s = request.args.get('s', 0)
        e = request.args.get('e', 100)
        
        data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_{args.res}/{args.set_}/"
        try:
            os.path.isfile(sample_json)
            f = open(sample_json)
            sample_pairs = json.load(f)['pair']
            print(f"[#] Found {sample_json} file!!!")
        except:
            raise ValueError(f"Sample json file not found: {sample_json}")
        
        out += f"<h2> Sample json file: {sample_json} </h2>"
        
        to_show = list(sample_pairs.items())[int(s):int(e)]
        # for k, v in sample_pairs.items():
        for ts in to_show:
            k, v = ts
            out += "<table>"
            src = v['src']
            dst = v['dst']
            out += f"[#{k}] {src}=>{dst} : <img src=/files/{data_path}/{src}>, {dst} : <img src=/files/{data_path}/{dst}>" + "<br>" + "<br>"
            out += "</tr>"
                
            out += "</table>"
            out += "<br> <hr>"
                    
        return out
    
    return app
    
if __name__ == "__main__":
     
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)