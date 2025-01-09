from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)
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
        s = request.args.get('s')
        e = request.args.get('e')
        out = """
        <style>
            tr { display: block; float: left; }
            th, td { display: block; }
        </style>
        """
        out += "<table>"
        
        # out += "<tr>"
        # # for f in folders:
        # #     # out += f"<tr> {f} </tr>"
        # #     out += f"<p style=\"display: inline; margin:64px;\">{f}</p>"
        # out += "</tr>"
            
        for vid in glob.glob(f'./{args.path}/*.mp4')[int(s):int(e)]:
            out += "<tr>"
            # out += f"<td> <img src=\"/files/{vid}/path.png\" width=256px </td>"
            # for vid in vids:
            out += "<td>"
            src = vid.split('/')[-1].split('_')[1]
            src = src.replace('src=', '')
            p_src = f'./Out/inp/{src}'
            out += f"<img src=\"/files/{p_src}\" width=256px> </img>"
            out += f"""
                <video width=\"256\" height=\"256\" autoplay muted controls loop> 
                    <source src=\"/files/{vid}\" type=\"video/mp4\">
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
    app.run(host='0.0.0.0', port='1911', debug=True, threaded=False)
