from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
import argparse
parser = argparse.ArgumentParser(description='Flask server for video visualization')
parser.add_argument('--port', type=int, default=5555, help='Port number')
parser.add_argument('--dir', type=str, default='.', help='Directory to serve')
args = parser.parse_args()


def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('./', path)
    
    @app.route('/')
    def root():
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
            
        # for vid in glob.glob(f'./*.mp4'):
        for vid in glob.glob(f'{args.dir}/*.mp4'):
            out += "<tr>"
            # out += f"<td> <img src=\"/files/{vid}/path.png\" width=256px </td>"
            # for vid in vids:
            out += "<td>"
            print(vid)
            src = vid.split('/')[-1].split('_')[0] + '.jpg'
            out += f"<img src=\"/files/{src}\" width=256px> </img>"
            out += f"""
                <video width=\"768\" height=\"256\" autoplay muted controls loop> 
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
    app.run(host='0.0.0.0', port=args.port, debug=True, threaded=False)