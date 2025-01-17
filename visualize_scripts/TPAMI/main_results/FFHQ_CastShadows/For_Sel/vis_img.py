from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)
parser.add_argument('--port', required=True)
args = parser.parse_args()

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        # Query string

        
        # get a list of comma separated id from the url
        idx_str = request.args.get('idx', None)

        out = """
        <style>
            tr { display: block; float: left; }
            th, td { display: block; }
        </style>
        """
        out += "<table>"
            
        for img in sorted(glob.glob(f'{args.path}/*res.png')):
            out += "<tr>"
            out += "<td>"
            out += f"<img src=\"/files/{img}\" width=256px> </img>"
            out += "<td>"
            out += "</tr>"
        out += "</table>"
        return out
    return app
        
if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=args.port, debug=True, threaded=False)
