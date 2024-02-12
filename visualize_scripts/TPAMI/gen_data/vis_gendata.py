import numpy as np
from flask import Flask, request, send_file, send_from_directory
import glob

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        #NOTE: Root path
        out = ""
        out += f"<a href=\"/gen_data/\"> Generated data on {args.data_path} </a> <br>"
        return out

    @app.route('/gen_data/', methods=['GET'])
    def gen_data():
        args_page = request.args
        s = int(args_page.get('s'))
        e = int(args_page.get('e'))
        ffhq_path = '/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/train/'

        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        
        
        out += "<table>"
        
        img_list = sorted(glob.glob(f"{args.data_path}/*.png"))[s:e]
        nl = True
        for i, img in enumerate(img_list):
            if nl: 
                out += "<tr>"
                out += "<td style=\"text-align: left;\">"
                nl = False
            out += f"<img src=/files/{img} title=\"{img}\">"
            if 'relit' in img:
                light = img.split('/')[-1].split('_')[1]
                light = ffhq_path + light + '.jpg'
                out += f"<img src=/files/{light} title=\"{light}\">"
            if 'input' in img:
                out += "</tr>"
                out += "</td>"
                nl = True
                
            # if (i % 2) == 0:
            #     out += "</tr>"
        out += "</table>"
        out += "<br> <hr>"
        return out
    
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)
