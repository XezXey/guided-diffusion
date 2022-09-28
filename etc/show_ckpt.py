from pyexpat import model
from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import pandas as pd
import json
import time
import sys
import argparse

def list_ckpt(path):
    name = glob.glob(f"{path}/*_*")
    model_ckpt = []
    model_name = []
    for n in name:
        n = n.split('/')[-1]
        model_ckpt.append(n)
        if n.split('_')[0] not in model_name:
            model_name.append(n.split('_')[0])
    
    
    model_dict = dict([(k, {}) for k in model_name])
    for m in model_ckpt:
        name = m.split('_')[0]
        if 'ema' in m:
            rate = ''.join(m.split('_')[1:3])
            step = m.split('_')[-1].split('.')[0]
        elif 'model' in m:
            rate = 'model'
            step = m.split('_')[-1].split('.')[0].replace(rate, '')
            
        else: raise ValueError("[#] Ckpt is unknown.")
        
        # Update dict
        if rate not in model_dict[name].keys():
            model_dict[name][rate] = [step]
        else:
            model_dict[name][rate].append(step)
            
    for name in model_dict.keys():
        for rate in model_dict[name].keys():
            model_dict[name][rate] = sorted(model_dict[name][rate])
    
        
    # print("#"*50)
    # print(path)
    # print(model_dict)
    return model_dict
    

def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def root():
        out = ""
        out += "<h1> Available Checkpoint </h1>"
        machine = sorted([name for name in os.listdir(args.model_path) if os.path.isdir(os.path.join(args.model_path, name))])
        for m in machine:
            m_path = f"{args.model_path}/{m}/"
            out += f"<h2> {m} </h2>"
            # List the contents 
            if len(os.listdir(m_path)) == 0:
                out += "<h3 style=\"color:red\"> [#] This directory is not mounted! </h3>"
            else:
                # List the model & its checkpoint sorted by date
                files = os.listdir(m_path)
                files = [f"{m_path}/{f}" for f in files]
                files.sort(key=lambda x: os.path.getmtime(x))
                files.reverse()
                
                for model_path in files:
                    if ((time.time()-os.path.getmtime(model_path)) / 604800) > args.week_period:
                        continue
                        
                    model_name = model_path.split('/')[-1]
                    if ('test' in model_name) or ('dev' in model_name) or ('tmp' in model_name): continue
                    # out += f"{model_name}"
                    
                    out += "<table border=\"1\">"
                    ckpt_dict = list_ckpt(model_path)
                    out += f"<tr> <th style=\"text-align:center\"> {model_name} </th> </tr>"
                    out += "</table>"
                    
                    out += "<table border=\"1\">"
                    if len(ckpt_dict) == 0:
                        out += f"<tr><td style=\"color:red; text-align:center\"> Checkpoint is not found! </td></tr>"
                    else:
                        # Architecture name
                        df = pd.DataFrame.from_dict(ckpt_dict).transpose()
                        html = df.style.set_table_styles([
                            {"selector": "td, th", "props": [("border", "1px solid grey !important")]},
                        ])
                        out += html.render()
                        
                    out += "</table>"
                    out += "<br>"
    
            
            out += "<hr>"
        return out
    
    return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--week_period', type=int, default=2)
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)