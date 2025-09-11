from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
import argparse
import pandas as pd
    
parser = argparse.ArgumentParser()
parser.add_argument('--port', required=True)
parser.add_argument('--host', default='0.0.0.0')

args = parser.parse_args()

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('./', path)
    
    @app.route('/')
    def root():
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
        for c in ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]:
            out += "<h1> C = " + c + "</h1>"
        
            out += "<tr><td>"
            for p in [f"./ACgrid_{c}C_res.mp4", f"./ACgrid_{c}C_ren.mp4"]:
                if os.path.exists(p):
                    out += f"""
                        <video controls autoplay muted loop>
                            <source src=/files/{p} type="video/mp4">
                        </video>
                    """
                else: 
                    out += "<td> <p style=\"color:red\">Video not found!</p> </td>"
            out += "</td></tr>"
                
            out += "</table>"
            out += "<br> <hr>"
                    
        return out

    return app

if __name__ == "__main__":
    
    data_path = "/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/valid/" 
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=True)
