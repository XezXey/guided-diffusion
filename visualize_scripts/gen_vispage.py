from flask import Flask, request, send_file
import glob, os
print(__name__)
app = Flask(__name__)

@app.route('/files/<path:path>')
def servefile(path):
  return send_file(path)

@app.route('/')
def root():
  out = ""
  for d in glob.glob("src=0/*"):
    if not os.path.isdir(d): continue
    out += d + "<br>"
    for f in glob.glob(d + "/*.png")[::2]:
      out += "<img src=files/" + f + ">"
    out += "<br>"
  return out
