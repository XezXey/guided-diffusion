# app.py
from flask import Flask, abort, send_file, render_template_string, url_for
from pathlib import Path
from urllib.parse import quote, unquote

app = Flask(__name__)

# >>> Adjust if needed
BASE_DIR = Path("./rotate_hdr_axis=1_figure").resolve()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

def safe_join_base(relpath: str) -> Path:
    relpath = relpath.strip().lstrip("/")
    full = (BASE_DIR / relpath).resolve()
    if not str(full).startswith(str(BASE_DIR)):
        abort(404)
    return full

@app.route("/")
def index():
    subdirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    tmpl = """
    <!doctype html>
    <html>
    <head><meta charset="utf-8"><title>HDR Map Browser</title></head>
    <body>
      <h1>Select a map folder</h1>
      <ul>
      {% for d in subdirs %}
        <li><a href="{{ url_for('view_map', relpath=d['rel']) }}">{{ d['name'] }}</a> ({{ d['count'] }} images)</li>
      {% endfor %}
      </ul>
    </body>
    </html>
    """
    items = []
    for d in subdirs:
        count = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        items.append({"name": d.name, "rel": quote(d.relative_to(BASE_DIR).as_posix()), "count": count})
    return render_template_string(tmpl, subdirs=items)

@app.route("/map/<path:relpath>")
def view_map(relpath):
    folder = safe_join_base(unquote(relpath))
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    tmpl = """
    <!doctype html>
    <html>
    <head><meta charset="utf-8"><title>{{ folder_name }}</title></head>
    <body>
      <a href="{{ url_for('index') }}">← Back</a>
      <h1>{{ folder_name }}</h1>
      <ul>
      {% for f in files %}
        <li>
          <img src="{{ url_for('raw_file', relpath=f['rel']) }}" style="max-height:768px;">
          {{ f['name'] }}
        </li>
      {% endfor %}
      </ul>
    </body>
    </html>
    """
    items = [{"name": f.name, "rel": quote(f.relative_to(BASE_DIR).as_posix())} for f in files]
    return render_template_string(tmpl, folder_name=folder.name, files=items)

@app.route("/raw/<path:relpath>")
def raw_file(relpath):
    target = safe_join_base(unquote(relpath))
    if not target.exists() or not target.is_file():
        abort(404)
    return send_file(target)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--base", default=str(BASE_DIR), help="Base directory to browse")
    args = parser.parse_args()

    BASE_DIR = Path(args.base).resolve()  # ✅ just reassign, no global needed
    app.run(host=args.host, port=args.port, debug=False)
