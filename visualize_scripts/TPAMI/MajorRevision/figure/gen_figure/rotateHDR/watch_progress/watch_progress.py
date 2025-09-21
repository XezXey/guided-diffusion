#!/usr/bin/env python3
import os, glob, json, time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rotate_axis', type=int, required=True, help='rotate sh axis')
parser.add_argument('--sample_json', type=str, nargs='+', required=True, help='sample json file')
parser.add_argument('--hdr_list', type=str, nargs='+', required=True, help='hdr list to check')
parser.add_argument('--c_list', type=str, nargs='+', default=['1.0', '0.9', '0.8', '0.7', '0.6', '0.5'], help='c list to check')
parser.add_argument('--sd_list', type=int, nargs='+', default=[75, 50, 25], help='shadow diff list to check')
parser.add_argument('--out_html', type=str, default='progress.html', help='output html file')
args = parser.parse_args()

# ==== CONFIG ====
INTERVAL_SEC = 60                     # scan interval
OUT_HTML = args.out_html               # output dashboard
BAR_LEN = 30
C_LIST = args.c_list
SD_LIST = args.sd_list
HDR_LIST = [os.path.basename(hdr).split('.')[0] for hdr in args.hdr_list]
SAMPLE_JSONS = args.sample_json
# ==============

def progress_bar(n, total, length=BAR_LEN):
    total = max(1, total)
    ratio = n / total
    filled = int(length * ratio)
    return "█" * filled + "-" * (length - filled), ratio * 100

def latest_file_mtime(path_glob):
    files = glob.glob(path_glob)
    if not files:
        return None
    latest = max(os.path.getmtime(f) for f in files)
    return datetime.fromtimestamp(latest)

def count_status(samples, hdr_map, c, sd):
    total = len(samples)
    done, latest = 0, None
    for p in samples.values():
        src, dst = p['src'], p['dst']
        path = (
            f"/data/mint/TPAMI_MajorRevision/Ours/ffhq_hdr_finale/"
            f"log=paired+difareli+cs+nodpm+trainset_256_cfg=paired+difareli+cs+nodpm+trainset_256.yaml"
            f"_SD{sd}_{c}C_sColor_Lmax10_rAxis{args.rotate_axis}/ema_300000/valid/render_face_hdr/"
            f"{hdr_map}/src={src}/dst={dst}/Lerp_1000/n_frames=60/"
        )
        if os.path.isdir(path) and len(glob.glob(f"{path}/res_frame*.png")) == 60:
            done += 1
            mtime = latest_file_mtime(f"{path}/res_frame*.png")
            if mtime and (latest is None or mtime > latest):
                latest = mtime
    return done, total, latest  # latest can be None

def scan_one_json(sample_json):
    with open(sample_json, "r") as f:
        samples = json.load(f)["pair"]
    rows = []
    now = datetime.now()
    for hdr_map in HDR_LIST:
        for c in C_LIST:
            for sd in SD_LIST:
                done, total, last_dt = count_status(samples, hdr_map, c, sd)
                bar, pct = progress_bar(done, total)
                rows.append({
                    "json": os.path.basename(sample_json),
                    "hdr": hdr_map,
                    "c": c,
                    "sd": sd,
                    "done": done,
                    "total": total,
                    "pct": pct,
                    "bar": bar,
                    "last_file": last_dt.isoformat(sep=" ", timespec="seconds") if last_dt else "—",
                    "scanned": now.isoformat(sep=" ", timespec="seconds"),
                })
    return rows

def render_html(all_rows, changed_keys, prev_values, interval):
    head = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{interval}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relighting Progress on rotate axis = {args.rotate_axis} </title>
<style>
  :root {{
    --bg: #0b0f14;
    --card: #111827;
    --text: #e5e7eb;
    --muted: #9ca3af;
    --changed-bg: #2b0f0f;
    --changed-text: #f87171;
    --bar-bg: #374151;
    --bar-fill: #34d399;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, Ubuntu, Cantarell, Helvetica Neue, Arial; margin: 0; padding: 24px; }}
  .wrap {{ max-width: 1280px; margin: 0 auto; }}
  h1 {{ margin: 0 0 8px 0; font-weight: 700; }}
  .sub {{ color: var(--muted); margin-bottom: 16px; }}
  .card {{ background: var(--card); border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 8px 10px; text-align: left; font-size: 14px; }}
  th {{ color: var(--muted); font-weight: 600; border-bottom: 1px solid #1f2937; }}
  tr + tr td {{ border-top: 1px dashed #1f2937; }}
  .mono {{ font-family: var(--mono); }}
  .pct {{ white-space: nowrap; }}
  .bar {{ position: relative; height: 10px; border-radius: 999px; background: var(--bar-bg); overflow: hidden; min-width: 220px; }}
  .bar > span {{ position: absolute; left: 0; top: 0; bottom: 0; background: var(--bar-fill); }}
  tr.updated td {{ background: var(--changed-bg); color: var(--changed-text); transition: background 0.5s ease, color 0.5s ease; animation: pulse 1.2s ease-in-out 0s 1; }}
  @keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(248,113,113,0.7); }} 100% {{ box-shadow: 0 0 0 14px rgba(248,113,113,0); }} }}
  .footer {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
  .small {{ font-size: 12px; color: var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Relighting Progress on rotate axis = {args.rotate_axis} </h1>
  <div class="sub">Auto-refresh every {interval}s. Red rows changed since last scan. When highlighted, the “Done” and “Latest File” cells show <span class="mono">prev → curr</span>.</div>
"""
    body = []
    groups = defaultdict(list)
    for r in all_rows:
        groups[r["json"]].append(r)

    now_str = datetime.now().isoformat(sep=" ", timespec="seconds")

    for json_name, rows in groups.items():
        # rows = sorted(rows, key=lambda x: (x["hdr"], -x["c"], x["sd"]))
        rows = sorted(rows, key=lambda x: (x["hdr"], x["c"], x["sd"]))
        body.append(f'<div class="card"><div class="small">JSON: <span class="mono">{json_name}</span></div>')
        body.append("""
<table>
  <thead>
    <tr>
      <th>HDR</th>
      <th>c</th>
      <th>sd</th>
      <th>Progress</th>
      <th class="pct">Done</th>
      <th class="pct">Percent</th>
      <th>Latest File</th>
      <th>Scanned</th>
    </tr>
  </thead>
  <tbody>
""")
        for r in rows:
            key = (json_name, r["hdr"], r["c"], r["sd"])
            cls = "updated" if key in changed_keys else ""
            width = f"{r['pct']:.6f}%"
            # build "Done" cell with prev->curr if changed
            if key in changed_keys and key in prev_values and prev_values[key] is not None:
                prev_done, prev_last = prev_values[key]
                done_txt = f"{prev_done}→{r['done']} / {r['total']}"
                last_file_txt = f"{prev_last} → {r['last_file']}" if (prev_last and r["last_file"] and prev_last != r["last_file"]) else r["last_file"]
            else:
                done_txt = f"{r['done']}/{r['total']}"
                last_file_txt = r["last_file"]

            bar_html = f'<div class="bar"><span style="width:{width}"></span></div>'
            pct_txt = f"{r['pct']:.1f}%"
            body.append(
                f'<tr class="{cls}">'
                f'<td class="mono">{r["hdr"]}</td>'
                f'<td class="mono">{r["c"]}</td>'
                f'<td class="mono">{r["sd"]}</td>'
                f'<td>{bar_html}</td>'
                f'<td class="mono pct">{done_txt}</td>'
                f'<td class="mono pct">{pct_txt}</td>'
                f'<td class="mono">{last_file_txt}</td>'
                f'<td class="mono">{r["scanned"]}</td>'
                f'</tr>'
            )
        body.append("</tbody></table></div>")

    foot = f"""
  <div class="footer">Generated at {now_str}</div>
</div>
</body>
</html>
"""
    return head + "\n".join(body) + foot

def monitor():
    # prev_map keeps the *current known* (done, last_file) per key
    prev_map = {}
    while True:
        all_rows = []
        changed_keys = set()
        # prev_values records the previous (done, last_file) for rows that changed this scan
        prev_values = {}

        for sj in SAMPLE_JSONS:
            try:
                rows = scan_one_json(sj)
            except Exception as e:
                rows = [{
                    "json": os.path.basename(sj),
                    "hdr": "—", "c": "—", "sd": "—",
                    "done": 0, "total": 1, "pct": 0.0, "bar": "-"*BAR_LEN,
                    "last_file": f"ERROR: {e}", "scanned": datetime.now().isoformat(sep=" ", timespec="seconds")
                }]
            all_rows.extend(rows)

        # detect changes and capture previous values
        for r in all_rows:
            key = (r["json"], r["hdr"], r["c"], r["sd"])
            curr = (r["done"], r["last_file"])
            if key not in prev_map:
                # first time seen -> mark changed with no prev (shows as current only)
                changed_keys.add(key)
                prev_values[key] = None
                prev_map[key] = curr
            elif prev_map[key] != curr:
                changed_keys.add(key)
                prev_values[key] = prev_map[key]  # (prev_done, prev_last_file)
                prev_map[key] = curr

        # write HTML atomically
        html = render_html(all_rows, changed_keys, prev_values, INTERVAL_SEC)
        tmp_path = OUT_HTML + ".tmp"
        Path(tmp_path).write_text(html, encoding="utf-8")
        os.replace(tmp_path, OUT_HTML)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] wrote {OUT_HTML} "
              f"({len(all_rows)} rows, {len(changed_keys)} changed)")

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    monitor()
