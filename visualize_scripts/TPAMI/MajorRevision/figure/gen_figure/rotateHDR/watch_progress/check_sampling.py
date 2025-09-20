import os, glob, json, time, sys
from datetime import datetime
from guided_diffusion.mint_logger import createLogger
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rotate_axis', type=int, default=2, help='rotate sh axis')
parser.add_argument('--sample_json', type=str, nargs='+', required=True, help='sample json file')
args = parser.parse_args()

c_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
sd_list = [75, 50, 25]
hdr_list = ['064_hdrmaps_com_free_2K', '125_hdrmaps_com_free_2K', '117_hdrmaps_com_free_2K']

def progress_bar(n, total, length=30):
    total = max(1, total)
    ratio = n / total
    filled = int(length * ratio)
    return "█" * filled + "-" * (length - filled), ratio * 100

def latest_file_mtime(path_glob):
    files = glob.glob(path_glob)
    if not files:
        return None
    latest = max(os.path.getmtime(f) for f in files)
    return datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S")

def count_status(samples, hdr_map, c, sd):
    total = len(samples)
    done, latest = 0, None
    for p in samples.values():
        src, dst = p['src'], p['dst']
        path = (
            f"/data/mint/TPAMI_MajorRevision/Ours/ffhq_hdr_finale/"
            f"log=paired+difareli+cs+nodpm+trainset_256_cfg=paired+difareli+cs+nodpm+trainset_256.yaml"
            f"_SD{sd}_{c}C_sColor_Lmax10_rAxis1/ema_300000/valid/render_face_hdr/"
            f"{hdr_map}/src={src}/dst={dst}/Lerp_1000/n_frames=60/"
        )
        if os.path.isdir(path) and len(glob.glob(f"{path}/res_frame*.png")) == 60:
            done += 1
            mtime = latest_file_mtime(f"{path}/res_frame*.png")
            if mtime and (latest is None or mtime > latest):
                latest = mtime
    return done, total, latest or "—"

def scan(sample_json):
    with open(sample_json, "r") as f:
        samples = json.load(f)["pair"]

    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for hdr_map in hdr_list:
        for c in c_list:
            for sd in sd_list:
                done, total, last_file = count_status(samples, hdr_map, c, sd)
                bar, pct = progress_bar(done, total)
                lines.append(
                    f"HDR={hdr_map}, c={c}, sd={sd} |{bar}| {done}/{total} ({pct:.1f}%) "
                    f"| last file: {last_file} | scanned: {now}"
                )
    return lines

def monitor(sample_jsons, interval=60):
    try:
        while True:
            # clear screen (ANSI)
            sys.stdout.write("\033[H\033[J")
            sys.stdout.flush()

            for sj in sample_jsons:
                print(f"=== {os.path.basename(sj)} ===")
                for line in scan(sj):
                    print(line)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    monitor(args.sample_json, interval=60)

