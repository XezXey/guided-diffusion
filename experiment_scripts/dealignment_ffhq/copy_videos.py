import shutil
import os
import glob

out_dir = './good_videos'
in_dir = './videos'
os.makedirs(out_dir, exist_ok=True)
# for f in ['loki_6', 'loki_7', 'joker_2', 'joker_3', 'matrix_1', 'tommy_1', 'tommy_1', 'tony_1']:
for f in ['tony_1']:
    dir = glob.glob(f'{in_dir}/{f}/*')
    for d in dir:
        light_name = d.split('/')[-1]
        d = d + '/compare_rmv_border/'
        os.makedirs(f'{out_dir}/{f}/{light_name}', exist_ok=True)
        for frame in glob.glob(f'{d}/*.png'):
            # print(frame)
            shutil.copy(src=frame, dst=f'{out_dir}/{f}/{light_name}')