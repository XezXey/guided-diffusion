import numpy as np
import glob, os, json

n_frames = 2
path = '/data/mint/dataset_generation/random_target/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml_step=250/ema_085000/train/render_face/reverse_sampling/'
src_dst_done = [p.split('/')[-2:] for p in glob.glob(f'{path}/*/*')]
# print(src_dst_done[:10])


def find_progress(start, end, src_dst_done):
    with open('../sample_scripts/dataset_generation/sample_json/generated_dataset_seed=47.json', 'r') as f:
        json_data = json.load(f)['pair']
        src_dst_fromfile = []
        for k in list(json_data.keys())[start:end]:
            src_dst_fromfile.append([f"src={json_data[k]['src']}", f"dst={json_data[k]['dst']}"])
    # print(src_dst_fromfile[:10])
        
    src_dst_fromfile_orig = src_dst_fromfile.copy()
    for done in src_dst_done:
        try:
            idx = src_dst_fromfile.index(done)
            src_dst_fromfile[idx] = None
        except:
            pass
    # print(src_dst_fromfile)
    progress = np.where(np.array(src_dst_fromfile, dtype=object) == None)[0]
    print("Progress : ", progress[:10], progress[-10:])
    # print(progress[1:] - progress[:-1])
    if len(progress) == 0:
        raise Exception("No progress found")
    
    # remove if last progress is not continuous
    progress_chk = progress[1:] - progress[:-1]
    limit = 10
    if len(np.where(progress_chk != 1)[0]) < limit:
        # exclude index in progress where progress is not continuous and its length is less than limit
        progress = progress[np.where(progress_chk == 1)[0]]
        progress_chk = progress_chk[np.where(progress_chk == 1)[0]]
    
    assert np.allclose((progress_chk), 1)   # check if progress is continuous
    print("=====================================================")
    print(f"[#] For {start} to {end} frames, progress is continuous")
    print(f"[#] Progress is : {progress[-1]} which is image at {src_dst_fromfile_orig[progress[-1]]}")
    print(f"[#] You should start at {progress[-1]+1} which is image at {src_dst_fromfile_orig[progress[-1]+1]}")
    print(f"[#] After Re-indexing...")
    print(f"[#] You should run from {start+progress[-1]+1} to {end} which is {end - (start+progress[-1]+1)} images")
    print("=====================================================")

chunk = [(0, 66670), (66670, 133340), (133340, 200010)]
# chunk = [(0, 4000)]
# chunk = [(200010, 266680), (266680, 333350), (333350, 400020)]
# chunk = [(400020, 466690), (466690, 533360), (533360, 600030)]

for c in chunk:
    find_progress(c[0], c[1], src_dst_done)

