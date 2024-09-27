import numpy as np
import glob, tqdm
# path = '/data/mint/DPM_Dataset/Generated_Dataset_TargetLight/Generated_Dataset_cast_shadows/'
path = '/data/mint/DPM_Dataset/Generated_Dataset_TargetLight/Generated_Dataset_hardlink/'
set_ = 'train'


# sub_f = ['/shadow_diff_SS_with_c_simplified_npy/', '/rendered_images/deca_masked_face_images_woclip/']
# tgt_shape = [(256, 256, 1), (256, 256, 3)]
sub_f = ['/rendered_images/deca_masked_face_images_woclip/']
tgt_shape = [(256, 256, 3)]

for f in sub_f:
    count = 0
    files = glob.glob(f'{path}/{f}/{set_}/*.npy')
    for file in tqdm.tqdm(files):
        data = np.load(file)
        assert data.shape == tgt_shape[sub_f.index(f)]
        count += 1
    print("[#] Total files with correct shapes: ", count)