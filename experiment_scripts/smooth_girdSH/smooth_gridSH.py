import cv2 as cv
import numpy as np
import os
import tqdm
import glob
import matplotlib.pyplot as plt
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--cx', type=float, default=0)
parser.add_argument('--cy', type=float, default=0)
parser.add_argument('--rx', type=float, default=1)
parser.add_argument('--ry', type=float, default=1)
parser.add_argument('--resize', action='store_true', default=False)
parser.add_argument('--baseline', action='store_true', default=False)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--out_path', type=str, default='./vid_out_testpath/')
parser.add_argument('--set_', type=str, default='valid')
args = parser.parse_args()

def smooth_spiral(sj_name, n_frames, savepath):
    n = 300
    rounds = 4

    num = 7
    x_half = (num-1) / 2
    y_half = (num-1) / 2

    maxr = 1
    minr = 0.1

    # Change path here!!!
    if args.baseline:
        dat_path = f"/data/mint/Generated_Relighting_Dataset/{sj_name}/dst=60000.jpg/Lerp_1000/n_frames={n_frames}/"
        args.model = 'Masked_Face_woclip+BgNoHead+shadow_256'
    else:
        dat_path = f"/data/mint/sampling/paired_training_gridSH/log={args.model}_cfg={args.model}.yaml/model_020000/valid/render_face/reverse_sampling/{sj_name}/dst=60000.jpg/Lerp_1000/n_frames={n_frames}/"
    if not os.path.exists(dat_path):
        print(f"[#] No file...{dat_path}")
        return
    all_x = []
    all_y = []
    cx = args.cx
    rx = args.rx
    
    cy = args.cy
    ry = args.ry

    os.makedirs(f"/data/mint/smooth_rotlight_gen/{args.model}/cx{cx}_rx{rx}_cy{cy}_ry{ry}/{sj_name}/", exist_ok=True)
    os.makedirs(f"{args.out_path}/{args.model}/cx{cx}_rx{rx}_cy{cy}_ry{ry}/", exist_ok=True)
    for i in range(n):
        t = i / n 
        tt = t * rounds * 2 * np.pi
        rad = (np.cos(t * 2 * np.pi) + 1) / 2 * (maxr - minr) + minr
        
        # x = np.sin(tt) * half * rad + half
        # x = np.sin(tt) * x_half * rad * 0.4 + x_half -1
        x = np.sin(tt) * x_half * rad * rx + x_half + cx
        y = np.cos(tt) * y_half * rad * ry + y_half + cy
        
        x = np.clip(x, 0, num-1)
        y = np.clip(y, 0, num-1)
        
        all_x.append(x)
        all_y.append(y)
        
        ix = int(x)
        iy = int(y)
        ix2 = np.clip(ix + 1, 0, num-1)
        iy2 = np.clip(iy + 1, 0, num-1)
        tx = x - ix
        ty = y - iy
        
        a = cv.imread(f"{dat_path}/res_%02d_%02d.png" % (ix, iy)) / 255.0
        b = cv.imread(f"{dat_path}/res_%02d_%02d.png" % (ix2, iy)) / 255.0
        c = cv.imread(f"{dat_path}/res_%02d_%02d.png" % (ix, iy2)) / 255.0
        d = cv.imread(f"{dat_path}/res_%02d_%02d.png" % (ix2, iy2)) / 255.0
        # print(ix, iy)
        out_img = ((a*(1-ty) + c*ty)*(1-tx) + (b*(1-ty) + d*ty)*tx) * 255
        if args.resize:
            out_img = cv.resize(out_img, (64, 64))
        cv.imwrite(f"/data/mint/smooth_rotlight_gen/{args.model}/cx{cx}_rx{rx}_cy{cy}_ry{ry}/{sj_name}/%05d.png" % i, out_img)
    

    if savepath:
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        t = np.arange(len(all_x))

        # Create colormap
        cmap = plt.cm.get_cmap('cool')

        # Plot the path with a colormap based on time
        plt.scatter(all_y, all_x, c=t, cmap=cmap) # Switch x-y axes

        plt.xlim(-1, 7)
        plt.xticks(np.arange(-1, 8, 1))
        plt.ylim(-1, 7)
        plt.yticks(np.arange(-1, 8, 1))
        plt.gca().invert_yaxis()

        # Add a colorbar to show the time domain
        cbar = plt.colorbar()
        cbar.set_label('Time')

        # Add a grid
        plt.grid(True)

        # Show the plot
        plt.show()
        plt.savefig(f"{args.out_path}/cx{cx}_rx{rx}_cy{cy}_ry{ry}/path.png")

    os.system(f"ffmpeg -y -i /data/mint/smooth_rotlight_gen/{args.model}/cx{cx}_rx{rx}_cy{cy}_ry{ry}/{sj_name}/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 {args.out_path}/{args.model}/cx{cx}_rx{rx}_cy{cy}_ry{ry}/{sj_name}.mp4 2> tmp.txt")


if __name__ == '__main__':
    if args.set_ is None:
        for i, sj in tqdm.tqdm(enumerate(os.listdir('/data/mint/Generated_Relighting_Dataset/'))):
        # for i, sj in tqdm.tqdm(enumerate(['src=60265.jpg', 'src=60268.jpg', 'src=60340.jpg', 'src=60374.jpg', 'src=60414.jpg', 'src=60865.jpg', 'src=61003.jpg', 'src=61062.jpg', 'src=61777.jpg'])):
            smooth_spiral(sj_name = sj, n_frames=49, savepath=False if i==0 else False)
    else:
        path = f'/data/mint/DPM_Dataset/generated_dataset_80perc/gen_images/{args.set_}'

        imgs = glob.glob(f'{path}/*.png')
        sj_dict = {}
        for f in imgs:
            sj_dict['src=' + f.split('/')[-1].split('_')[0] + '.jpg'] = f
        # for i, sj in tqdm.tqdm(enumerate(sj_dict.keys())):
            # smooth_spiral(sj_name = sj, n_frames=49, savepath=False)
        data = zip(list(sj_dict.keys()), [49] * len(sj_dict.keys()), [False] * len(sj_dict.keys()))
        with multiprocessing.Pool() as pool:
            _ = pool.starmap(smooth_spiral, data)
