import numpy as np
import torch as th
import argparse
import cv2
import torch.nn.functional as F
import os, glob, tqdm
from PIL import Image
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--set', nargs='+', type=str, default=['train', 'valid'])
parser.add_argument('--savepath', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--ext', type=str, default='jpg')
args = parser.parse_args()

class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        name = img_path.split('/')[-1].split('.')[0]
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        image = (image / 255.0)
        return image, name 

def extract_laplacian(img, name, set_):
    def create_gaussian_pyramid(img, level=2):
        pys = [img]
        for i in range(level-1):
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_AREA)
            pys.append(img)
        return pys

    def create_laplacian_pyramid(img, level=5):
        pys = create_gaussian_pyramid(img, level)
        for i in range(level-1):
            pys[i] = pys[i] - cv2.resize(pys[i+1], (pys[i].shape[1], pys[i].shape[0]))
        return pys

    lap = create_laplacian_pyramid(img)
    np.save(os.path.join(args.savepath, set_, name), lap[0][..., np.newaxis], allow_pickle=True)

if __name__ == '__main__':
    img_path = glob.glob(os.path.join(args.path, '*.jpg'))
    for set_ in args.set:
        os.makedirs(os.path.join(args.savepath, set_), exist_ok=True)
        img_path = glob.glob(os.path.join(args.path, set_, f'*.{args.ext}'))
        dataset = CustomImageDataset(img_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        for i, data in tqdm.tqdm(enumerate(dataloader, 0)):
            img, name = data
            img = [tmp for tmp in img]
            data = [(img[i].cpu().numpy(), name[i], set_) for i in range(len(img))]
            pool = mp.Pool(processes=mp.cpu_count())
            _ = pool.starmap(extract_laplacian, data)
            pool.close()
            pool.join()

