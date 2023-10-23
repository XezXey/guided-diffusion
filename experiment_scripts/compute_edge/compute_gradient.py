import numpy as np
import torch as th
import argparse
import cv2
import torch.nn.functional as F
import os, glob, tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--savepath', type=str, required=True)
parser.add_argument('--set', nargs='+', type=str, default=['train', 'valid'])
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
        image = np.array(image)[..., np.newaxis]
        image = (image.transpose((2, 0, 1)) / 255.0).astype(np.float32)
        return image, name 

def gradient(img, device='cuda'):
    # image is in [B, 1, H, W] format
    assert img.shape[1] == 1
    
    sobel_x = th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=th.float32).to(device)
    sobel_y = th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=th.float32).to(device)

    # Apply Sobel operators to compute gradients
    gradient_x = F.conv2d(img, sobel_x.view(1, 1, 3, 3), padding='same')
    gradient_y = F.conv2d(img, sobel_y.view(1, 1, 3, 3), padding='same')

    # Compute the magnitude of the gradient
    gradient_magnitude = th.sqrt((gradient_x**2 + gradient_y**2) + 1e-6)
    # gradient_magnitude = th.sqrt(th.clamp((gradient_x**2 + gradient_y**2), 0.0) + 1e-6)

    return gradient_x, gradient_y, gradient_magnitude

def extract_edge(img, name, set_):
    gx, gy, gm = gradient(img)
    for i in range(gm.shape[0]):
        np.save(os.path.join(args.savepath, set_, name[i]), gm[i].cpu().numpy().transpose((1, 2, 0)))

if __name__ == '__main__':
    for set_ in args.set:
        os.makedirs(os.path.join(args.savepath, set_), exist_ok=True)
        img_path = glob.glob(os.path.join(args.path, set_, f'*.{args.ext}'))
        dataset = CustomImageDataset(img_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        for i, data in tqdm.tqdm(enumerate(dataloader, 0)):
            img, name = data
            extract_edge(img.cuda(), name, set_)
            # print(name, img.shape)
            
            