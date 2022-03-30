import numpy as np
import matplotlib.pyplot as plt
import torch as th
import warnings
warnings.filterwarnings("ignore") 
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import blobfile as bf
import tqdm
import cv2 
from PIL import Image
from detector.mtcnn.visualization_utils import show_bboxes
from utils import get_central_face_attributes, align_face


class ArcFaceDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def preprocess(fn, vis=False):
    src_img = cv2.imread(fn)  # BGR

    bounding_boxes, landmarks = get_central_face_attributes(fn, detector='retinaface')
    img_detected = show_bboxes(src_img.copy(), bounding_boxes, landmarks)
    img_aligned = align_face(fn, landmarks)

    if vis:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(7, 4), dpi=80)
        ax[0].imshow(src_img[..., ::-1])
        ax[1].imshow(img_detected[..., ::-1])
        ax[2].imshow(img_aligned[..., ::-1])
        plt.show()

    return img_aligned[..., ::-1]   # Return in RGB

def get_arcface_emb(img_path, device, vis=False):

    # Model parameters
    image_w = 112
    image_h = 112
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'normalize': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'none':None
    }

    # loading model
    checkpoint = th.load('./cond_utils/arcface/pretrained/BEST_checkpoint_r18.tar', map_location=device)
    model = checkpoint['model'].module.to(device)

    ## load data for my testing only
    paths = _list_image_files_recursively(img_path)


    images_aligned = []
    emb_dict = {}
    for i, fn in tqdm.tqdm(enumerate(paths), desc='Open&Resize images'):
        prep_image = preprocess(fn, vis)
        images_aligned.append(prep_image)
        emb_dict[fn.split('/')[-1]] = None

    # reshape to prefered width height(112, 112)
    images_aligned = np.stack(images_aligned)
    # create dataset
    dataset = ArcFaceDataset(images_aligned, data_transforms['normalize'])
    loader = DataLoader(
        dataset,
        num_workers=24,
        batch_size=1000,
        shuffle=False,
        drop_last=False
    )

    # infer
    model.eval()
    emb = []
    with th.no_grad():
        for i, input_image in tqdm.tqdm(enumerate(loader), desc='Generate Face Embedding'):
            input_image = input_image.to(device)
            features = model(input_image)
            emb.append(features.detach().cpu().numpy())

    emb = np.concatenate(emb, axis=0)
    for i, k in enumerate(emb_dict.keys()):
        emb_dict[k] = {'faceemb':emb[i]}

    return emb_dict, emb
