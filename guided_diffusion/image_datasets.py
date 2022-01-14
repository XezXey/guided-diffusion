import math
import random

from PIL import Image
from matplotlib import image
from numpy.core.numeric import full_like
import pandas as pd
import blobfile as bf
from mpi4py import MPI
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from .recolor_util import recolor as recolor
import matplotlib.pyplot as plt

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    flip=False,
    random_crop=False,
    random_flip=False,
    out_c='rgb',
    z_cond=False,
    precomp_z="",
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    if z_cond and precomp_z != "":
        precomp_z = pd.read_csv(precomp_z, header=None, sep=" ", index_col=False, names=["img_name"] + list(range(27)), lineterminator='\n')
        precomp_z = precomp_z.set_index('img_name').T.to_dict('list')
    else: precomp_z = None

    img_dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        flip=flip,
        out_c=out_c, 
        precomp_z=precomp_z
    )

    if deterministic:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


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


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
        flip=False,
        out_c='rgb',
        precomp_z=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.precomp_z = precomp_z
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.flip = flip
        self.out_c = out_c

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        arr = self.augmentation(pil_image=pil_image)
        arr = self.recolor(img=arr, out_c=self.out_c)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        if self.precomp_z is not None:
            img_name = path.split('/')[-1]
            out_dict["precomp_z"] = np.array(self.precomp_z[img_name])

        return np.transpose(arr, [2, 0, 1]), out_dict

    def augmentation(self, pil_image):
        # Resize image by cropping to match the resolution
        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        if self.flip:
            arr = arr[:, ::-1]
        
        return arr
    
    def recolor(self, img, out_c):
        if out_c == 'sepia':
            img_ = recolor.rgb_to_sepia(img)
            img_ = img_.astype(np.float32) / 127.5 - 1
        elif out_c == 'hsv':
            img_ = recolor.rgb_to_hsv(img)
            img_h_norm = img_[..., [0]].astype(np.float32) / 90.0 - 1
            img_s_norm = img_[..., [1]].astype(np.float32) / 127.5 - 1
            img_v_norm = img_[..., [2]].astype(np.float32) / 127.5 - 1
            img_ = np.concatenate((img_h_norm, img_s_norm, img_v_norm), axis=2)
        elif out_c == 'hls':
            img_ = recolor.rgb_to_hls(img)
            img_h_norm = img_[..., [0]].astype(np.float32) / 90.0 - 1
            img_l_norm = img_[..., [1]].astype(np.float32) / 127.5 - 1
            img_s_norm = img_[..., [2]].astype(np.float32) / 127.5 - 1
            img_ = np.concatenate((img_h_norm, img_l_norm, img_s_norm), axis=2)
        elif out_c == 'ycrcb':
            img_ = recolor.rgb_to_ycrcb(img)
            img_ = img_.astype(np.float32) / 127.5 - 1
        elif out_c == 'luv':
            img_ = recolor.rgb_to_luv(img)
            img_ = img_.astype(np.float32) / 127.5 - 1
        elif out_c in ['rgb', 'rbg', 'brg', 'bgr', 'grb', 'gbr']:
            img_ = recolor.rgb_sw_chn(img, ch=out_c)
            img_ = img_.astype(np.float32) / 127.5 - 1
        else: raise NotImplementedError

        return img_


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
