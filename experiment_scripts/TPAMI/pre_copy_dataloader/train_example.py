# init the autoencoder
from typing import Any
import os
import arch_example
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch
import glob
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler, Sampler
import numpy as np
print("[#] Torch lightning version:", pl.__version__)
print("[#] PyTorch version:", torch.__version__)

# Create a DataLoader with batch_size=64 and shuffle=True
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = glob.glob(data + '/*.txt')
        self.labels = glob.glob(labels + '/*.txt')
        self.data_path_dict = {v.split('/')[-1]: v for v in self.data}
        self.labels_path_dict = {v.split('/')[-1]: v for v in self.labels}
        self.data_path_dict_keylist = list(self.data_path_dict.keys())
        self.labels_path_dict_keylist = list(self.labels_path_dict.keys())
        # self.__getitem__(0)
        # exit()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x_load_path = self.data_path_dict[self.data_path_dict_keylist[index]].replace('data_src', 'data_tmp')
        y_load_path = self.labels_path_dict[self.labels_path_dict_keylist[index]].replace('data_src', 'data_tmp')
        # print(x_load_path, y_load_path)
        # print(os.path.exists(x_load_path), os.path.exists(y_load_path))
        x = np.loadtxt(x_load_path, delimiter=',').astype(np.float32)
        y = np.loadtxt(y_load_path).astype(np.float32)
        # print(x.shape, y.shape)
        return x, y, index

# Trainloader
pl.seed_everything(47)
batch_size = 1
train_dataset = CustomDataset('./data_src/input', './data_src/label')
# train_dataset = CustomDataset('/home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/pre_copy_dataloader/data_src/input', '/home/mint/Dev/DiFaReli/difareli-faster/experiment_scripts/TPAMI/pre_copy_dataloader/data_src/label')
# record_indices_sampler = RecordIndicesSampler(train_dataset)
# assert False
            
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True, pin_memory=True, persistent_workers=True)
# mockup_loader = DataLoader(mockup_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True, pin_memory=True, persistent_workers=True)
# print(record_indices_sampler.indices)

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 10))

autoencoder = arch_example.LitAutoEncoder(encoder, decoder, 1, train_loader, batch_size, train_dataset)
# autoencoder.pl_trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=mockup_loader)
autoencoder.pl_trainer.fit(model=autoencoder, train_dataloaders=train_loader)
exit()