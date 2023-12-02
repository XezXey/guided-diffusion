
import glob, os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler, Sampler

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return np.loadtxt(self.data[index])
    
if __name__ == '__main__':
    if os.path.exists('./3.txt'):
        os.system('rm 3.txt')
    file = glob.glob('./*.txt') + ['./3.txt']
    print(file)
    dataset = CustomDataset(file)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True, pin_memory=True, persistent_workers=True)

    for i, x in enumerate(dataloader):
        print("before : ", i, x)
        if i == 0:
            with open('./3.txt', 'w') as f:
                f.write('3')
        print("after : ", i, x)
        input(dataset.data)