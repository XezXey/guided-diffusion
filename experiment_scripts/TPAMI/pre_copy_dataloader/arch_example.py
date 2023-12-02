import os, copy, subprocess
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.strategies import DDPStrategy
import time


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, n_gpus, train_loader, batch_size, train_dataset):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.pl_trainer = pl.Trainer(
                    devices=n_gpus,
                    num_nodes=1,
                    max_epochs=1e6,
                    accelerator='gpu',
                    profiler='simple',
                    strategy=DDPStrategy(find_unused_parameters=False),
                    detect_anomaly=True,
                    reload_dataloaders_every_n_epochs=1,
                )
        self.lazy_load_samples_size = 100
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, dat_idx = batch
        # print(self.global_rank, batch_idx, dat_idx, x.shape, y.shape)
        # time.sleep(10)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def on_train_epoch_start(self):
        # Prepare the data on every start of epoch
        
        self.all_files = {'input': [], 'label': []}
        print(self.train_loader, self.global_rank)
        # Get all indices for one epoch
        all_indices = []
        all_labels = []
        all_true_indices = []
        n_data = self.pl_trainer.train_dataloader.__len__() * self.batch_size
        lazy_load_n_chunk = n_data // self.lazy_load_samples_size
        print(self.train_dataset.data_path_dict_keylist[:10])
        print(self.global_rank, lazy_load_n_chunk, n_data, self.lazy_load_samples_size)
        
        # for batch_data, batch_labels, index in enumerate(self.train_dataloader):
        # for batch_idx, batch in enumerate(self.pl_trainer.train_dataloader):
        #     # print(self.global_rank, batch_idx, batch[-1])
        #     # batch_indices = list(range(len(all_indices), len(all_indices) + len(batch_labels)))
        #     # all_indices.extend(batch_indices)
        #     all_true_indices.append(batch[-1].tolist())
        #     # all_labels.extend(batch_labels.tolist())
        # print("MIMI ", all_true_indices)
        
        print("MIMIMI")
        all_true_indices = [i for i in self.pl_trainer.train_dataloader.loaders._index_sampler]
        # print(minit == all_true_indices)
        print("ON START EPOCH : ", all_true_indices)
        print("FUCK YOU")
            
        # Create a lazy load chunk by using indices of lazy_load_chunk_size 
        # (e.g. 0 - 50, 49 - 101, 99 - 151, ...) where the lazy_load_chunk_size = 50 in this example
        # Let's (-1 and +1) for the first and the last to the lazy_load_chunk_idx to avoid some edge cases
        # lazy_load_chunk_idx = [i for i in range(0, self.pl_trainer.train_dataloader.__len__(), lazy_load_n_chunk)] + [self.pl_trainer.train_dataloader.__len__()]
        lazy_load_chunk_idx = [i for i in range(0, self.pl_trainer.train_dataloader.__len__(), self.lazy_load_samples_size)] + [self.pl_trainer.train_dataloader.__len__()]
        print(lazy_load_chunk_idx)
        # create tuple of start and end indices for each chunk
        out = []
        for i in range(len(lazy_load_chunk_idx)-1):
            # if i == 0:
            #     out.append((lazy_load_chunk_idx[i], lazy_load_chunk_idx[i+1]+1))
            # else:
            out.append((lazy_load_chunk_idx[i], lazy_load_chunk_idx[i+1]))
        lazy_load_chunk_idx = out
        # Map the lazy_load_chunk_idx to the true indices
        self.lazy_load_chunk_idx = lazy_load_chunk_idx
        self.lazy_load_chunk_start = [i[0] for i in lazy_load_chunk_idx]
        # Create a list of files to be copied to this machine [[file1, file2, ...], [file10, file11, ...], ...]
        # print(lazy_load_chunk_idx)
        # print(len(all_true_indices))
        # input()
        for c in lazy_load_chunk_idx:
            sub = sum(all_true_indices[c[0]:c[1]], [])
            tmp_data = [self.train_dataset.data_path_dict[self.train_dataset.data_path_dict_keylist[j]] for j in sub]
            tmp_label = [self.train_dataset.labels_path_dict[self.train_dataset.labels_path_dict_keylist[j]] for j in sub]
            self.all_files['input'].append(tmp_data)
            self.all_files['label'].append(tmp_label)
        self.chunk_ptr = 0
        # print(self.current_files)
        # print(all_true_indices)
        # print(self.train_dataset.data_path_dict_keylist)
        # exit()
        # print(len(all_true_indices), lazy_load_chunk_idx)
        # Copying for the first batch
        self.current_files = {
            'input': self.all_files['input'][self.lazy_load_chunk_start.index(0)],
            'label': self.all_files['label'][self.lazy_load_chunk_start.index(0)]
        }
        print(self.current_files['input'])
        # print('6947.txt' in self.current_files['input'])
        # exit()
        # print(self.current_files)
        if os.path.exists('./data_tmp/input'):
            os.system(f"rm -r ./data_tmp/input")
        if os.path.exists('./data_tmp/label'):
            os.system(f"rm -r ./data_tmp/label")
        os.makedirs('./data_tmp/input', exist_ok=True)
        os.makedirs('./data_tmp/label', exist_ok=True)
        assert self.chunk_ptr == self.lazy_load_chunk_start.index(0)
        # os.system(f"cp {' '.join(self.current_files['input'])} ./data_tmp/input/")        
        # os.system(f"cp {' '.join(self.current_files['label'])} ./data_tmp/label/")
        
        subprocess.run(f"cp {' '.join(self.current_files['input'])} ./data_tmp/input/".split(' '))    
        subprocess.run(f"cp {' '.join(self.current_files['label'])} ./data_tmp/label/".split(' '))
        # os.system('sleep 50')
        self.chunk_ptr += 1
        self.previous_files = copy.deepcopy(self.current_files)
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # print("[#] Finished training batch", self.global_rank, batch_idx, batch[-1])
        
        # print(f"[#] Counting file in data_tmp/input: {len(os.listdir('./data_tmp/input'))} from {self.global_rank}")
        # print(f"[#] Counting file in data_tmp/label: {len(os.listdir('./data_tmp/label'))} from {self.global_rank}")
        
        # # Copy the data to this machine
        n = self.lazy_load_samples_size
        if batch_idx+n in self.lazy_load_chunk_start:
            # print("[#] Copying data to this machine...")
            # print("[#] Loading from index of : ", self.lazy_load_chunk_start.index(batch_idx+50))
            self.current_files = {
                'input': self.all_files['input'][self.lazy_load_chunk_start.index(batch_idx+n)],
                'label': self.all_files['label'][self.lazy_load_chunk_start.index(batch_idx+n)]
            }
            assert self.chunk_ptr == self.lazy_load_chunk_start.index(batch_idx+n)
            # os.system(f"cp {' '.join(self.current_files['input'])} ./data_tmp/input/")        
            # os.system(f"cp {' '.join(self.current_files['label'])} ./data_tmp/label/")
            
            subprocess.run(f"cp {' '.join(self.current_files['input'])} ./data_tmp/input/".split(' '))    
            subprocess.run(f"cp {' '.join(self.current_files['label'])} ./data_tmp/label/".split(' '))
            # os.system('sleep 10')
            # print(f"[#] Done copying...from {self.global_rank}")
            assert self.current_files['input'] != self.previous_files['input']
            assert self.current_files['label'] != self.previous_files['label']
            
            self.chunk_ptr += 1
            # print(f"[#] Counting file in data_tmp/input: {len(os.listdir('./data_tmp/input'))} from {self.global_rank}")
            # print(f"[#] Counting file in data_tmp/label: {len(os.listdir('./data_tmp/label'))} from {self.global_rank}")
            
        if batch_idx-(n*2) in self.lazy_load_chunk_start and batch_idx-(n*2) != 0:
            os.system(f"rm {' '.join(self.previous_files['input']).replace('data_src', 'data_tmp')}")
            os.system(f"rm {' '.join(self.previous_files['label']).replace('data_src', 'data_tmp')}")
            # print(f"[#] Done removing...from {self.global_rank}")
            self.previous_files = copy.deepcopy(self.current_files)
        
    
    def on_train_epoch_end(self):
        # Prepare the data on every start of epoch
        print(self.train_loader, self.global_rank)
        all_true_indices = [i for i in self.pl_trainer.train_dataloader.loaders._index_sampler]
        print("End of epoch")
        print(all_true_indices)
       