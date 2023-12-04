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
from pytorch_lightning.utilities.rank_zero import rank_zero_only



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
                )
        self.lazy_load_samples_size = 10000 // (n_gpus * batch_size)
        

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
        self.chunk_ptr = 0
        # This will store all files to be copied to this machine (Dynamically change every training epoch)
        self.all_files = {'input': [], 'label': []}
        # Get all indices for one epoch
        all_true_indices = []
        #NOTE: Each of thread will have their own train_dataloader, also the indices
        # print("[#] Length : " , self.pl_trainer.train_dataloader.__len__(), "of rank", self.global_rank)
        
        all_true_indices = [i for i in self.pl_trainer.train_dataloader.loaders._index_sampler]
            
        # Create a lazy load chunk by using indices of lazy_load_chunk_size 
        # (e.g. 0 - 50, 49 - 101, 99 - 151, ...) where the lazy_load_chunk_size = 50 in this example
        print(self.pl_trainer.train_dataloader.__len__(), self.batch_size)
        lazy_load_chunk_idx = [i for i in range(0, self.pl_trainer.train_dataloader.__len__(), self.lazy_load_samples_size)] + [self.pl_trainer.train_dataloader.__len__()]
        # Create tuple of start and end indices for each chunk
        out = []
        out_start = []
        for i in range(len(lazy_load_chunk_idx)-1):
            out.append((lazy_load_chunk_idx[i], lazy_load_chunk_idx[i+1]))
            out_start.append(lazy_load_chunk_idx[i])
        self.lazy_load_chunk_idx = out
        self.lazy_load_chunk_start = out_start
        
        # Create a list of files to be copied to this machine [[file1, file2, ...], [file10, file11, ...], ...]
        print(f"[#] Chunk idx: {self.lazy_load_chunk_idx} from {self.global_rank}")
        for c in self.lazy_load_chunk_idx:
            sub = sum(all_true_indices[c[0]:c[1]], [])
            tmp_data = [self.train_dataset.data_path_dict[self.train_dataset.data_path_dict_keylist[j]] for j in sub]
            tmp_label = [self.train_dataset.labels_path_dict[self.train_dataset.labels_path_dict_keylist[j]] for j in sub]
            self.all_files['input'].append(tmp_data)
            self.all_files['label'].append(tmp_label)
        
        # Pre-copying for the first batch
        self.current_files = {
            'input': self.all_files['input'][self.lazy_load_chunk_start.index(0)],
            'label': self.all_files['label'][self.lazy_load_chunk_start.index(0)]
        }
        subprocess.run(f"cp {' '.join(self.current_files['input'])} ./data_tmp/input/".split(' '))
        subprocess.run(f"cp {' '.join(self.current_files['label'])} ./data_tmp/label/".split(' '))
        
        self.chunk_ptr += 1
        self.previous_files = copy.deepcopy(self.current_files)
        
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # # Copy the data to this machine
        
        print(f"[#] Counting file in data_tmp/input: {len(os.listdir('./data_tmp/input'))} from {self.global_rank}")
        print(f"[#] Counting file in data_tmp/label: {len(os.listdir('./data_tmp/label'))} from {self.global_rank}")
        n = self.lazy_load_samples_size
        if batch_idx + n in self.lazy_load_chunk_start:
            print(f"[#] Copying...from {self.global_rank}")
            self.current_files = {
                'input': self.all_files['input'][self.lazy_load_chunk_start.index(batch_idx + n)],
                'label': self.all_files['label'][self.lazy_load_chunk_start.index(batch_idx + n)]
            }
            assert self.chunk_ptr == self.lazy_load_chunk_start.index(batch_idx + n)
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
            
        if batch_idx - (n*2) in self.lazy_load_chunk_start and batch_idx - (n*2) != 0:
            os.system(f"rm {' '.join(self.previous_files['input']).replace('data_src', 'data_tmp')}")
            os.system(f"rm {' '.join(self.previous_files['label']).replace('data_src', 'data_tmp')}")
            # print(f"[#] Done removing...from {self.global_rank}")
            self.previous_files = copy.deepcopy(self.current_files)
        
    
    @rank_zero_only
    def on_train_epoch_end(self):
        # Prepare the data on every start of epoch
        if os.path.exists('./data_tmp/input'):
            os.system(f"rm -r ./data_tmp/input")
        if os.path.exists('./data_tmp/label'):
            os.system(f"rm -r ./data_tmp/label")
        os.makedirs('./data_tmp/input', exist_ok=True)
        os.makedirs('./data_tmp/label', exist_ok=True)
        print(f"[#] Finished epoch {self.global_rank} and removing all files in data_tmp")
       