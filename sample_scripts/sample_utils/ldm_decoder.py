import numpy as np
import torch as th
import argparse
import sys
import os
import tqdm
sys.path.append('../')
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from PIL import Image
import matplotlib.pyplot as plt
import blobfile as bf

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = th.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_model(cfg_file, ckpt_file):
    config = OmegaConf.load(cfg_file)  
    model = load_model_from_config(config, ckpt_file)
    return model


class LDM_Decoder:
    def __init__(self, decoder_type, scale) -> None:
        if decoder_type in ['kl-f4', 'kl-f8', 'vq-f4', 'vq-f8']:
            cfg_file = "/home/mint/Dev/DiFaReli/LDM/latent-diffusion/models/first_stage_models/{decoder_type}/config.yaml"
            ckpt_file = "/home/mint/Dev/DiFaReli/LDM/latent-diffusion/models/first_stage_models/{decoder_type}/config.ckpt"
        else: raise NotImplementedError(f"Decoder type {decoder_type} not implemented")
        self.model = get_model(cfg_file, ckpt_file)
        
    