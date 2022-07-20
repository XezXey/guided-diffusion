# from __future__ import print_function 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--m1', type=str, required=True)
parser.add_argument('--m2', type=str, required=True)
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--cfg_name', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, required=True)
args = parser.parse_args()

import os, sys, glob
import numpy as np
import torch as th
import pytorch_lightning as pl
sys.path.insert(0, '../')
from sample_utils import ckpt_utils

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if th.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

# def compare_models(model1, model2):
#     for p1, p2 in zip(model1.parameters(), model2.parameters()):
#         if p1.data.ne(p2.data).sum() > 0:
#             return False
#     return True

if __name__ == '__main__':
    # Load ckpt
    ckpt_loader1 = ckpt_utils.CkptLoader(log_dir=args.m1, cfg_name=args.cfg_name)
    model_dict1, _ = ckpt_loader1.load_model(ckpt_selector=args.ckpt_selector, step=args.step)

    ckpt_loader2 = ckpt_utils.CkptLoader(log_dir=args.m2, cfg_name=args.cfg_name)
    model_dict2, _ = ckpt_loader2.load_model(ckpt_selector=args.ckpt_selector, step=args.step)

    print(compare_models(model_dict1['ImgCond'], model_dict2['ImgCond']))
