import numpy as np
import matplotlib.pyplot as plt
import PIL
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', type=str, required=True)
args = parser.parse_args()

print(glob.glob(f'{args.sample_dir}/*.mp4'))