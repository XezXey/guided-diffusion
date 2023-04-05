import argparse
import configparser
import cv2
import numpy as np
import glob
import os

# This script for generate random OpticalFlow config and random warping
# The purpose is to compare how good each config is?
parser = argparse.ArgumentParser(description='Script for generate random OpticalFlow config and random warping')
parser.add_argument('--image_dir', dest='image_dir', help='path to images directory')
parser.add_argument('--n_samples', dest='n_samples', help='number of samples to eval')
parser.add_argument('--step_size_limit', dest='step_size_limit', help='limit of how large step is')
args = parser.parse_args()

# Read the image directory to find the possible range for random
file_pattern = '/raw_frame*.png'
image_filelist = sorted(glob.glob(args.image_dir + file_pattern))
frame_list = np.random.choice(a=np.arange(0, len(image_filelist)-1, 1), size=int(args.n_samples), replace=False)
step_list = np.random.choice(a=np.arange(1, int(args.step_size_limit)), size=int(args.n_samples), replace=True)

# Write the random into a random
random_eval_step = ""
for frame, step_size in list(zip(frame_list, step_list)):
  # Write to the frame
  random_eval_step += '{},{}\n'.format(frame, step_size)

print(random_eval_step)
config = configparser.ConfigParser()
random_eval_path = args.image_dir + '/random_eval.ini'
with open(random_eval_path, 'w') as config_writer:
  config['IMAGES_DIR'] = {'n_frame':len(image_filelist), 'random_eval_step':random_eval_step, 'n_samples':args.n_samples}
  config.write(config_writer)
