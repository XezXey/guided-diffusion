# Author: Deepak Pathak (c) 2016
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
import time
import argparse
import sys
import os
import subprocess
from tqdm import tqdm
# Adding the pyflow path directories
sys.path.append('../../pyflow')
# sys.path.append('/home/puntawat/Mint/Work/Vision/3D_Human_Reconstruction/human_dynamics/pyflow')
import pyflow
import matplotlib.pyplot as plt
import cv2
import multiprocessing
import configparser
import glob

parser = argparse.ArgumentParser(description="Optical flow debugger by warpping the images")
parser.add_argument('--image_dir', dest='image_dir', help='Input the image directories', required=True)
parser.add_argument('--output_savepath', dest='output_savepath', help='Save path of warped images', required=True)
parser.add_argument('--video_name', dest='video_name', help='current process video', required=True)
# parser.add_argument('--file_pattern', dest='file_pattern', help='filename pattern of the input')
parser.add_argument('--config_opticalflow_path', dest='config_opticalflow_path', help='path to a config_opticalflow.ini file')
args = parser.parse_args()

# Flow Options:
alpha = 0.03 # Smoothness of frame (Higher means small movement/difference will be captured => small optical flow offset)
ratio = 0.95 # Decreasing rate of the image size per each pyramid
minWidth = 5 # Size of receptive field for each image (Lower is better to capture a large movement)
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


def compute_flow(i, im1, im2, normalize, output_optical_flow_path_fw, output_optical_flow_path_bw):
  # Normalization flag for input images
  if normalize:
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) /255.
  # Calculate a forward sequence optical flow 
  u, v, im2W = pyflow.coarse2fine_flow(
    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
  flows_forward = np.concatenate((u[..., None], v[..., None]), axis=2)
  # Create a Forward frame
  cv2.imwrite(output_optical_flow_path_fw + 'outflow_warped_frame{}.png'.format(i), im2W[:, :, ::-1] * 255)

  # Calculate a backward sequence optical flow 
  u, v, im2W = pyflow.coarse2fine_flow(
    im2, im1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
  flows_backward = np.concatenate((u[..., None], v[..., None]), axis=2)
  # Create a Backward frame
  cv2.imwrite(output_optical_flow_path_bw + 'outflow_warped_frame{}.png'.format(i), im2W[:, :, ::-1] * 255)
  print("SHAPE : ", flows_forward.shape, flows_backward.shape)
  return flows_forward, flows_backward

def optical_flow_estimation(images, config_opticalflow, output_path='./', normalize=True):
  # Assign the value of parameters from dict.
  global alpha, ratio, minWidth
  alpha = config_opticalflow['alpha']
  ratio = config_opticalflow['ratio']
  minWidth = config_opticalflow['minWidth']
  # Create an optical flow folder for saving a results
  config_opticalflow_path = "{}alpha_{}ratio_{}minWidth/".format(alpha, ratio, minWidth)
  output_optical_flow_path = output_path + "OpticalFlowEstimation/" + config_opticalflow_path
  output_optical_flow_path_fw = output_optical_flow_path + "Forward/"
  output_optical_flow_path_bw = output_optical_flow_path + "Backward/"
  if not os.path.exists(output_optical_flow_path) or not os.path.exists(output_optical_flow_path_fw) or not os.path.exists(output_optical_flow_path_bw):
    # Create the directory
    os.makedirs(output_optical_flow_path)
    os.makedirs(output_optical_flow_path_fw)
    os.makedirs(output_optical_flow_path_bw)
  else :
    print("[*]The Opticalflow Estimation already exists...")
    exit()

  # Computing flows
  flows_forward = []
  flows_backward = []
  pooling = multiprocessing.Pool(multiprocessing.cpu_count())
  print("Optical Flow Estimation ...")
  # Using multiprocessing pool to compute flows in parallel
  flows_fw_bw = pooling.starmap(compute_flow, (zip(list(range(len(images)-1)),
                                            [images[i] for i in range(len(images)-1)],
                                            [images[i+1] for i in range(len(images)-1)],
                                            [normalize] * (len(images)-1),
                                            [output_optical_flow_path_fw] * (len(images)-1),
                                            [output_optical_flow_path_bw] * (len(images)-1),)))
  flows_fw_bw = np.array(flows_fw_bw)
  # Split forward and backward from flows_fw_bw (#N_frame, 2(forward and backward), h, w, 2(u and v))
  flows_forward = np.array(flows_fw_bw[:, 0, ...])
  flows_backward = np.array(flows_fw_bw[:, 1, ...])
  # Summary and save to the output path
  print("OPTICAL FLOW SUMMARY : ")
  print("Forward Sequence FLOWS : ", flows_backward.shape)
  print("Backward Sequence FLOWS : ", flows_forward.shape)
  print("Saving optical_flow_estimation to : ", output_optical_flow_path)
  np.save(output_optical_flow_path + "flows_forward.npy", flows_forward)
  np.save(output_optical_flow_path + "flows_backward.npy", flows_backward)
  return flows_forward, flows_backward

if __name__ == '__main__':
  # Running a flow estimation to given input/video
  config = configparser.ConfigParser()
  config.read(args.config_opticalflow_path)
  output_path = args.output_savepath
  image_dir = args.image_dir
  alpha_list = config['CONFIG_OPTICALFLOW']['alpha'].split('\n')
  ratio_list = config['CONFIG_OPTICALFLOW']['ratio'].split('\n')
  minWidth_list = config['CONFIG_OPTICALFLOW']['minWidth'].split('\n')
  images = np.array([cv2.imread(frame)[..., ::-1] for frame in sorted(glob.glob(image_dir + '/raw_frame*.png'))])
  for alpha in alpha_list:
    for ratio in ratio_list:
      for minWidth in minWidth_list:
        config_name = '{}alpha_{}ratio_{}minWidth'.format(alpha, ratio, minWidth)
        config_opticalflow = {'alpha':float(alpha),
                              'ratio':float(ratio),
                              'minWidth':float(minWidth)}
        forward_flow, backward_flow = optical_flow_estimation(images, config_opticalflow, output_path)

