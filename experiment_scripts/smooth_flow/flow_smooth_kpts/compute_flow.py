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
sys.path.append('./pyflow')
import pyflow
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import multiprocessing
import configparser
import glob

parser = argparse.ArgumentParser(description="Optical flow debugger by warpping the images")
parser.add_argument('--image_dir', dest='image_dir', help='Input the image directories', required=True)
parser.add_argument('--output_savepath', dest='output_savepath', help='Save path of warped images', required=True)
parser.add_argument('--video_name', dest='video_name', help='current process video', required=True)
parser.add_argument('--config_opticalflow_path', dest='config_opticalflow_path', help='path to a config_opticalflow.ini file')
parser.add_argument('--chunk_size', dest='chunk_size', type=int, help='Chunking for multiprocess')
args = parser.parse_args()

# Flow Options:
alpha = 0.012 # Smoothness of frame (Higher means small movement/difference will be captured => small optical flow offset)
ratio = 0.75 # Decreasing rate of the image size per each pyramid
minWidth = 20 # Size of receptive field for each image (Lower is better to capture a large movement)
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

def flow_image(flow, im1):
  hsv = np.zeros(im1.shape, dtype=np.uint8)
  hsv[:, :, 0] = 255
  hsv[:, :, 1] = 255
  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  hsv[..., 0] = ang * 180 / np.pi / 2
  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
  rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  return rgb
  

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
  # cv2.imwrite(f'{output_optical_flow_path_fw}outflow_warped_frame_{i}.png', np.concatenate((im1[:, :, ::-1]*255, im2[:, :, ::-1]*255, im2W[:, :, ::-1] * 255), axis=1))
  # cv2.imwrite(f"{output_optical_flow_path_fw}frame_{i.split('_')[0]}.png", im1[:, :, ::-1] * 255)
  # cv2.imwrite(f"{output_optical_flow_path_fw}frame_{i.split('_')[1]}.png", im2[:, :, ::-1] * 255)
  # cv2.imwrite(f"{output_optical_flow_path_fw}outflow_warped_frame_{i}.png", im2W[:, :, ::-1] * 255)
  # cv2.imwrite(f"{output_optical_flow_path_fw}outflow_frame_{i}.png", flow_image(flow=flows_forward, im1=im1))

  # Calculate a backward sequence optical flow 
  u, v, im2W = pyflow.coarse2fine_flow(
    im2, im1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
  flows_backward = np.concatenate((u[..., None], v[..., None]), axis=2)
  # Create a Backward frame
  # cv2.imwrite(f"{output_optical_flow_path_bw}frame_{i.split('_')[0]}.png", im1[:, :, ::-1] * 255)
  # cv2.imwrite(f"{output_optical_flow_path_bw}frame_{i.split('_')[1]}.png", im2[:, :, ::-1] * 255)
  # cv2.imwrite(f"{output_optical_flow_path_bw}outflow_warped_frame_{i}.png", im2W[:, :, ::-1] * 255)
  # cv2.imwrite(f"{output_optical_flow_path_bw}outflow_frame_{i}.png", flow_image(flow=flows_backward, im1=im1))
  # cv2.imwrite(f'{output_optical_flow_path_bw}outflow_warped_frame_{i}.png', np.concatenate((im2[:, :, ::-1]*255, im1[:, :, ::-1]*255, im2W[:, :, ::-1] * 255), axis=1))
  
  
  print(f"SHAPE of {i}: ", flows_forward.shape, flows_backward.shape)
  return {i : {
              'fw' : flows_forward, 
              'bw' : flows_backward
            }
        }

def optical_flow_estimation(frames_idx, images, config_opticalflow, output_path='./', normalize=True):
  # Assign the value of parameters from dict.
  global alpha, ratio, minWidth
  alpha = config_opticalflow['alpha']
  ratio = config_opticalflow['ratio']
  minWidth = config_opticalflow['minWidth']
  # Create an optical flow folder for saving a results
  config_opticalflow_path = f"{alpha}alpha_{ratio}ratio_{minWidth}minWidth/"
  output_optical_flow_path = f"{output_path}/{args.video_name}/{config_opticalflow_path}"
  output_optical_flow_path_fw = f"{output_optical_flow_path}/Forward/"
  output_optical_flow_path_bw = f"{output_optical_flow_path}/Backward/"
  
  # Create the directory
  os.makedirs(output_optical_flow_path, exist_ok=True)
  os.makedirs(output_optical_flow_path_fw, exist_ok=True)
  os.makedirs(output_optical_flow_path_bw, exist_ok=True)

  # Computing flows
  pooling = multiprocessing.Pool(multiprocessing.cpu_count())
  print("Optical Flow Estimation ...")
  # Using multiprocessing pool to compute flows in parallel
  frames_idx = [f'frame{frames_idx[i]}_frame{frames_idx[i+1]}' for i in range(len(frames_idx)-1)]
  # print(frames_idx)
  inputs = list(zip(frames_idx,
              [images[i] for i in range(len(images)-1)],
              [images[i+1] for i in range(len(images)-1)],
              [normalize] * (len(images)-1),
              [output_optical_flow_path_fw] * (len(images)-1),
              [output_optical_flow_path_bw] * (len(images)-1),))
  
  chunk_size = args.chunk_size
  input_chunks = []
  for i in range(0, len(inputs), chunk_size):
      input_chunks.append(inputs[i:i+chunk_size])
      
  flows_res = {}
  count = 0
  for ic in tqdm(input_chunks):
      results = pooling.starmap(compute_flow, ic)
      flows_res.update({k: v for d in results for k, v in d.items()})
      np.save(file=f'{output_optical_flow_path}/{args.video_name}_sub_flows_{count}.npy', arr=results)
      count+=1
  
  
  flows_fw = []
  flows_bw = []
  for _, v in flows_res.items():
    flows_fw.append(v['fw'])
    flows_bw.append(v['bw'])
    
  flows_fw = np.stack(flows_fw, axis=0)
  flows_bw = np.stack(flows_bw, axis=0)
  # Summary and save to the output path
  print("OPTICAL FLOW SUMMARY : ")
  print("Forward Sequence FLOWS : ", flows_bw.shape)
  print("Backward Sequence FLOWS : ", flows_fw.shape)
  print("Saving optical_flow_estimation to : ", output_optical_flow_path)
  np.save(file=f'{output_optical_flow_path}/{args.video_name}_flows.npy', arr=flows_res)

def read_images(image_dir):
  print("[#] Reading images...")
  fn_list = glob.glob(image_dir + '/frame*.png')
  fn_list = sorted(fn_list, key=lambda x:int(x.split('/')[-1][5:-4]))
  frame_idx = []
  imgs = []
  for f in tqdm(fn_list):
    imgs.append(np.array(Image.open(f)))
    frame_idx.append(f.split('/')[-1][5:-4])
  return imgs, frame_idx
    
    
  

if __name__ == '__main__':
  # Running a flow estimation to given input/video
  config = configparser.ConfigParser()
  config.read(args.config_opticalflow_path)
  output_path = args.output_savepath
  image_dir = args.image_dir
  alpha_list = config['CONFIG_OPTICALFLOW']['alpha'].split('\n')
  ratio_list = config['CONFIG_OPTICALFLOW']['ratio'].split('\n')
  minWidth_list = config['CONFIG_OPTICALFLOW']['minWidth'].split('\n')
  images, frames_idx = read_images(image_dir)
  for alpha in alpha_list:
    for ratio in ratio_list:
      for minWidth in minWidth_list:
        config_name = '{}alpha_{}ratio_{}minWidth'.format(alpha, ratio, minWidth)
        config_opticalflow = {'alpha':float(alpha),
                              'ratio':float(ratio),
                              'minWidth':float(minWidth)}
        optical_flow_estimation(frames_idx, images, config_opticalflow, output_path)

