import os
import configparser
import subprocess
import ipdb
import argparse

parser = argparse.ArgumentParser(description='Params evaluation of opticalflow')
parser.add_argument('--evaluation_opticalflow_path', dest='evaluation_opticalflow_path', help='path to evaluation_opticalflow.ini file', required=True)
parser.add_argument('--mode', dest='mode', help='Mode for running the warping')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.evaluation_opticalflow_path)
image_dir_list = config['IMAGE_DIR']['image_dir'].split('\n')
config_list = config['CONFIG_OPTICALFLOW']['config_opticalflow'].split('\n')
step_size_limit = int(config['CONFIG_WARPING']['config_warping_stepsize_limit'])
if args.mode=='random':
  step_size_limit=2

for video_name in image_dir_list:
  for config_opticalflow in config_list:
    for step_size in range(1, step_size_limit):
      cmd = ['python3', 'optical_flow_warp.py',
             '--image_dir', './Data/{}/imagesFrame'.format(video_name),
             '--forward_flow', './Data/{}/OpticalFlowEstimation/{}/flows_forward.npy'.format(video_name, config_opticalflow),
             '--backward_flow', './Data/{}/OpticalFlowEstimation/{}/flows_backward.npy'.format(video_name, config_opticalflow),
             '--output_savepath', './Data/{}/OpticalFlowEstimation/{}/warpedResults/'.format(video_name, config_opticalflow),
             '--output_random_savepath', './Data/{}/OpticalFlowEstimation/'.format(video_name),
             '--config', config_opticalflow,
             '--mode', args.mode,
             '--step_size', str(step_size),
             '--video_name', video_name]
      print("Running command : ", end='')
      print(' '.join(cmd))
      try:
        err = subprocess.call(cmd)
        if err:
          ipdb.set_trace()
      except OSError:
        ipdb.set_trace()
        print('OSError')

