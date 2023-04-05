import os
import configparser
import subprocess
import ipdb

config = configparser.ConfigParser()
config.read('./evaluation_opticalflow.ini')
print(config['IMAGE_DIR_LIST']['image_dir_list'].split('\n'))
image_dir_list = config['IMAGE_DIR_LIST']['image_dir_list'].split('\n')
config_list = config['CONFIG_OPTICALFLOW']['config_opticalflow'].split('\n')
step_size_limit = int(config['CONFIG_WARPING']['config_warping_stepsize_limit'])

for video_name in tqdmimage_dir_list:
  for config_opticalflow in config_list:
    for step_size in range(1, step_size_limit):
      cmd = ['python3', 'optical_flow_warp.py',
             '--image_dir', '../demo_output/{}/OpticalFlowEstimation/imagesFrame'.format(video_name),
             '--forward_flow', '../demo_output/{}/OpticalFlowEstimation/{}/flows_forward.npy'.format(video_name, config_opticalflow),
             '--backward_flow', '../demo_output/{}/OpticalFlowEstimation/{}/flows_backward.npy'.format(video_name, config_opticalflow),
             '--output_savepath', '../demo_output/{}/OpticalFlowEstimation/{}/warpedResults/'.format(video_name, config_opticalflow),
             '--config', config_opticalflow,
             '--mode', 'auto',
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

