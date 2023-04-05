import configparser
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Generating a config file for Optical Flow warping and evaluation")
parser.add_argument('--config_opticalflow_path', dest='config_opticalflow_path', help='path to a config_opticalflow.ini file', required=True)
parser.add_argument('--evaluation_opticalflow_path', dest='evaluation_opticalflow_path', help='path to a evaluation_opticalflow.ini file', required=True)
parser.add_argument('--image_dir', dest='image_dir', help='path of images directory', required=True)
parser.add_argument('--mode', dest='mode', help='mode of receive a config of opticalflow', required=True)
args = parser.parse_args()


config_opticalflow = configparser.ConfigParser()
config_opticalflow.read(args.config_opticalflow_path)

config_evaluation = configparser.ConfigParser()
config_evaluation.read(args.evaluation_opticalflow_path)
config_evaluation['CONFIG_OPTICALFLOW']['config_opticalflow'] = ''
config_name_list = []

if args.mode=='random':
  # Random a config for opticalflow estimation
  alpha_list = np.random.choice(a=np.around(np.linspace(0.005, 0.045, num=30), decimals=3), size=1)
  ratio_list = np.random.choice(a=np.around(np.arange(0.6, 0.96, 0.03), decimals=2), size=1)
  minWidth_list = np.random.choice(a=np.arange(2, 30), size=1)
  with open(args.config_opticalflow_path, 'w') as config_writer:
    config_opticalflow['CONFIG_OPTICALFLOW'] = {'alpha':alpha_list[0], 'ratio':ratio_list[0], 'minWidth':minWidth_list[0]}
    config_opticalflow.write(config_writer)
elif args.mode=='config':
  alpha_list = config_opticalflow['CONFIG_OPTICALFLOW']['alpha'].split('\n')
  ratio_list = config_opticalflow['CONFIG_OPTICALFLOW']['ratio'].split('\n')
  minWidth_list = config_opticalflow['CONFIG_OPTICALFLOW']['minWidth'].split('\n')
print("[*]List of evaluation config")
for alpha in alpha_list:
  for ratio in ratio_list:
    for minWidth in minWidth_list:
      print('===>{}alpha_{}ratio_{}minWidth'.format(float(alpha), float(ratio), float(minWidth)))
      config_name_list.append('{}alpha_{}ratio_{}minWidth'.format(float(alpha), float(ratio), float(minWidth)))

config_name_string = '\n'.join(config_name_list)
config_evaluation['CONFIG_OPTICALFLOW']['config_opticalflow'] = config_name_string
config_evaluation['IMAGE_DIR']['image_dir'] = args.image_dir
with open(args.evaluation_opticalflow_path, 'w') as config_writer:
  config_evaluation.write(config_writer)
