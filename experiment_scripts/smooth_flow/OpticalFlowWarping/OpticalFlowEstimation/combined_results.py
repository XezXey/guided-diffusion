import pandas as pd
import numpy as np
import configparser
import argparse
import xlsxwriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Combined the multiple OpticalFlow config results')
# parser.add(

config = configparser.ConfigParser()
config.read('./evaluation_opticalflow.ini')
step_size_limit = int(config['CONFIG_WARPING']['config_warping_stepsize_limit'])

image_dir_list = config['IMAGE_DIR_LIST']['image_dir_list'].split('\n')
config_list = config['CONFIG_OPTICALFLOW']['config_opticalflow'].split('\n')
step_size_limit = int(config['CONFIG_WARPING']['config_warping_stepsize_limit'])

def aggregate_results(result_path, config_opticalflow):
  metrics_list = config['CONFIG_EVALUATION_METRICS']['config_evaluation_metrics'].split('\n')
  result_df = pd.read_excel(result_path)
  result_df = result_df.aggregate(['min', 'max', 'mean', np.std, np.var, 'count'])
  col_names = []
  values = []
  for i in range(len(metrics_list)):
    col_names = col_names + list('{}_{}'.format(metrics_list[i], eval_metrics) for eval_metrics in list(result_df[metrics_list[i]].index))
    values = values + list(result_df[metrics_list[i]].ravel())
  data = {config_opticalflow : values}
  return pd.DataFrame.from_dict(data, orient='index', columns=col_names)

for video_name in tqdm(image_dir_list, leave=True):
  tqdm(image_dir_list).set_description(video_name)
  summary_df_list = []
  for step_size in range(1, step_size_limit):
    agg_df = []
    for config_opticalflow in config_list:
      warpedResults_path = '../demo_output/{}/OpticalFlowEstimation/{}/warpedResults/{}_step_size/'.format(video_name, config_opticalflow, step_size)
      warpedResults_filename = '{}_{}step_warped_l2_results.xlsx'.format(config_opticalflow, step_size)
      # Concatenate each config(each row) into the agg_df
      agg_df.append(aggregate_results(warpedResults_path + warpedResults_filename, config_opticalflow))
    # Append agg_df into summary_df_list : The summary_df_list will contain agg_df of each step_size
    summary_df_list.append(pd.concat(agg_df))

  # Write the summary_df of every step_size and configs into excel files
  writer = pd.ExcelWriter('../demo_output/{}/OpticalFlowEstimation/opticalflow_params_summary.xlsx'.format(video_name), engine='xlsxwriter')
  for i in range(1, step_size_limit):
    # print(summary_df_list[i])
    pd.DataFrame(summary_df_list[i-1]).to_excel(writer, sheet_name='{}step_size'.format(i), index=True)
  writer.save()
  writer.close()



