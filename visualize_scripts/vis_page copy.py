from tabnanny import check
import numpy as np
import streamlit as st
import argparse
import glob, os

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_folder', type=str)
args = parser.parse_args()

def path_to_dict(path, d):

    print("PATH", path)
    print("ABS", os.path.abspath(path))
    name = os.path.basename(os.path.abspath(path))
    print("NAME", name)

    # if os.path.isdir(path):
    #     if name not in d:
    #         d[name] = {}
    #     for x in os.listdir(path):
    #         path_to_dict(os.path.join(path,x), d[name])
    # else:
    #     d['files'].append(name)
    return d


mydict = path_to_dict('.', d = {'dirs':{},'files':[]})

if __name__ == '__main__':
    # print(f'{args.experiment_folder}/sampling_results/')
    # print(glob.glob(f'{args.experiment_folder}/sampling_results/*'))
    # print(glob.glob(f'{args.experiment_folder}/sampling_results/*'))
    sample_folder = os.listdir(f'{args.experiment_folder}/sampling_results/')
    # Folder picker button
    # st.title('DPM Sampling Visualizer')
    # selected = st.selectbox('Select a Sampling Folder?', sample_folder)
    # model_log = os.listdir(f'{args.experiment_folder}/sampling_results/{selected}')
    # model = {m : None for m in model_log}
    # for m in model.keys():
    #     checkpoint = os.listdir(f'{args.experiment_folder}/sampling_results/{selected}/{m}/')
    #     model[m] = {ckpt : None for ckpt in checkpoint}
    #     checkpoint = os.listdir(f'{args.experiment_folder}/sampling_results/{selected}/{m}/')
    #     for ckpt in checkpoint:
    #         dataset = os.listdir(f'{args.experiment_folder}/sampling_results/{selected}/{m}/{ckpt}')
    #         model[m][ckpt] = {d : None for d in dataset}
    #         for d in dataset:
    #             condition = os.listdir(f'{args.experiment_folder}/sampling_results/{selected}/{m}/{ckpt}/{d}')
    #             for cond in condition:
    #                 file = os.listdir(f'{args.experiment_folder}/sampling_results/{selected}/{m}/{ckpt}/{d}/{cond}/')
    
    mydict = path_to_dict(path=f'{args.experiment_folder}/sampling_results/', d = {})
    print(mydict)




