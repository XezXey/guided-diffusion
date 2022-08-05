from email.policy import default
from pickletools import optimize
import pprint
from tabnanny import check
from turtle import onclick
from click import option
import numpy as np
import argparse
from skimage import io
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_folder', type=str)
args = parser.parse_args()

import streamlit as st
st.set_page_config(layout="wide")
if 'model_counter' not in st.session_state:
    st.session_state['model_counter'] = 0
if 'model_selector' not in st.session_state:
    st.session_state['model_selector'] = []

import glob, os
import json
import torchvision
import torch as th


def np_video(frames):
    print("Creating the video...", end='')
    torchvision.io.write_video('./temp_mint.mp4', video_array=frames, fps=5)
    video_file = open('./temp_mint.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(data=video_bytes)
    os.remove('temp_mint.mp4')


def check_leaf_dir(path):
    file = os.listdir(path)
    file = [os.path.join(path, f) for f in file]
    is_leaf = [os.path.isdir(f) for f in file]
    return is_leaf

def path_to_dict(path, d):

    name = os.path.basename(os.path.abspath(path))

    is_leaf_dir = check_leaf_dir(path)
    if os.path.isdir(path) and all(is_leaf_dir):
        if name not in d:
            d[name] = {}
        for x in os.listdir(path):
            path_to_dict(os.path.join(path,x), d[name])
    else:
        file = os.listdir(os.path.join(path))
        d[name] = [os.path.join(path, f) for f in file]
    return d

def get_frames(path):

    frames = []
    frame_anno = []
    for p in path:
        frame_idx = os.path.splitext(p.split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_anno.append(int(frame_idx))
        im = io.imread(p)
        frames.append(im)
    sorted_idx = np.argsort(frame_anno)
    sorted_frames = np.stack(frames)[sorted_idx]
    return sorted_frames


if __name__ == '__main__':
    mydict = path_to_dict(path=f'{args.experiment_folder}/sampling_results/', d = {})

    st.title('DPM Sampling Visualizer')

    with st.sidebar:
        col1, col2 = st.columns([1, 1])
        with col1:
            add_button = st.button("Add model")
        with col2:
            rem_button = st.button("Remove model")
        
        if add_button and st.session_state.model_counter < 4:
            #TODO: #N of Max model to add
            st.session_state.model_counter += 1
            st.experimental_rerun()
        if rem_button and st.session_state.model_counter > 1:
            st.session_state.model_counter -= 1
            st.experimental_rerun()

    model_selector = {}
    with st.sidebar:
        for i in range(st.session_state.model_counter):
            with st.expander(label=f"Model #{i+1}"):
                model_selector[i] = {
                    'sampling_folder': st.selectbox(
                        label = f"Sampling Folder #{i+1}", 
                        options=mydict['sampling_results'].keys()
                    )
                }
                model_selector[i].update({
                    'model_name': st.selectbox(
                        label = f"Model name #{i+1}", 
                        options=mydict['sampling_results'][model_selector[i]['sampling_folder']].keys(),
                        index=i
                    )
                })

                model_selector[i].update({
                    'ckpt': st.selectbox(
                        label = f"Checkpoint #{i+1}", 
                        options=mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']].keys()
                    )
                })
                model_selector[i].update({
                    'dataset': st.selectbox(
                        label = f"Dataset #{i+1}", 
                        options=mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']].keys()
                    )
                })
                model_selector[i].update({
                    'cond': st.selectbox(
                        label = f"Condition #{i+1}", 
                        options=mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']].keys()
                    )
                })

                # model_selector[i].update({
                #     'src': list(mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']].keys()),
                # })

                # for each_src in list(mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']].keys()):
                #     model_selector[i].update({
                #         'subject' : {each_src : {}}
                #     })
                #     # model_selector[i][each_src] = list(mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']][each_src].keys())
                #     for each_dst in list(mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']]['subject'][each_src].keys()):
                #         model_selector[i][each_src].update({
                #             each_dst : mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']]['subject'][each_src].keys()
                #         })

                # model_selector[i].update({
                #     'dst': mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']][model_selector[i]['src']].keys()
                # })

                # print(model_selector[i]['src'])
                # print(mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']].keys())
                # print(mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']][model_selector[i]['src']].keys())
                info = f'''
                    name = {model_selector[i]['model_name'].split('=')[1]}\n
                    cfg = {model_selector[i]['model_name'].split('=')[2]}
                    '''
                st.success(info)

    toggle_view = [None]
    if st.session_state.model_counter >= 1:
        col_layout = st.columns(st.session_state.model_counter)
        with st.container():
            for i in range(st.session_state.model_counter):
                with col_layout[i]:
                    st.text(f"Model #{i+1} : {model_selector[i]['model_name'].split('=')[1]}")
                    subject = mydict['sampling_results'][model_selector[i]['sampling_folder']][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']]
                    src = st.selectbox(label="Select Source sample : ", options=sorted(subject.keys()), key=f"{model_selector[i]['model_name'].split('=')[1]}")
                    dst = st.selectbox(label="Select Destination sample : ", options=sorted(subject[src].keys()), key=f"{model_selector[i]['model_name'].split('=')[1]}")
                    frames = get_frames(subject[src][dst])
                    with st.expander(label=f"{src}, {dst}"):
                        np_video(frames)
                        if (st.checkbox(label=f'toggle view #{i+1}', key=f'toggle view #{i+1}')):
                            grid_image = torchvision.utils.make_grid(th.tensor(frames).permute(0, 3, 1, 2), nrow=6).numpy()
                            grid_image = np.transpose(grid_image, axes=(1, 2, 0))
                            st.image(image=grid_image, width=None)#, use_column_width=True)
                        else:
                            for f in frames:
                                st.image(image=f, width=None, use_column_width=True)