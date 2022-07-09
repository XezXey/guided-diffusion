from pickletools import optimize
import pprint
from tabnanny import check
from turtle import onclick
from click import option
import numpy as np
import streamlit as st


st.set_page_config(layout="wide")

import argparse
import glob, os
import json


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_folder', type=str)
args = parser.parse_args()

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

if __name__ == '__main__':
    sample_folder = os.listdir(f'{args.experiment_folder}/sampling_results/')
    mydict = path_to_dict(path=f'{args.experiment_folder}/sampling_results/', d = {})
    if 'model_counter' not in st.session_state:
        st.session_state['model_counter'] = 0
    if 'model_selector' not in st.session_state:
        st.session_state['model_selector'] = []
    # Folder picker button
    # st.title('DPM Sampling Visualizer')
    # selected = st.selectbox('Select a Sampling Folder?', sample_folder)
    # model_log = os.listdir(f'{args.experiment_folder}/sampling_results/{selected}')
    # model = {m : None for m in model_log}



    # with st.expander("Choosing model to compare"):
    col1, col2 = st.columns([.5,1])
    with col1:
        add_button = st.button("Add model")
    with col2:
        rem_button = st.button("Remove model")
    
    if add_button:
        st.session_state.model_counter += 1
        st.experimental_rerun()
    if rem_button:
        if st.session_state.model_counter > 1:
            st.session_state.model_counter -= 1
        st.experimental_rerun()

    print(mydict.keys())
    print(mydict['sampling_results'].keys())
    print(mydict['sampling_results']['ddim_reverse'].keys())

    model_selector = {}
    with st.container():
        for i in range(st.session_state.model_counter):
            model_selector[i] = {
                'model_name': st.selectbox(
                    label = f"Model #{i + 1}", 
                    options=mydict['sampling_results']['ddim_reverse'].keys()
                )
            }
            model_selector[i].update({
                'ckpt': st.selectbox(
                    label = f"Checkpoint #{i + 1}", 
                    options=mydict['sampling_results']['ddim_reverse'][model_selector[i]['model_name']].keys()
                )
            })
            model_selector[i].update({
                'dataset': st.selectbox(
                    label = f"Dataset #{i + 1}", 
                    options=mydict['sampling_results']['ddim_reverse'][model_selector[i]['model_name']][model_selector[i]['ckpt']].keys()
                )
            })
            model_selector[i].update({
                'cond': st.selectbox(
                    label = f"Condition #{i + 1}", 
                    options=mydict['sampling_results']['ddim_reverse'][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']].keys()
                )
            })

    col_layout = st.columns(st.session_state.model_counter)
    with st.container():
        for i in range(st.session_state.model_counter):
            with col_layout[i]:
                image = mydict['sampling_results']['ddim_reverse'][model_selector[i]['model_name']][model_selector[i]['ckpt']][model_selector[i]['dataset']][model_selector[i]['cond']]
                for j in range(len(image)):
                    st.image(image=image[j], caption=f"Image path = {image[j]}", use_column_width=True)

# first_level_choice =  st.sidebar.selectbox('', ('Cars', 'Food', 'Electronics'))
# b1 = st.sidebar.button('submit level 1 choice')

# if b1: 
#     if first_level_choice == 'Cars':
#               second_level_choice_car = st.sidebar.selectbox('', ('Honda', 'Opel', 'Tesla'))
#               b21 = st.sidebar.button('submit car choice')
#               if b21: 
#                   st.write('your car choice is ' + second_level_choice_car)
#     if first_level_choice == 'Food':
#               second_level_choice_food = st.sidebar.selectbox('', ("Egg", "Pizza", "Spinach"))
#               b22 = st.sidebar.button('submit food choice')
#               if b22: 
#                   st.write('your food choice is ' + second_level_choice_food)
#     if first_level_choice == 'Electronics':
#               second_level_choice_elec = st.sidebar.selectbox('', ("Headphones", "Laptop", "Phone"))
#               b23 = st.sidebar.button('submit elec choice')
#               if b23: 
#                   st.write('your electronics choice is ' + second_level_choice_elec)