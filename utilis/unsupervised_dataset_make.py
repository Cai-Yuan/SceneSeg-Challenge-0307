from utilis import  read_pkl
import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import cmath
import torch
from mmcv import Config
from src import get_data
from torch.utils.data import DataLoader
from difference_of_feature_similarity import *






def unsupervised_train_set_make():
    """1,只用四个点，2，每次embedding 也用十个点"""
    config_path = r"..\config\all.py"
    cfg = Config.fromfile(config_path)
    data_root = r'..\data'
    imdbidlist = os.listdir(data_root)
    new_col=['place_dis','cast_dis','action_dis','audio_dis','label']





    for imdbid in imdbidlist:

        df=pd.DataFrame(columns=new_col)

        file_path = os.path.join(data_root, imdbid)
        imdbid_data = read_pkl(file_path)

        place_feature = np.array(imdbid_data['place'])
        cast_feature = np.array(imdbid_data['cast'])
        action_feature = np.array(imdbid_data['action'])
        audio_feature = np.array(imdbid_data['audio'])
        ground_truth_label= np.array(imdbid_data['scene_transition_boundary_ground_truth'].int())

        place_ump_feature = umap_embedding(place_feature)
        cast_ump_feature = umap_embedding(cast_feature)
        action_ump_feature = umap_embedding(action_feature)
        audio_ump_feature = umap_embedding(audio_feature)

        for i in range(1, len(place_feature)-2):


            ps = relative_distance_4  (place_ump_feature[i - 1:i + 3])
            cs = relative_distance_4  (cast_ump_feature[i - 1:i + 3])
            acts = relative_distance_4(action_ump_feature[i - 1:i + 3])
            auds = relative_distance_4(audio_ump_feature[i - 1:i + 3])

            df.loc[i] = {'place_dis': ps,'cast_dis': cs, 'action_dis' : acts, 'audio_dis' : auds,'label':ground_truth_label[i]}
        saving_name = os.path.join(cfg.csv_save_root, imdbid.split('.')[0] + '.xlsx')
        df.to_excel(saving_name, header = True, index=False)


def train_val_test():
    config_path = r"..\config\all.py"
    cfg = Config.fromfile(config_path)

    data_root = r'..\data'
    imdbidlist = os.listdir(data_root)
    df = pd.read_excel(os.path.join(cfg.csv_save_root, imdbidlist[0].split('.')[0] + '.xlsx'),index_col=False)

    for i, imdbid in enumerate(imdbidlist):
        saving_name = os.path.join(cfg.csv_save_root, imdbid.split('.')[0] + '.xlsx')

        if i<52 and i!=0:
            temp_df=pd.read_excel(saving_name)
            df=pd.concat([df, temp_df], axis=0)

        elif i>52:
            break
    df.index = range(len(df))
    df.to_excel(os.path.join(cfg.csv_save_root, 'unsupervised_train' + '.xlsx'), header = True, index=False)
    print(1)












if __name__=='__main__':
    #unsupervised_train_set_make()
    train_val_test()




