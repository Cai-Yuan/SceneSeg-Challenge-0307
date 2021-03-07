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


def umap_embedding(data,label=None, plot=False):

    featureMatrix=data
    umapreducer = umap.UMAP(random_state=42)
    embeddingnew = umapreducer.fit_transform(featureMatrix)

    return embeddingnew

def distance_of_two_point(a, b):

    temp = cmath.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return np.real(temp)

def plot_features_dis(data,saving_name=None):

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(9, 9),dpi=100)
    #plt.tick_params(labelsize=2)
    title=['place feature difference','cast feature difference','action feature difference','audio feature difference','scene_transition_boundary_ground_truth']
    ground_truth= [i * 10 for i in data[4]]
    for i in range(4):
        axes[i].plot(data[i])
        axes[i].set_title(title[i], fontsize=12, pad=2)
        axes[i].plot(ground_truth, color = '#FF00EE', alpha = 0.7)
        if i<3:
            axes[i].set_ylim(0,50)
        elif i==3:
            axes[i].set_ylim(0, 20)

    plt.subplots_adjust(left=0.05, bottom=0.04, right=0.96, top=0.96, wspace=0, hspace=0.4)
    plt.savefig(saving_name, dpi=100)
    # plt.show()
    return None



def relative_distance_7(value):
    #输入必须是7个值

    leng=len(value)
    target=leng//2

    point1_x = np.average(value[0:target,  0])
    point1_y = np.average(value[0:target,  1])
    point2_x = np.average(value[target+1:, 0])
    point2_y = np.average(value[target+1:, 1])
    dis3=distance_of_two_point([point1_x, point1_y], [point2_x, point2_y])

    dis1=0
    dis2=0

    for i in range(leng):
        temp= distance_of_two_point(value[i],value[target])
        if i<target:
            dis1=dis1+( 1/abs(target-i))*temp
        elif i>target:
            dis2=dis2+( 1/abs(target-i))*temp

    dis_max=[dis1 * 6 / 11, dis2 * 6 / 11, dis3]

    return dis_max[1]*dis_max[2]/(dis_max[0]*dis_max[0])


def relative_distance_4(value):
    #输入必须是4个值
    leng=4
    target=1

    point1_x = np.average(value[0:target,  0])
    point1_y = np.average(value[0:target,  1])
    point2_x = np.average(value[target+1:, 0])
    point2_y = np.average(value[target+1:, 1])
    dis3=distance_of_two_point([point1_x, point1_y], [point2_x, point2_y])

    dis1=0
    dis2=0

    for i in range(leng):
        temp= distance_of_two_point(value[i],value[target])
        if i<target:
            dis1=dis1+( 1/abs(target-i))*temp
        elif i>target:
            dis2=dis2+( 1/abs(target-i))*temp

    dis_max=[dis1 , dis2 * 2 / 3, dis3]

    if dis_max[1]*dis_max[2]/(dis_max[0]*dis_max[0])>30:
        return 30
    else:
        return dis_max[1]*dis_max[2]/(dis_max[0]*dis_max[0])







def similarity_of_features():
    data_root = r'..\data'
    imdbidlist = os.listdir(data_root)
    new_col=['place_dis','cast_dis','action_dis','audio_dis','label']





    for imdbid in imdbidlist:

        file_path = os.path.join(data_root, imdbid)
        imdbid_data = read_pkl(file_path)

        place_feature = np.array(imdbid_data['place'])
        cast_feature = np.array(imdbid_data['cast'])
        action_feature = np.array(imdbid_data['action'])
        audio_feature = np.array(imdbid_data['audio'])
        ground_truth_label= np.array(imdbid_data['scene_transition_boundary_ground_truth'].int())


        place_ump_feature= umap_embedding(place_feature)
        cast_ump_feature = umap_embedding(cast_feature)
        action_ump_feature = umap_embedding(action_feature)
        audio_ump_feature = umap_embedding(audio_feature)
        feature_dis,place_dis, cast_dis, action_dis, audio_dis = [], [0,0,0], [0,0,0], [0,0,0], [0,0,0]

        for i in range(3, len(place_ump_feature) - 4):
            end1 = 3

            place_dis.append(relative_distance(place_ump_feature[i - end1:i + end1 + 1, :]))
            cast_dis.append(relative_distance(cast_ump_feature[i - end1:i + end1 + 1, :]))
            action_dis.append(relative_distance(action_ump_feature[i - end1:i + end1 + 1, :]))
            audio_dis.append(relative_distance(audio_ump_feature[i - end1:i + end1 + 1, :]))

        feature_dis.append(place_dis)
        feature_dis.append(cast_dis)
        feature_dis.append(action_dis)
        feature_dis.append(audio_dis)
        feature_dis.append(ground_truth_label[3:-4])


        #plot_features_dis(feature_dis, saving_name=os.path.join(r"..\image",imdbid.split('.')[0]+'.tif' ))

        print(1)

def make_training_set(data_place, data_cast, data_act, data_aud, target):
    place_feature =  data_place.reshape(80,-1)
    cast_feature =   data_cast.reshape(80,-1)
    action_feature = data_act.reshape(80,-1)
    audio_feature =  data_aud.reshape(80,-1)

    place_ump_feature = umap_embedding(place_feature)
    cast_ump_feature = umap_embedding(cast_feature)
    action_ump_feature = umap_embedding(action_feature)
    audio_ump_feature = umap_embedding(audio_feature)

    for i in range(3, len(place_ump_feature) - 4):
        end1 = 3

        place_dis.append(relative_distance(place_ump_feature[i - end1:i + end1 + 1, :]))
        cast_dis.append(relative_distance(cast_ump_feature[i - end1:i + end1 + 1, :]))
        action_dis.append(relative_distance(action_ump_feature[i - end1:i + end1 + 1, :]))
        audio_dis.append(relative_distance(audio_ump_feature[i - end1:i + end1 + 1, :]))




    print(1)






if __name__=='__main__':

    config_path = r"..\config\all.py"
    cfg = Config.fromfile(config_path)


    trainSet, testSet, valSet = get_data(cfg)  # 这就包含里面的所有特征
    train_loader = DataLoader(
        trainSet, batch_size=cfg.batch_size,
        shuffle=False, **cfg.data_loader_kwargs)
    test_loader = DataLoader(
        testSet, batch_size=cfg.batch_size,
        shuffle=False, **cfg.data_loader_kwargs)
    val_loader = DataLoader(
        valSet, batch_size=cfg.batch_size,
        shuffle=True, **cfg.data_loader_kwargs)

    for batch_idx, (data_place, data_cast, data_act, data_aud, target, IDs) in enumerate(train_loader):
        data_place = np.array(data_place)


        data_cast  = np.array(data_cast)
        data_act   = np.array(data_act)
        data_aud   = np.array(data_aud)
        target =   target.view(-1).cuda()

        a=make_training_set(data_place[:,:,1,:], data_cast[:,:,1,:],
                            data_act[:,:,1,:], data_aud[:,:,1,:], target)


    #similarity_of_features()
    '''
    #unsupervised_clustering()
    #similarity_of_features()
    '''













