from __future__ import print_function

import random
import sys
from multiprocessing import Manager, Pool, Process
sys.path.append(".")
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import *
#from torchvision import datasets, transforms
from utilis import read_json, read_pkl, read_txt_list, strcal
#from utilis.package import *
from src.models import LGSS
from src.models import LGSS_image
import numpy as np





class Preprocessor(data.Dataset):
    """类，传入一个id list，返回这个list里面所有的特征"""
    def __init__(self, cfg, listIDs, data_dict):  #listIDs就是训练，或者测试，或者验证的视频id
        self.shot_num = cfg.shot_num  #4
        self.data_root = cfg.data_root  #'../data'
        self.listIDs = listIDs   #训练集所有的视频id, 以及后面是个序列的子id [{'imdbid': 'tt1375666.pkl', 'shotid': '0001'},....10ge
        self.data_dict = data_dict  #包含三个数据，标注，地点，人物，和动作，声音
        self.shot_boundary_range = range(-cfg.shot_num//2+1,cfg.shot_num//2+1)  #-1，3
        self.mode = cfg.dataset.mode  #是有几类特征，这里是4个都用
        assert(len(self.mode) > 0)
    
    def __len__(self):
        return len(self.listIDs)

    def __getitem__(self, index):
        """就是传入一个list，然后返回每一个对应的4个特征和标签，有可能是返回一个片段的数据"""
        ID_list = self.listIDs[index]  #[{'imdbid': 'tt1375666.pkl', 'shotid': '0001'},....
        if isinstance(ID_list, (tuple, list)):  # 视频id ， 镜头id序列
            place_feats, cast_feats, act_feats, aud_feats, labels = [], [], [], [], []
            i=0

            for ID in ID_list:  #这里面应该有十个[{},...10,...{}],id 就一个={'imdbid': 'tt1375666.pkl', 'shotid': '0001'}
                place_feat, cast_feat, act_feat, aud_feat, label = self._get_single_item(ID) #这里面每一个含有四个特征，-1 0 1 2
                #print(i,':', ID['imdbid'],'shotid:',ID['shotid'] )
                i=i+1
                place_feats.append(place_feat)
                cast_feats.append(cast_feat)
                act_feats.append(act_feat)
                aud_feats.append(aud_feat)
                labels.append(label)

            if 'place' in self.mode:
                place_feats = torch.stack(place_feats)  #把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量 这就变成了 10，4，2048
            if 'cast' in self.mode:
                cast_feats = torch.stack(cast_feats)
            if 'act' in self.mode:
                act_feats = torch.stack(act_feats)
            if 'aud' in self.mode:
                aud_feats = torch.stack(aud_feats)
            labels = torch.tensor(np.array(labels),dtype=torch.long)  #torch.tensor(已有数据，dtype=torch.float)

            return place_feats, cast_feats, act_feats, aud_feats, labels, ID_list
        else:
            return self._get_single_item(ID_list)

    def _get_single_item(self, ID):  #{'imdbid': 'tt1375666.pkl', 'shotid': '0001'}
        """输入必须包括视频id和镜头id，返回某视频，一个shot的数据"""
        imdbid = ID['imdbid']  ##{'imdbid': 'tt1375666.pkl', 'shotid': '0000'}
        shotid = ID['shotid']  #0000


        if self.data_dict["annos_dict"].get(imdbid)[int(shotid)] == False:
            label=0
        else:
            label=1
        #print('shotid',shotid, ':',label)
        #label = self.data_dict["annos_dict"].get(imdbid).get(shotid) #首先是标注，后面是视频，在后来是片段
        place_feats,  cast_feats, act_feats, aud_feats  = [], [], [], []


        if 'place' in self.mode:
            for ind in self.shot_boundary_range: #-1,0,1,2   place_feat=data_dict['places_dict']['tt1375666.pkl'][0]
                place_feat=self.data_dict['places_dict'][imdbid][int(shotid)+ind]
                #place_feats.append(torch.from_numpy(place_feat).float())
                place_feats.append(place_feat.float())

        if 'cast' in self.mode:
            for ind in self.shot_boundary_range: #-1,0,1,2

                cast_feat = self.data_dict["casts_dict"].get(imdbid)[int(shotid)+ind]
                #cast_feat = np.mean(cast_feat_raw, axis=0)
                cast_feats.append((cast_feat).float())

        if 'act' in self.mode:
            for ind in self.shot_boundary_range:

                act_feat = self.data_dict["casts_dict"].get(imdbid)[int(shotid)+ind]
                act_feats.append((cast_feat).float())

        if 'aud' in self.mode:
            for ind in self.shot_boundary_range:

                aud_feat=self.data_dict['auds_dict'].get(imdbid)[int(shotid)+ind]
                aud_feats.append(aud_feat.float())


        if len(place_feats) > 0:
            place_feats = torch.stack(place_feats)  #浅显说法：把多个2维的张量凑成一个3维的张量
        if len(cast_feats) > 0:
            cast_feats = torch.stack(cast_feats)
        if len(act_feats) > 0:
            act_feats = torch.stack(act_feats)
        if len(aud_feats) > 0:
            aud_feats = torch.stack(aud_feats)
        return place_feats, cast_feats, act_feats, aud_feats, label


def data_partition(cfg, imdbidlist_json, annos_dict):
    """就是数据集的划分，返回的是以train，valisade，test为key的字典，value是视频id，镜头id。现在知道为什么有两个ID"""
    assert(cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len//2  #5

    idxs = []
    for mode in ['train', 'test', 'val']:
        one_mode_idxs = []

        for imdbid in imdbidlist_json[mode]:  # imdbid因该是 图像数据库ID  imdbidlist_json这个文件就包含了训练，测试和验证
            anno_dict = annos_dict[imdbid]   #标注字典，每一个视频的 imd：{镜头id：标注值
            shotid_list=list(range(len(anno_dict)-7))  #一共有多少个镜头2609 【0，2609】

            #shotid_list = sorted(anno_dict.keys())  # #一个序列，所有的镜头id。  {视频id：【镜头id：标注值]}
            shotid_tmp = 0
            for shotid in shotid_list: #2609个
                if (int(shotid) < shotid_tmp+seq_len_half): #不考虑前面5个镜头，也不考虑后面5个镜头  shotid_tmp+5 =[5,15,
                    continue
                shotid_tmp = int(shotid)+seq_len_half  #  5+5
                one_idxs = []

                for idx in range(-seq_len_half, seq_len_half):  #-5，4  所以有十个序列
                    one_idxs.append({'imdbid':imdbid, 'shotid': strcal(shotid, idx+1)})  # 前面是图像序列id，后面是一个序列的所有镜头所组成的id
                    #[0,...9]， 【6，15】，分成好多个这样的窗口 10 个一组

                one_mode_idxs.append(one_idxs)

        idxs.append(one_mode_idxs)

    partition = {}
    partition['train'] = idxs[0]
    partition['test'] = idxs[1]
    partition['val'] = idxs[2]  #{train：【视频id：，10个镜头】
    return partition


def data_pre_one(cfg,  imdbid, places_dict_raw,casts_dict_raw, acts_dict_raw,  auds_dict_raw, annos_dict_raw, annos_valid_dict_raw):
    """一个数据准备，就准备某一个视频"""
    data_root = cfg.data_root  #'./data'
    #label_fn = osp.join(data_root,'label318') #标记的路径
    #place_feat_fn = osp.join(data_root, 'place_feat') #位置的特征路径
    win_len = cfg.seq_len+cfg.shot_num # 好像是14 窗口的大小
    file_path=os.path.join(data_root,imdbid)
    imdbid_file=read_pkl(file_path)

    places_dict_raw.update({imdbid:imdbid_file['place']})
    casts_dict_raw.update({imdbid: imdbid_file['cast']})
    acts_dict_raw.update({imdbid: imdbid_file['action']})
    auds_dict_raw.update({imdbid: imdbid_file['audio']})
    annos_valid_dict_raw.update({imdbid: imdbid_file['scene_transition_boundary_ground_truth']})




def data_pre(cfg):
    """imdbidlist_json:所有视频的数据，后面是以视频id为key的字典"""

    data_root = cfg.data_root  #'../data'
    imdbidlist = os.listdir(data_root)     #cy



    imdbidlist_json, places_dict, acts_dict, casts_dict, auds_dict, annos_dict,annos_valid_dict= {}, {}, {},{},{}, {}, {}
    imdbidlist_json.update({"all":imdbidlist})
    imdbidlist_json.update({'train':imdbidlist[0:52]})  #['train', 'test', 'val']
    imdbidlist_json.update({'val':imdbidlist[52:58]})
    imdbidlist_json.update({'test': imdbidlist[58:64]})

    for imdbid in imdbidlist:
        file_path = os.path.join(data_root, imdbid)
        imdbid_file = read_pkl(file_path)
        places_dict.update({imdbid: imdbid_file['place']})
        casts_dict.update({imdbid: imdbid_file['cast']})
        acts_dict.update({imdbid: imdbid_file['action']})
        auds_dict.update({imdbid: imdbid_file['audio']})
        annos_valid_dict.update({imdbid: imdbid_file['scene_transition_boundary_ground_truth']})


    return imdbidlist_json, places_dict,acts_dict, casts_dict, auds_dict,annos_valid_dict

#没用
def get_anno_dict(anno_fn):
    """输入一个视频的地址，返回一个字典，每一个镜头id：标注"""
    contents = read_txt_list(anno_fn)  #读取标注文本
    anno_dict = {}
    for content in contents:
        shotid = content.split(' ')[0]
        value = int(content.split(' ')[1])  # 应该是，镜头id：值。这里的值也是0和1
        if value >= 0:
            anno_dict.update({shotid: value})
        elif value == -1:
            anno_dict.update({shotid: 1})  #为什么用字典呢？？
    return anno_dict


def main():
    from mmcv import Config
    cfg = Config.fromfile(r"C:\Users\yuanc\Desktop\cyuan research\sence segmentation\SceneSeg-master\lgss\config\all.py")
    imdbidlist_json, places_dict,acts_dict, casts_dict, auds_dict,annos_valid_dict = data_pre(cfg) #数据集的分类
    partition = data_partition(cfg, imdbidlist_json, annos_valid_dict)  #数据集的窗口细分

    data_dict = {"annos_dict": annos_valid_dict,
                 "places_dict": places_dict,
                 "casts_dict": casts_dict,
                 "acts_dict": acts_dict,
                 "auds_dict": auds_dict} # 视频id为key的字典
    if 0:
        place_feats, cast_feats, act_feats, aud_feats, labels = [], [], [], [], []
        listIDs = partition['train']
        for index, _ in enumerate(listIDs):
            ID_list = listIDs[index]
            for ID in ID_list:
                if _get_single_item(ID, cfg, data_dict) is None:
                    continue
                place_feat, cast_feat, act_feat, aud_feat, label = _get_single_item(ID, cfg, data_dict)
                place_feats.append(place_feat)
                cast_feats.append(cast_feat)
                act_feats.append(act_feat)
                aud_feats.append(aud_feat)
                labels.append(label)
        pdb.set_trace()

    batch_size = cfg.batch_size
    testSet = Preprocessor(cfg, partition['test'], data_dict)
    test_loader = DataLoader(testSet, batch_size=batch_size, \
                shuffle=False, **cfg.data_loader_kwargs)
    model = LGSS(cfg)

    dataloader = test_loader
    for batch_idx, (data_place,data_cast,data_act,data_aud,target) in enumerate(dataloader):
        data_place = data_place.cuda()if 'place'in cfg.dataset.mode else []
        data_cast  = data_cast.cuda() if 'cast' in cfg.dataset.mode else []
        data_act   = data_act.cuda()  if 'act' in cfg.dataset.mode else [] 
        data_aud   = data_aud.cuda()  if 'aud' in cfg.dataset.mode else []
        # print (data_cast.shape)
        # print (data_cast.shape,data_act.shape)
        print (data_place.shape,data_cast.shape,data_act.shape,data_aud.shape,target.shape)
        print (batch_idx,target.shape)
        # if i_batch > 1:
        #     break
        # pdb.set_trace()
    pdb.set_trace()


if __name__ == '__main__':
    main()
