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
from torchvision import datasets, transforms
from lgss.utilis import read_json, read_pkl, read_txt_list, strcal
from lgss.utilis.package import *


normalizer = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalizer])


class Preprocessor(data.Dataset):
    def __init__(self, cfg, listIDs):
        self.cfg = cfg
        self.shot_num = cfg.shot_num #4
        self.listIDs = listIDs #视频名， 镜头id
        self.shot_boundary_range = range(-cfg.shot_num//2+1, cfg.shot_num//2+1)  #-1到3
        self.mode = cfg.dataset.mode  #['image']
        self.transform = transformer
        assert(len(self.mode) > 0)
    
    def __len__(self):
        return len(self.listIDs)
    
    def __getitem__(self, index):
        ID_list = self.listIDs[index]   #[{'imdbid': 'demo',  'shotid': '0202'}...] 有是个，可能是序列的长度吧
        if isinstance(ID_list, (tuple, list)):
            imgs, cast_feats, act_feats, aud_feats, labels = [], [], [], [], []
            for ID in ID_list:
                img, label = self._get_single_item(ID)
                imgs.append(img)
                labels.append(label)
            if 'image' in self.mode:
                imgs = torch.stack(imgs)
            labels = np.array(labels)
            return imgs, cast_feats, act_feats, aud_feats, labels
        else:
            return self._get_single_item(ID_list)

    def _get_single_item(self, ID):
        shotid = ID["shotid"]   #{'imdbid': 'demo',  'shotid': '0202'}
        imgs = []
        label = 1  # this is a pesudo label
        if 'image' in self.mode:
            for ind in self.shot_boundary_range:  #-1到3
                name = 'shot_{}_img_1.jpg'.format(strcal(shotid, ind))   #shot_0202_img_1.jpg,保证必须是有四位
                path = os.path.join(
                    self.cfg.data_root, 'shot_keyf', self.cfg.video_name, name)
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)

        if len(imgs) > 0:
            imgs = torch.stack(imgs)
        return imgs, label


def data_partition(cfg, valid_shotids):
    assert(cfg.seq_len % 2 == 0)
    seq_len_half = cfg.seq_len//2

    idxs = []
    one_mode_idxs = []
    shotid_tmp = 0
    for shotid in valid_shotids:
        if int(shotid) < shotid_tmp+seq_len_half:
            continue
        shotid_tmp = int(shotid)+seq_len_half
        one_idxs = []
        for idx in range(-seq_len_half+1, seq_len_half+1):
            one_idxs.append({'imdbid': cfg.video_name, 'shotid': strcal(shotid, idx)})
        one_mode_idxs.append(one_idxs)
    idxs.append(one_mode_idxs)

    partition = {}
    partition['train'] = idxs[0]
    partition['test'] = idxs[0]
    partition['val'] = idxs[0] #因该是每一个视频里面都被划分为三个部分
    return partition


def data_pre(cfg):
    data_root = cfg.data_root  #'../data/demo'
    #img_dir_fn = osp.join(data_root, 'shot_keyf', cfg.video_name)
    img_dir_fn=r"C:\Users\yuanc\Desktop\cyuan research\sence segmentation\SceneSeg-master\data\demo\shot_keyf\demo"
    win_len = cfg.seq_len+cfg.shot_num  #

    files = os.listdir(img_dir_fn)
    shotids = [int(x.split(".jpg")[0].split("_")[1]) for x in files if x.split(".jpg")[0][-1] == "1"]
    to_be_del = []
    for shotid in shotids:
        del_flag = False
        for idx in range(-(win_len)//2+1, win_len//2+1):  #-6~8
            if ((shotid + idx) not in shotids):  #0 + -6
                del_flag = True
                break
        if del_flag:
            to_be_del.append(shotid)

    valid_shotids = []
    for shotid in shotids:
        if shotid in to_be_del:
            continue
        else:
            valid_shotids.append(shotid)   #为什么要去掉两端的数据
    return valid_shotids


def main():
    from mmcv import Config
    cfg = Config.fromfile(r"C:/Users/yuanc/Desktop/cyuan research/sence segmentation/SceneSeg-master/lgss/config/demo.py")

    valid_shotids = data_pre(cfg)
    partition = data_partition(cfg, valid_shotids)
    batch_size = cfg.batch_size
    testSet = Preprocessor(cfg, partition["test"])
    test_loader = DataLoader(
                testSet, batch_size=batch_size,
                shuffle=False, **cfg.data_loader_kwargs)

    dataloader = test_loader
    for batch_idx, (data_place, data_cast, data_act, data_aud, target) in enumerate(dataloader):
        print(data_place.shape)  # bs, seq_len, shot_num, 3, 224, 224
        print(batch_idx, target.shape)
        # if batch_idx > 1:
        #     break
        # pdb.set_trace()
    pdb.set_trace()   #_trace(),就可以设置


if __name__ == '__main__':
    main()
