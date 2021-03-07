from __future__ import print_function
from mmcv import Config
from tensorboardX import SummaryWriter
import src.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src import get_data
from torch.utils.data import DataLoader
from utilis import (cal_MIOU, cal_Recall, cal_Recall_time, get_ap, get_mAP_seq,
                    load_checkpoint, mkdir_ifmiss, pred2scene, save_checkpoint,
                    save_pred_seq, scene2video, to_numpy, write_json)
from utilis.package import *
#from utilis.difference_of_feature_similarity import unsupervised_clustering
from utilis.unsupervised_classification import *

final_dict = {}
train_iter, test_iter, val_iter = 0, 0, 0



def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('config', help='config file path',default=r'.\config\all.py')
    args = parser.parse_args()
    return args


def train(cfg, model, train_loader, optimizer, scheduler, epoch, criterion, writer):
    global train_iter

    model.train()
    for batch_idx, (data_place, data_cast, data_act, data_aud, target, IDs) in enumerate(train_loader):
        data_place = data_place.cuda() if 'place' in cfg.dataset.mode or 'image' in cfg.dataset.mode else []
        data_cast  = data_cast.cuda()  if 'cast'  in cfg.dataset.mode else []
        data_act   = data_act.cuda()   if 'act'   in cfg.dataset.mode else []
        data_aud   = data_aud.cuda()   if 'aud'   in cfg.dataset.mode else []  #[8,10,4,512]
        target = target.view(-1).cuda()

        optimizer.zero_grad()
        output = model(data_place, data_cast, data_act, data_aud)  #这个就是输出
        output = output.view(-1, 2)  #80，2
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        if loss.item() > 0:
            train_iter += 1
            writer.add_scalar('train/loss', loss.item(), train_iter)
        if batch_idx % cfg.logger.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, int(batch_idx * len(data_place)), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()




def test(cfg, model, test_loader, criterion, writer,unsup_model, mode='test'):
    global test_iter, val_iter, final_dict
    model.eval()  #在model(test_datasets)之前，需要加上model.eval(). 否则的话，有输入数据，即使不训练，它也会改变权值
    test_loss = 0
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    prob_raw, gts_raw = [], []
    preds, gts = [], []
    batch_num = 0
    with torch.no_grad():
        for data_place, data_cast, data_act, data_aud, target, IDs in test_loader:
            batch_num += 1
            data_place = data_place.cuda() if 'place' in cfg.dataset.mode or 'image' in cfg.dataset.mode else []
            data_cast  = data_cast.cuda()  if 'cast'  in cfg.dataset.mode else []
            data_act   = data_act.cuda()   if 'act'   in cfg.dataset.mode else []
            data_aud   = data_aud.cuda()   if 'aud'   in cfg.dataset.mode else []
            target = target.view(-1).cuda()  #因此就会将target里面的所有维度数据转化成一维的

            output = model(data_place, data_cast, data_act, data_aud)
            output = output.view(-1, 2)   #为什么要分成两列  80 2

            loss = criterion(output, target)

            unsup_pres = unsupervised_pre(cfg, unsup_model, IDs)
            if unsup_pres is not None:

                length = output.shape[0]
                for i in range(0, length):
                    output[i, 0] = 0.4 * (torch.abs(output[i, 0]) / (torch.abs(output[i, 0]) + torch.abs(output[i, 1]))) + 0.6 * unsup_pres[i, 0]
                    output[i, 1] = 1 - output[i, 0]

            if mode == 'test':
                test_iter += 1
                if loss.item() > 0:
                    writer.add_scalar('test/loss', loss.item(), test_iter)
            elif mode == 'val':
                val_iter += 1
                if loss.item() > 0:
                    writer.add_scalar('val/loss', loss.item(), val_iter)

            test_loss += loss.item()
            output = F.softmax(output, dim=1)


            prob = output[:, 1]
            gts_raw.append(to_numpy(target))
            prob_raw.append(to_numpy(prob))

            gt = target.cpu().detach().numpy()
            prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
            idx1 = np.where(gt == 1)[0]
            idx0 = np.where(gt == 0)[0]
            gt1 += len(idx1)
            gt0 += len(idx0)
            all_gt += len(gt)
            correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
            correct0 += len(np.where(gt[idx0] == prediction[idx0])[0])
        for x in gts_raw:
            gts.extend(x.tolist())
        for x in prob_raw:
            preds.extend(x.tolist())

    test_loss /= batch_num
    ap = get_ap(gts_raw, prob_raw)
    mAP, mAP_list = get_mAP_seq(test_loader, gts_raw, prob_raw)
    print("AP: {:.3f}".format(ap))
    print('mAP: {:.3f}'.format(mAP))
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct1 + correct0, all_gt,
        100. * (correct0 + correct1)/all_gt))
    print('Accuracy1: {}/{} ({:.0f}%), Accuracy0: {}/{} ({:.0f}%)'.format(
        correct1, gt1, 100.*correct1/(gt1+1e-5), correct0, gt0,
        100.*correct0/(gt0+1e-5)))
    if mode == "val" or mode == "test":
        return mAP.mean()
    elif mode == "test_final":
        final_dict.update({
            "AP":  ap,
            "mAP": mAP,
            "Accuracy":  100 * (correct0 + correct1)/all_gt,
            "Accuracy1": 100 * correct1/(gt1+1e-5),
            "Accuracy0": 100 * correct0/(gt0+1e-5),
            })
        return gts, preds



def main():
    config_path = r".\config\all.py"
    cfg = Config.fromfile(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    if cfg.trainFlag:  # copy running config to run files
        writer = SummaryWriter(logdir=cfg.logger.logs_dir)
        shutil.copy2(config_path, cfg.logger.logs_dir)  # 功能：复制文件 格式：shutil.copy('来源文件','目标地址') 返回值：复制之后的路径
        print(1)

    unsup_model=unsupervised_model()

    trainSet, testSet, valSet = get_data(cfg) #这就包含里面的所有特征
    train_loader = DataLoader(
                    trainSet, batch_size=cfg.batch_size,
                    shuffle=False, **cfg.data_loader_kwargs)
    test_loader = DataLoader(
                    testSet, batch_size=cfg.batch_size,
                    shuffle=False, **cfg.data_loader_kwargs)
    val_loader = DataLoader(
                    valSet, batch_size=cfg.batch_size,
                    shuffle=True, **cfg.data_loader_kwargs)

    model = models.__dict__[cfg.model.name](cfg).cuda()
    model = nn.DataParallel(model)  #nn.DataParallel函数来用多个GPU来加速训练
    if cfg.resume is not None:
        checkpoint = load_checkpoint(cfg.resume)  #是不是继续训练
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.__dict__[cfg.optim.name](
        model.parameters(), **cfg.optim.setting)
    scheduler = optim.lr_scheduler.__dict__[cfg.stepper.name](
        optimizer, **cfg.stepper.setting)
    criterion = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight).cuda())
    print("...data and model loaded")

    if cfg.trainFlag:
        print("...begin training")
        max_ap = -1
        for epoch in range(1, cfg.epochs + 1):
            train(cfg, model, train_loader, optimizer, scheduler, epoch, criterion, writer)
            print("Val Acc")
            ap = test(cfg, model, val_loader, criterion, writer, unsup_model, mode='val')
            print("Test Acc")
            test(cfg, model, test_loader, criterion, writer, unsup_model, mode='test')
            if ap > max_ap:
                is_best = True
                max_ap = ap
            else:
                is_best = False
            save_checkpoint({
                'state_dict': model.state_dict(), 'epoch': epoch + 1,
            }, is_best=is_best, fpath=osp.join(cfg.logger.logs_dir, 'checkpoint.pth.tar'))

    if cfg.testFlag:
        print('...test with saved model')
        checkpoint = load_checkpoint(
            osp.join(cfg.logger.logs_dir, 'model_best.pth.tar'))  #最佳模型的地址
        model.load_state_dict(checkpoint['state_dict']) #导入最佳模型的参数
        gts, preds = test(cfg, model, test_loader, criterion, mode='test_final')
        save_pred_seq(cfg, test_loader, gts, preds)
        if cfg.shot_frm_path is not None:
            Miou = cal_MIOU(cfg, threshold=0.5)
            Recall = cal_Recall(cfg, threshold=0.5)
            Recall_time = cal_Recall_time(cfg, recall_time=3, threshold=0.5)
            final_dict.update(
                {"Miou": Miou, "Recall": Recall, "Recall_time": Recall_time})
        else:
            print('...there is no correspondence file between shots and their frames')
        log_dict = {'cfg': cfg.__dict__['_cfg_dict'], 'final': final_dict}
        write_json(osp.join(cfg.logger.logs_dir, "log.json"), log_dict)

        if cfg.dataset.name == "demo":
            print('...visualize scene video in demo mode, the above quantitive metrics are invalid')
            scene_dict, scene_list = pred2scene(cfg, threshold=0.8)
            scene2video(cfg, scene_list)


if __name__ == '__main__':
    main()
