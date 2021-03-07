import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class audnet(nn.Module):
    def __init__(self, cfg):
        super(audnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3))

        self.conv2 = nn.Conv2d(64, 192, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=(1,3))

        self.conv3 = nn.Conv2d(192, 384, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3,3), stride=(2,2), padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2,2), padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3,2), padding=0)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, 512)

    def forward(self, x):  # [bs,1,257,90]   输入声音的维度  ([8, 10, 4, 257, 90])
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = x.squeeze()   #到这一步：320 512
        out = self.fc(x) #这是一个全连接层 320 512
        return out

class BNet_aud(nn.Module):
    """处理声音的特征，不知到是提取特征还是处理特征"""
    def __init__(self, cfg):
        super(BNet_aud, self).__init__()
        self.shot_num = cfg.shot_num #4
        self.channel = cfg.model.sim_channel #512
        self.audnet = audnet(cfg)
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num, 1))
        self.conv2 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num//2, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))

    def forward(self, x):  # [batch_size, seq_len, shot_num, 257, 90]
        '''
        context = x.view(    #x:torch.Size([8, 10, 4, 257, 90])
            x.shape[0]*x.shape[1]*x.shape[2], 1, x.shape[-2], x.shape[-1]) #torch.Size([320, 1, 257, 90])

        context = self.audnet(context).view(
            x.shape[0]*x.shape[1], 1, self.shot_num, -1)  #torch.Size([80, 1, 4, 512])  说明声音都不需要处理了，已经处理好了
        '''
        context=x.view(x.shape[0]*x.shape[1], 1,self.shot_num,-1)

        part1, part2 = torch.split(context, [self.shot_num//2]*2, dim=2) #永远是分成两部分[80,1,4,512]->[80,1,2,512]
        part1 = self.conv2(part1).squeeze() #【80，1，2，512】-》80，2，512
        part2 = self.conv2(part2).squeeze()  #【80，1，2，512】-》80，2，512
        sim = F.cosine_similarity(part1, part2, dim=2)
        bound = sim  #sim:torch.Size(320,512)
        return bound



class Cos(nn.Module):
    """"[8,10,4,512]->[80,512]"""
    def __init__(self, cfg):
        super(Cos, self).__init__()
        self.shot_num = cfg.shot_num  #4
        self.channel = cfg.model.sim_channel  #512,   dim of similarity vector
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(self.shot_num//2, 1)) #输入通道是1，输出通道是512

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim] 8 10, 4 512
        x = x.view(-1, 1, x.shape[2], x.shape[3]) #80 1 4 512
        part1, part2 = torch.split(x, [self.shot_num//2]*2, dim=2)  #在维度2上划分，分成【2，2】 part1=01,part2=23
        # batch_size*seq_len, 1, [self.shot_num//2], feat_dim [80 1 2 512]

        part1 = self.conv1(part1).squeeze() #80，512, 1，512 ->
        part2 = self.conv1(part2).squeeze()
        x = F.cosine_similarity(part1, part2, dim=2)  # batch_size,channel   计算两个张良的相似度，按DIM=2计算
                                                      # torch.Size([80, 512])
        return x


class BNet(nn.Module):
    """处理地点，人物，动作的特征 [8,10,4,512]->[80,512]"""
    def __init__(self, cfg):
        super(BNet, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model.sim_channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))   #对输入执行最大的池化
        self.cos = Cos(cfg)

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        context = x.view(x.shape[0]*x.shape[1], 1, -1, x.shape[-1])  #变成80，1，4，2048
        context = self.conv1(context)  # batch_size*seq_len,512,1,feat_dim    [80 512 1 2048]
        context = self.max3d(context)  # batch_size*seq_len,1,1,feat_dim     [80, 1, 1, 2048])
        context = context.squeeze()  #  batch_size*seq_len,feat_dim  torch.Size([80, 2048])
        sim = self.cos(x) #torch.Size([8,10,4 512])->[80 512]
        bound = torch.cat((context, sim), dim=1)  #80, 2560#在列上拼接 80,2048 和80 512在y方向上拼接 80， 2048+512
        #y=7
        return bound  #[80,2048+512]





class LGSSone(nn.Module):
    """只分析一个特征的lstm 的前后关系"""
    def __init__(self, cfg, mode="place"):
        super(LGSSone, self).__init__()
        self.seq_len = cfg.seq_len  #10
        self.num_layers = 1
        self.lstm_hidden_size = cfg.model.lstm_hidden_size  #512
        if mode == "place":
            self.input_dim = (cfg.model.place_feat_dim+cfg.model.sim_channel)  #2048+512
            self.bnet = BNet(cfg)
        elif mode == "cast":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model.cast_feat_dim+cfg.model.sim_channel)  #512+512
        elif mode == "act":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model.act_feat_dim+cfg.model.sim_channel)  #512+512 特征+隐藏层
        elif mode == "aud":
            self.bnet = BNet_aud(cfg)
            self.input_dim = cfg.model.aud_feat_dim  #512
        else:
            pass
        self.lstm = nn.LSTM(input_size=self.input_dim,     #输入为什么都是特征加相似度？？？
                            hidden_size=self.lstm_hidden_size,   #这个就是我们需要的隐藏信息
                            num_layers=self.num_layers,
                            batch_first=True,  #batch_first：True则输入输出的数据格式为 (batch, seq, feature)
                            bidirectional=cfg.model.bidirectional)

        if cfg.model.bidirectional:
            self.fc1 = nn.Linear(self.lstm_hidden_size*2, 100) #是用于设置网络中的全连接层的，输出100
        else:
            self.fc1 = nn.Linear(self.lstm_hidden_size, 100)
        self.fc2 = nn.Linear(100, 2)  #最后判断是0或者1

    def forward(self, x):
        """[8,10,4,2048]-> 【80 2】"""
        x = self.bnet(x)  #【80，2048+512】 变成8 10 2048+512
        x = x.view(-1, self.seq_len, x.shape[-1])  #必须是这个格式    每次运行时取3个含有5个字的句子(且句子中每个字的维度为10进行运行)
        # torch.Size([128, seq_len, 3*channel])  [8,10,2560]
        self.lstm.flatten_parameters()
        out, (_, _) = self.lstm(x, None)
        # out: tensor of shape (batch_size, seq_length, hidden_size)  隐藏层就代表了  #[8, 10, 1024]) 10242为两个隐藏曾的长度
        out = F.relu(self.fc1(out))   #out=【8 10 100】
        out = self.fc2(out)   #8,10,2
        out = out.view(-1, 2)  #输出都是两列的，不管是什么特征 【80,2】
        return out


class LGSS(nn.Module):
    def __init__(self, cfg):
        super(LGSS, self).__init__()
        self.seq_len = cfg.seq_len  #10
        self.mode = cfg.dataset.mode #4类
        self.num_layers = 1
        self.lstm_hidden_size = cfg.model.lstm_hidden_size  #512
        self.ratio = cfg.model.ratio  #四个特征所占的比重 0.5 0.2 0.2 0.1

        if 'place' in self.mode:
            self.bnet_place = LGSSone(cfg, "place")
        if 'cast' in self.mode:
            self.bnet_cast = LGSSone(cfg, "cast")
        if 'act' in self.mode:
            self.bnet_act = LGSSone(cfg, "act")
        if 'aud' in self.mode:
            self.bnet_aud = LGSSone(cfg, "aud") #建立四个

    def forward(self, place_feat, cast_feat, act_feat, aud_feat):
        out = 0
        if 'place' in self.mode:
            place_bound = self.bnet_place(place_feat)
            out += self.ratio[0]*place_bound  #0.5*  torch.Size([80, 2])
        if 'cast' in self.mode:
            cast_bound = self.bnet_cast(cast_feat)
            out += self.ratio[1]*cast_bound
        if 'act' in self.mode:
            act_bound = self.bnet_act(act_feat)
            out += self.ratio[2]*act_bound
        if 'aud' in self.mode:
            aud_bound = self.bnet_aud(aud_feat)
            out += self.ratio[3]*aud_bound  #四个预测的叠加
        return out   #80 2


if __name__ == '__main__':
    from mmcv import Config
    cfg = Config.fromfile(r"C:\Users\yuanc\Desktop\cyuan research\sence segmentation\SceneSeg-master\lgss\config\all.py")
    model = LGSS(cfg)  #建立模型

    place_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 2048) #torch.Size([8, 10, 4, 2048])
    cast_feat = torch.randn(cfg.batch_size,  cfg.seq_len, cfg.shot_num, 512)  #torch.Size([8, 10, 4, 512])标准正太分布。
    act_feat = torch.randn(cfg.batch_size,   cfg.seq_len, cfg.shot_num, 512)  #torch.Size([8, 10, 4, 512])
    aud_feat = torch.randn(cfg.batch_size,   cfg.seq_len, cfg.shot_num, 512) # torch.Size([8, 10, 4, 257, 90])

    output = model(place_feat, cast_feat, act_feat, aud_feat)   #输出是个什么格式呢？
    print(cfg.batch_size)
    print(output.data.size())   #([80, 2])
    print(1)
