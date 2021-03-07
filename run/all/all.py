experiment_name = "all"
experiment_description = "scene segmentation with all modality"
# overall confg
data_root = r'C:\Users\yuanc\Desktop\cyuan research\sence segmentation\SceneSeg-master\lgss\data'
shot_frm_path = data_root
shot_num = 4  # even
seq_len = 10  # even
gpus = "0" # "0,1,2,3,4,5,6,7"

# dataset settings
dataset = dict(
    name="all",
    mode=['place', 'cast', 'act', 'aud'],
)
# model settings
model = dict(
    name='LGSS',
    sim_channel=512,  # dim of similarity vector
    place_feat_dim=2048,
    cast_feat_dim=512,
    act_feat_dim=512,
    aud_feat_dim=512,

    aud=dict(cos_channel=512),
    bidirectional=True,
    lstm_hidden_size=512,
    ratio=[0.5, 0.2, 0.2, 0.1]
    )

# optimizer
optim = dict(name='Adam',
             setting=dict(lr=1e-2, weight_decay=5e-4))
stepper = dict(name='MultiStepLR',
               setting=dict(milestones=[15]))
loss = dict(weight=[0.5, 5])

# runtime settings
resume = None
trainFlag = 1
testFlag = 0  #cahnge
batch_size = 8  #128
epochs = 30
logger = dict(log_interval=200, logs_dir=r"..\run/{}".format(experiment_name))
data_loader_kwargs = dict(num_workers=0, pin_memory=False, drop_last=True) #32
