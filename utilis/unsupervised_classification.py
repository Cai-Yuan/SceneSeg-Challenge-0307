import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mmcv import Config
from src import get_data
from torch.utils.data import DataLoader
from utilis import  read_pkl
import os


def load_excel(path):
    file=pd.read_excel(path)
    return file

def data_balance(features,targets):
    balanced_feature=[]
    size=[]
    final=[]

    b=(targets.iloc[:,0]).unique()


    for i in range(len(b)):
        temp=(targets[ targets.iloc[:,0]==i]).index
        size.append(len(temp))
        balanced_feature.append(features.loc[temp,:])
        balanced_feature[i]['label']=i


    max_size=max(size)
    for i in range(len(b)):
        final.append(balanced_feature[i].loc[balanced_feature[i].index.repeat(max_size//size[i])])

    balance_final_feature=shuffle(pd.concat(final, axis=0))
    final_features = balance_final_feature.drop(columns=['label'])
    final_targets = pd.DataFrame(balance_final_feature['label'])

    return final_features,final_targets

def models_bulid(trainfeatures, trainlabel):
    # 建立模型--patient
    #knn_patient = KNeighborsClassifier(n_neighbors=8).fit(trainfeatures, trainlabel)
    #svm_patient = SVC(kernel='rbf', C=100, gamma=0.1, probability=True).fit(trainfeatures, trainlabel)
    random_forest_patient = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=6).fit(trainfeatures, trainlabel)
    #gradient_boosted_patient = GradientBoostingClassifier(random_state=20).fit(trainfeatures, trainlabel)
    return  random_forest_patient
    #return [knn_patient, svm_patient, random_forest_patient, gradient_boosted_patient]

def unsupervised_model():
    feature_data = load_excel(r"C:\Users\yuanc\Desktop\cyuan research\sence segmentation\SceneSeg-Challenge_CaiYuan\image\dis_feature\unsupervised_train.xlsx")
    feature_data=feature_data.iloc[0:10000, :]

    nn = feature_data.loc[(feature_data['label'] ==0) ]
    pp = feature_data.loc[(feature_data['label'] ==1 ) ]


    selected_task = pd.concat([nn, pp], axis=0)
    x = selected_task.iloc[:, range(4)]
    y = selected_task.loc[:, ['label']]
    x.index = range(len(x))
    y.index = range(len(y))

    normalized_value = pd.DataFrame(StandardScaler().fit_transform(x))
    normalized_value.columns  =x.columns.values.tolist()
    final_useful_data=pd.concat([normalized_value,y], axis=1)
    final_useful_data = final_useful_data.dropna(axis = 0, how='any')

    features = final_useful_data.drop(columns='label')
    targets = pd.DataFrame(final_useful_data['label'])

    train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.2, random_state=42)
    train_x, train_y = data_balance(train_x, train_y)

    class_model = models_bulid(train_x, train_y)

    return class_model


def unsupervised_pre(cfg,model, IDs):
    data_root = cfg.feature_dis_save_root
    imdbid=IDs[0]['imdbid'][0]
    shotid1=  int(IDs[0]['shotid'][0])
    shotid2=int(IDs[-1]['shotid'][-1])
    if len(set(IDs[0]['imdbid'])) != 1:
        return None

    file_path = os.path.join(data_root, imdbid.split('.')[0]+'.xlsx')
    imdbid_data = pd.read_excel(file_path)

    feature=imdbid_data.iloc[shotid1-1:shotid2,[0,1,2,3]]
    probas_ = np.round(model.predict_proba(feature), 2)


    return probas_


def classification():
    feature_data = load_excel(r"C:\Users\yuanc\Desktop\cyuan research\sence segmentation\SceneSeg-Challenge_CaiYuan\image\dis_feature\unsupervised_train.xlsx")
    feature_data=feature_data.iloc[0:10000, :]

    nn = feature_data.loc[(feature_data['label'] ==0) ]
    pp = feature_data.loc[(feature_data['label'] ==1 ) ]



    selected_task = pd.concat([nn, pp], axis=0)
    x = selected_task.iloc[:, range(4)]
    y = selected_task.loc[:, ['label']]
    x.index = range(len(x))
    y.index = range(len(y))



    normalized_value = pd.DataFrame(StandardScaler().fit_transform(x))
    normalized_value.columns  =x.columns.values.tolist()
    final_useful_data=pd.concat([normalized_value,y], axis=1)
    final_useful_data = final_useful_data.dropna(axis = 0, how='any')

    ####################################### ################################
    ############## ###################   ##########################

    #classifers(data, label, feature_num=10):  # data包括
    features = final_useful_data.drop(columns='label')
    targets = pd.DataFrame(final_useful_data['label'])

    train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.3, random_state=42)
    train_x, train_y = data_balance(train_x, train_y)

    #train_x, test_x = feature_normalization(train_x, test_x)
    # Convert y to one-dimensional array (vector)
    train_y = np.array(train_y).reshape((-1,))
    test_y = np.array(test_y).reshape((-1,))

    # 建立模型--patient
    class_models = models_bulid(train_x, train_y)

    fpr = []
    tpr = []
    AUC = []
    probility = []

    probility.append(test_y)
    for i in range(len(class_models)):
        # pridiction=svm.predict(x_test)
        probas_ = np.round(class_models[i].predict_proba(test_x), 2)

        print(classification_report(test_y, class_models[i].predict(test_x)))
        fpr_, tpr_, thresholds = roc_curve(test_y, probas_[:, 1], pos_label=1, drop_intermediate=True)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
        probility.append(probas_)
        AUC_ = auc(fpr_, tpr_)
        fpr.append(fpr_)
        tpr.append(tpr_)
        AUC.append(AUC_)

        # 开始画图
    AUC = np.round(AUC, 2)
    font = {'weight': 'normal'}
    plt.figure(figsize=(7, 7))
    # plt.subplot(1,2,2)
    # plt.title('ROCs of classification models at patient level: \nrejection-%s VS rejection-%s' %(str(key_list[0]),str(key_list[1])), fontsize=16, fontdict=font)
    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)

    color_selection=['green','red','skyblue','blue']
    model=['KNN,AUC', 'SVM,AUC', 'RF,AUC', 'GradientBoosting,AUC']
    for i in range(len(class_models)):
        plt.plot(fpr[i], tpr[i], color=color_selection[i], linewidth=3, label='KNN_patient, AUC=' + str(AUC[i]))

    plt.legend(fontsize=14)  # 显示图例
    # 设置刻度字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.savefig(os.path.join(os.path.abspath('..') ,'data','Calculation results', 'classfication_patient.png'), dpi=400, bbox_inches='tight')
    plt.show()

if __name__=='__main__':

    config_path = r"..\config\all.py"
    cfg = Config.fromfile(config_path)
    model=unsupervised_model()

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
        target     =  target.view(-1).cuda()
        unsupervised_clustering(cfg,model,  IDs)
        print(1)
