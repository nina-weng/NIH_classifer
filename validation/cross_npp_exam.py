import pandas as pd
import pytorch_lightning as pl
from torchvision import models
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchmetrics import Accuracy,AUROC
from torchmetrics.classification import MultilabelAUROC

import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from collections import defaultdict
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.utils import resample
from tabulate import tabulate

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='white')

from skimage.io import imread
from PIL import Image
import torchvision.transforms as T


import sys
from dataloader.dataloader import NIHDataResampleModule,NIHDataset
from torch.utils.data import DataLoader
from prediction.models import ResNet, DenseNet


D_set = ['Pneumothorax']
dataset = 'NIH'
f_perc_set = [0,50,100]

run_dir = '/work3/ninwe/run/{}/disease/'.format(dataset)
out_dir = '/work3/ninwe/run/{}/cross_npp/'.format(dataset)
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)



def test(model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()


def get_test_dataloader(img_data_dir,df,D):
    test_set = NIHDataset(img_data_dir,df, image_size=image_size, augmentation=False,
                                        pseudo_rgb=False,single_label=D,crop=None)
    dataloader = DataLoader(test_set,batch_size=64, shuffle=False, num_workers=0)
    return dataloader

def main(D,dataset,f_per):
    run_config_npp_None = 'resnet50-lr1e-06-ep20-pt1-aug1-{}%female-D{}-npp{}-ml0-rs0-imgs224-cropNone-mpara1'.format(
        f_per, D, 'None')
    run_config_npp_1 = 'resnet50-lr1e-06-ep20-pt1-aug1-{}%female-D{}-npp{}-ml0-rs0-imgs224-cropNone-mpara1'.format(
        f_per, D, '1')

    # H-PARA FOR MODELS
    model_choose = 'resnet'
    num_classes = 1
    lr = 1e-6
    pretrained = True
    model_scale = '50'

    if model_choose == 'resnet':
        model_type = ResNet
    elif model_choose == 'densenet':
        model_type = DenseNet

    # load trained model
    ckpt_dir_none = run_dir + '\\' + run_config_npp_None + '\\version_0\\checkpoints\\'
    file_list = os.listdir(ckpt_dir_none)
    assert len(file_list) == 1
    ckpt_path = ckpt_dir_none + file_list[0]
    model_npp_none = model_type.load_from_checkpoint(ckpt_path,
                                                     num_classes=num_classes, lr=lr, pretrained=pretrained,
                                                     model_scale=model_scale,
                                                     )
    model_npp_none.eval()

    ckpt_dir_1 = run_dir + '\\' + run_config_npp_1 + '\\version_0\\checkpoints\\'
    file_list = os.listdir(ckpt_dir_1)
    assert len(file_list) == 1
    ckpt_path = ckpt_dir_1 + file_list[0]
    model_npp_1 = model_type.load_from_checkpoint(ckpt_path,
                                                  num_classes=num_classes, lr=lr, pretrained=pretrained,
                                                  model_scale=model_scale,
                                                  )
    model_npp_1.eval()


    # load the test samples as dataloader
    image_size = (224, 224)
    if dataset == 'NIH':
        img_data_dir = '/work3/ninwe/dataset/NIH/preproc_224x224/'

    test_set_npp_None = pd.read_csv(run_dir + '\\' + run_config_npp_None + '\\test.version_0.csv')
    test_set_npp_1 = pd.read_csv(run_dir + '\\' + run_config_npp_1 + '\\test.version_0.csv')

    print(test_set_npp_None.shape)
    print(test_set_npp_1.shape)

    pid_none = test_set_npp_None['Patient ID'].unique()
    pid_none.sort()
    pid_1 = test_set_npp_1['Patient ID'].unique()
    pid_1.sort()

    print(pid_none[:20])
    print(pid_1[:20])

    dataloader_test_none = get_test_dataloader(img_data_dir, test_set_npp_None, D)
    dataloader_test_1 = get_test_dataloader(img_data_dir, test_set_npp_1, D)

    print('TESTING')
    save_dir = out_dir+'{}/'.format(D)
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")
    index_meaning = ['None', '1']
    for idx, trainon_model in enumerate([model_npp_none, model_npp_1]):
        for jdx, teston_dataloader in enumerate([dataloader_test_none, dataloader_test_1]):
            print(idx, jdx)
            print('train on npp={}, test on npp={}'.format(index_meaning[idx], index_meaning[jdx]))

            trainon_model.to(device)

            cols_names_classes = ['class_' + str(i) for i in range(0, num_classes)]
            cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
            cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

            preds_test, targets_test, logits_test = test(trainon_model, teston_dataloader, device)
            df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
            df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
            df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
            df = pd.concat([df, df_logits, df_targets], axis=1)
            df.to_csv(os.path.join(save_dir, 'train_{}_test_{}.csv'.format(index_meaning[idx], index_meaning[jdx])),
                      index=False)




if __name__ == '__main__':
    for D in D_set:
        for f_per in f_perc_set:
            main(D,dataset,f_per)