import sys
sys.path.append('../../NIH_classifer')
from validation.model_tmpuse import ResNet_chexpert,ResNet_NIH
from dataloader.dataloader import NIHDataResampleModule
from validation.dataloader_tmpuse import CheXpertDataResampleModule
from prediction.disease_prediction import get_cur_version
import os
import torch
from prediction.disease_prediction import test_func
import pandas as pd

def load_model(dataset_type,ckpt_dir):
    model_choose = 'resnet'
    num_classes = 1
    lr = 1e-6
    pretrained = True
    model_scale = '50'

    if model_choose == 'resnet':
        if dataset_type == 'NIH':
            model_type = ResNet_NIH
        elif dataset_type == 'chexpert':
            model_type = ResNet_chexpert


    file_list = os.listdir(ckpt_dir)
    assert len(file_list) == 1
    ckpt_path = ckpt_dir + file_list[0]
    model = model_type.load_from_checkpoint(ckpt_path,
                                                num_classes=num_classes, lr=lr, pretrained=pretrained,
                                                model_scale=model_scale,
                                                )

    return model





# ## Hyperparameters: D=Pneumothorax, 0%female, rs=range[0,5]
def main(f_perc):
    run_config_NIH = []
    run_config_chexpert = []
    version_no_NIH = []
    version_no_chexpert = []

    for i in range(1):
        run_config_NIH.append(
            'resnet50-lr1e-06-ep20-pt1-aug1-{}%female-DPneumothorax-npp1-ml0-rs{}-imgs224'.format(f_perc,i))
        run_config_chexpert.append('resnet50-lr1e-06-ep20-pt1-aug1-{}%female-DPneumothorax-npp1-ml0-rs{}-imgs224'.format(f_perc,i))
        version_no_NIH.append(0)
        version_no_chexpert.append(0)

    run_dir_NIH = 'D:\\ninavv\\phd\\research\\run_results\\NIH_results\\disease\\'
    run_dir_chexpert = 'D:\\ninavv\\phd\\research\\run_results\\chexpert_results\\disease\\'

    run_dir_chexpert = '/work3/ninwe/run/chexpert/disease/'
    run_dir_NIH = '/work3/ninwe/run/NIH/disease/'

    rs_chose = 0

    ckpt_dir_NIH = run_dir_NIH + '/' + run_config_NIH[rs_chose] + '/version_0/checkpoints/'
    NIH_model = load_model('NIH',ckpt_dir_NIH)

    ckpt_dir_chexpert = run_dir_chexpert + '/' + run_config_chexpert[rs_chose] + '/version_0/checkpoints/'
    chexpert_model = load_model('chexpert', ckpt_dir_chexpert)

    img_data_dir = '/work3/ninwe/dataset/NIH/preproc_224x224/'
    csv_file_img = '../datafiles/'+'Data_Entry_2017_v2020_clean_split.csv'
    image_size=(224,224)
    batch_size=64
    num_workers=1
    augmentation=False
    chose_disease_str = 'Pneumothorax'
    run_dir = '/work3/ninwe/run/NIH/disease/'
    run_config='interdataset_D{}_fpec{}_rs{}'.format(chose_disease_str,f_perc,rs_chose)
    out_dir = run_dir + run_config
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir_NIH = out_dir+'/NIH_datasplit/'
    if not os.path.exists(out_dir_NIH):
        os.makedirs(out_dir_NIH)

    cur_version = get_cur_version(out_dir)


    data_NIH = NIHDataResampleModule(img_data_dir=img_data_dir,
                                 csv_file_img=csv_file_img,
                                 image_size=image_size,
                                 pseudo_rgb=False,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 augmentation=augmentation,
                                 outdir=out_dir_NIH,
                                 version_no=cur_version,
                                 female_perc_in_training=f_perc,
                                 chose_disease=chose_disease_str,
                                 random_state=rs_chose,
                                 num_classes=1,
                                 num_per_patient=1,
                                 crop=None)

    img_data_dir = '/work3/ninwe/dataset/'
    csv_file_img = '../datafiles/' + 'chexpert.sample.csv'
    out_dir_chexpert= out_dir + '/chexpert_datasplit/'
    if not os.path.exists(out_dir_chexpert):
        os.makedirs(out_dir_chexpert)

    cur_version = get_cur_version(out_dir)

    data_chexpert = CheXpertDataResampleModule(img_data_dir=img_data_dir,
                                csv_file_img=csv_file_img,
                                image_size=image_size,
                                pseudo_rgb=False,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                augmentation=augmentation,
                                outdir=out_dir_chexpert,
                                version_no=cur_version,
                                female_perc_in_training=f_perc,
                                chose_disease=chose_disease_str,
                                random_state=rs_chose,
                                num_classes=1,
                                num_per_patient=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")

    NIH_model.to(device)
    chexpert_model.to(device)

    num_classes=1
    cols_names_classes = ['class_' + str(i) for i in range(0, num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    print('1.TESTING NIH test set Using NIH model')
    preds_test, targets_test, logits_test = test_func(NIH_model, data_NIH.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.testNIHtrainNIH.version_{}.csv'.format(cur_version)), index=False)

    print('2.TESTING NIH test set Using chexpert model')
    preds_test, targets_test, logits_test = test_func(chexpert_model, data_NIH.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.testNIHtrainChexpert.version_{}.csv'.format(cur_version)), index=False)

    print('3.TESTING chexpert test set Using chexpert model')
    preds_test, targets_test, logits_test = test_func(chexpert_model, data_chexpert.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.testChexperttrainChexpert.version_{}.csv'.format(cur_version)), index=False)

    print('4.TESTING chexpert test set Using NIH model')
    preds_test, targets_test, logits_test = test_func(NIH_model, data_chexpert.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.testChexperttrainNIH.version_{}.csv'.format(cur_version)), index=False)


if __name__ == '__main__':
    for f_perc in [0,50,100]:
        main(f_perc)