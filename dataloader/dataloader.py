
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

import torchvision.transforms as T
import pytorch_lightning as pl
import torchvision.transforms.functional as F
from skimage.io import imread
from PIL import Image
from tqdm import tqdm
import os




# disease_labels = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
DISEASE_LABELS = ['Effusion', 'Emphysema', 'Nodule', 'Atelectasis', 'Infiltration', 'Mass',
                  'Pleural_Thickening', 'Pneumothorax',
                  'Consolidation', 'Fibrosis', 'Cardiomegaly', 'Pneumonia', 'Edema', 'Hernia']#, 'No Finding']

class NIHDataset(Dataset):
    def __init__(self, img_data_dir, df_data, image_size, augmentation=False, pseudo_rgb = False):
        self.df_data = df_data
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb


        # self.labels = ['melanoma','nevus','basal cell carcinoma','actinic keratosis','benign keratosis','dermatofibroma',
        #   'vascular lesion','squamous cell carcinoma','others']
        self.labels=DISEASE_LABELS

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        # for idx, _ in enumerate(tqdm(range(len(self.df_data)), desc='Loading Data')):
        for idx in tqdm((self.df_data.index), desc='Loading Data'):
            img_path = img_data_dir + self.df_data.loc[idx, 'Image Index']
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(self.df_data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            sample = {'image_path': img_path, 'label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        # image = torch.from_numpy(sample['image'])
        image = T.ToTensor()(sample['image'])
        label = torch.from_numpy(sample['label'])

        # image = torch.permute(image, dims=(2, 0, 1))
        ### image = image / 255



        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)





        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        # image = imread(sample['image_path']).astype(np.float32)
        try:
            image = Image.open(sample['image_path']).convert('RGB') #PIL image
        except:
            print('PIL not working on image: {}'.format(sample['image_path']))
            image = imread(sample['image_path']).astype(np.float32)


        return {'image': image, 'label': sample['label']}

    def exam_augmentation(self,item):
        assert self.do_augment == True, 'No need for non-augmentation experiments'

        sample = self.get_sample(item) #PIL
        image = T.ToTensor()(sample['image'])

        if self.do_augment:
            image_aug = self.augment(image)

        image_all = torch.cat((image,image_aug),axis= 1)
        assert image_all.shape[1]==self.image_size[0]*2, 'image_all.shape[1] = {}'.format(image_all.shape[1])
        return image_all


class NIHDataModule(pl.LightningDataModule):
    def __init__(self, img_data_dir,csv_file_img, image_size, pseudo_rgb, batch_size, num_workers,augmentation,
                 view_position='all',vp_sample=False,only_gender=None,save_split=True,outdir=None,version_no=None,
                 gi_split=False,gender_setting=None,fold_num=0):
        super().__init__()
        self.img_data_dir = img_data_dir
        self.csv_file_img = csv_file_img
        self.view_position = view_position
        self.vp_sample = vp_sample
        self.only_gender = only_gender
        self.save_split=save_split
        self.outdir = outdir
        self.version_no = version_no
        self.gi_split = gi_split
        self.gender_setting = gender_setting
        self.fold_num = fold_num


        if self.gi_split:
            df_train = pd.read_csv('../datafiles/{}/Fold_{}/train.csv'.format(self.gender_setting,self.fold_num),header=0)
            df_valid = pd.read_csv('../datafiles/{}/Fold_{}/dev.csv'.format(self.gender_setting,self.fold_num), header=0)
            df_test_male = pd.read_csv('../datafiles/{}/Fold_{}/test_males.csv'.format(self.gender_setting,self.fold_num), header=0)
            df_test_female = pd.read_csv('../datafiles/{}/Fold_{}/test_female.csv'.format(self.gender_setting,self.fold_num), header=0)
            df_test = pd.concat([df_test_male, df_test_female])
            if self.view_position == 'AP' or self.view_position == 'PA':
                df_train = df_train[df_train['View Position'] == self.view_position]
                df_valid = df_valid[df_valid['View Position'] == self.view_position]
                df_test = df_test[df_test['View Position'] == self.view_position]

                df_train.reset_index(inplace=True)
                df_valid.reset_index(inplace=True)
                df_test.reset_index(inplace=True)

                if self.save_split:
                    df_train.to_csv(os.path.join(self.outdir, 'train.version_{}.csv'.format(self.version_no)),
                                    index=False)
                    df_valid.to_csv(os.path.join(self.outdir, 'val.version_{}.csv'.format(self.version_no)), index=False)
                    df_test.to_csv(os.path.join(self.outdir, 'test.version_{}.csv'.format(self.version_no)),
                                   index=False)

            df_train.reset_index(inplace = True)
            df_valid.reset_index(inplace=True)
            df_test.reset_index(inplace=True)

            self.df_train = df_train
            self.df_valid = df_valid
            self.df_test = df_test
        else:
            df_train,df_valid,df_test = self.dataset_split(self.csv_file_img)
            self.df_train = df_train
            self.df_valid = df_valid
            self.df_test = df_test

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation=augmentation


        self.train_set = NIHDataset(self.img_data_dir,self.df_train, self.image_size, augmentation=augmentation, pseudo_rgb=pseudo_rgb)
        self.val_set = NIHDataset(self.img_data_dir,self.df_valid, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.test_set = NIHDataset(self.img_data_dir,self.df_test, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def dataset_split(self,csv_all_img):
        df= pd.read_csv(csv_all_img,header=0)

        if self.view_position == 'AP' or self.view_position == 'PA':
            df = df[df['View Position'] == self.view_position]

        df_train = df[df.split == "train"]
        df_val = df[df.split == "valid"]
        df_test = df[df.split == "test"]

        if self.only_gender != None:
            df_train = df_train[df_train['Patient Gender'] == self.only_gender]

        if self.vp_sample == True:
            sample_rate = 0.4
            seed = 2023
            df_train = df_train.sample(frac=sample_rate, replace=False, random_state=seed)

        df_train.reset_index(inplace=True)
        df_val.reset_index(inplace=True)
        df_test.reset_index(inplace=True)

        if self.save_split:
            df_train.to_csv(os.path.join(self.outdir, 'train.version_{}.csv'.format(self.version_no)), index=False)
            df_val.to_csv(os.path.join(self.outdir, 'val.version_{}.csv'.format(self.version_no)), index=False)
            df_test.to_csv(os.path.join(self.outdir, 'test.version_{}.csv'.format(self.version_no)), index=False)

        return df_train,df_val,df_test


class NIHDataResampleModule(pl.LightningDataModule):
    def __init__(self, img_data_dir,csv_file_img, image_size, pseudo_rgb, batch_size, num_workers,augmentation,
                 outdir,version_no,
                 female_perc_in_training = None,
                 chose_disease='No Finding',
                 random_state=None):
        super().__init__()
        self.img_data_dir = img_data_dir
        self.csv_file_img = csv_file_img

        self.outdir = outdir
        self.version_no = version_no

        # new parameters
        self.female_perc_in_training = female_perc_in_training
        assert self.female_perc_in_training in [0,50,100], 'Not implemented female_perc_in_training: {}'.format(self.female_perc_in_training)
        self.chose_disease = chose_disease # str, one of the labels
        self.rs = random_state
        self.male, self.female = 'M', 'F'
        self.genders = [self.female, self.male]

        # pre-defined parameter
        self.num_per_gender = 13000
        self.disease_pervalence_total,self.disease_pervalence_female, self.disease_pervalence_male = self.get_prevalence()
        self.perc_train, self.perc_val, self.perc_test = 0.6,0.1,0.3
        assert self.perc_val+self.perc_test+self.perc_train == 1


        df_train,df_valid,df_test = self.dataset_sampling()
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation=augmentation




        self.train_set = NIHDataset(self.img_data_dir,self.df_train, self.image_size, augmentation=augmentation, pseudo_rgb=pseudo_rgb)
        self.val_set = NIHDataset(self.img_data_dir,self.df_valid, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.test_set = NIHDataset(self.img_data_dir,self.df_test, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)


    def dataset_sampling(self):
        '''
        doc: https://docs.google.com/document/d/1N1XJWFqF_5CDYkdbbXlzXmtepd-PFPhtKaDXwjKVYH8/edit
        :param csv_file_img:
        :return:
        '''
        df = pd.read_csv(self.csv_file_img, header=0)
        #patient_id_list = list(set(df['Patient ID'].to_list()))
        grouped = df.groupby('Patient ID')
        df_per_patient = grouped.apply(lambda x: x.sample(n=1, random_state=self.rs))


        train_set, val_set, test_set = None, None, None
        for each_gender in self.genders:
            for isDisease in [True, False]:
                this_df = df_per_patient[(df_per_patient['Patient Gender'] == each_gender) &
                                         (df_per_patient[self.chosen_disease] == isDisease)]
                print('{}+{}, number of samples:{}'.format(each_gender, isDisease, len(this_df)))

                N = int(self.num_per_gender * self.disease_pervalence_total[self.chosen_disease]) if isDisease else int(
                    self.num_per_gender * (1 - self.disease_pervalence_total[self.chosen_disease]))
                print('N:{}'.format(N))

                this_df = this_df.sample(n=N, random_state=self.rs)
                this_train, this_val, this_test = self.set_split(this_df, self.perc_train,self.perc_val,self.perc_test, self.rs)

                if each_gender == self.female and self.female_perc_in_training != 0:
                    if train_set is None:
                        train_set = this_train
                    else:
                        train_set = pd.concat([train_set, this_train], axis=0)
                if each_gender == self.male and self.female_perc_in_training != 100:
                    if train_set is None:
                        train_set = this_train
                    else:
                        train_set = pd.concat([train_set, this_train], axis=0)

                if val_set is None:
                    val_set = this_val
                else:
                    val_set = pd.concat([val_set, this_val], axis=0)
                if test_set is None:
                    test_set = this_test
                else:
                    test_set = pd.concat([test_set, this_test], axis=0)

        train_set.reset_index(inplace=True)
        val_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)

        # save splits
        train_set.to_csv(os.path.join(self.outdir, 'train.version_{}.csv'.format(self.version_no)), index=False)
        val_set.to_csv(os.path.join(self.outdir, 'val.version_{}.csv'.format(self.version_no)), index=False)
        test_set.to_csv(os.path.join(self.outdir, 'test.version_{}.csv'.format(self.version_no)), index=False)

        return train_set,val_set,test_set

    def get_prevalence(self):
        df = pd.read_csv(self.csv_file_img, header=0)
        df_per_patient = df.groupby(['Patient ID', 'Patient Gender']).mean()
        df_per_patient_p = df_per_patient.mean()[DISEASE_LABELS].to_list()

        df_per_patient_gender_p = df_per_patient.groupby(['Patient Gender']).mean()[DISEASE_LABELS]
        df_per_patient_gender_p_male = df_per_patient_gender_p.loc['M'].to_list()
        df_per_patient_gender_p_female = df_per_patient_gender_p.loc['F'].to_list()

        assert len(df_per_patient_gender_p_female) == len(DISEASE_LABELS)
        assert len(df_per_patient_gender_p_male) == len(DISEASE_LABELS)
        assert len(df_per_patient_p) == len(DISEASE_LABELS)

        dict_per_patient_p = {}
        for i,each_l in enumerate(DISEASE_LABELS): dict_per_patient_p[each_l] = df_per_patient_p[i]

        dict_per_patient_gender_p_female = {}
        for i, each_l in enumerate(DISEASE_LABELS): dict_per_patient_gender_p_female[each_l] = df_per_patient_gender_p_female[i]

        dict_per_patient_gender_p_male = {}
        for i, each_l in enumerate(DISEASE_LABELS): dict_per_patient_gender_p_male[each_l] = df_per_patient_gender_p_male[i]

        print('Disease prevalence total: {}'.format(dict_per_patient_p))
        print('Disease prevalence Female: {}'.format(dict_per_patient_gender_p_female))
        print('Disease prevalence Male: {}'.format(dict_per_patient_gender_p_male))

        return dict_per_patient_p,dict_per_patient_gender_p_female,dict_per_patient_gender_p_male

    def set_split(self,df,train_frac,val_frac,test_frac,rs):
        test = df.sample(frac=test_frac, random_state=rs)

        # get everything but the test sample
        train_val = df.drop(index=test.index)
        train = train_val.sample(frac=train_frac / (train_frac + val_frac), random_state=rs)
        val = train_val.drop(index=train.index)

        return train, val, test



