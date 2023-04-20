
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
                  'Consolidation', 'Fibrosis', 'Cardiomegaly', 'Pneumonia', 'Edema', 'Hernia', 'No Finding']

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
                 view_position='all',vp_sample=False,only_gender=None,save_split=True,outdir=None,version_no=None,gi_split=False):
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

        if self.gi_split:
            self.df_train = pd.read_csv('../datafiles/100%_female/FOLD_0/train.csv',header=0)
            self.df_val = pd.read_csv('../datafiles/100%_female/FOLD_0/dev.csv', header=0)
            df_test_male = pd.read_csv('../datafiles/100%_female/FOLD_0/test_males.csv', header=0)
            df_test_female = pd.read_csv('../datafiles/100%_female/FOLD_0/test_female.csv', header=0)
            df_test = pd.concat([df_test_male, df_test_female])
            df_test.reset_index(inplace = True)
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

        if self.save_split:
            df_train.to_csv(os.path.join(self.outdir, 'train.version_{}.csv'.format(self.version_no)), index=False)
            df_val.to_csv(os.path.join(self.outdir, 'val.version_{}.csv'.format(self.version_no)), index=False)
            df_test.to_csv(os.path.join(self.outdir, 'test.version_{}.csv'.format(self.version_no)), index=False)

        return df_train,df_val,df_test


