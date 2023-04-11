
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
import pandas as pd
import os
from tqdm import tqdm
import numpy as np


def main():

    # point to the parent directory that contains the folder 'isic_images'
    img_data_dir = '/work3/ninwe/dataset/NIH/'
    # img_data_dir = 'D:\\ninavv\\phd\\data\\NIH\\'
    csv_file = '../datafiles/'+'Data_Entry_2017_v2020_clean_split.csv'

    df = pd.read_csv(csv_file,header=0)

    #df['path_preproc'] = df['path']

    preproc_dir = 'preproc_224x224/'
    out_dir = img_data_dir

    if not os.path.exists(out_dir + preproc_dir):
        os.makedirs(out_dir + preproc_dir)

    for idx, p in enumerate(tqdm(df['Image Index'])):
        preproc_filename = p
        #df.loc[idx, 'path_preproc'] = preproc_dir + preproc_filename
        out_path = out_dir + preproc_dir + preproc_filename

        if not os.path.exists(out_path):
            image = imread(img_data_dir +'/images/'+ p)
            image = resize(image, output_shape=(224, 224), preserve_range=True)
            imsave(out_path, image.astype(np.uint8))

    #df.to_csv(csv_file)


if __name__ == '__main__':
    main()