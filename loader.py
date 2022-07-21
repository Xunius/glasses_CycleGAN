'''Load dataset and create dataloader

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2022-07-20 23:20:52.
'''
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image





class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloader(config):

    # read in csv label file
    train_labels = pd.read_csv(config['train_label_file'], index_col=0)
    # get "glasses" column
    train_labels = train_labels['glasses']

    img_dir = config['image_folder']
    img_file_glasses, img_file_no_glasses, img_file_test = [], [], []

    # loop through image files
    img_files = os.listdir(img_dir)
    for fii in img_files:
        # face id
        idii = int(fii.split('.')[0].split('-')[1])

        if idii in train_labels.index:
            # img is in train set
            labelii = int(train_labels[idii])

            if labelii == 1:
                img_file_glasses.append(os.path.join(img_dir, fii))
            else:
                img_file_no_glasses.append(os.path.join(img_dir, fii))
        else:
            # img is in test set
            img_file_test.append(os.path.join(img_dir, fii))

    print('NO. glasses image:', len(img_file_glasses))
    print('NO. no glasses image:', len(img_file_no_glasses))
    print('NO. test image:', len(img_file_test))

    # split train-val
    train_imgs_glasses = img_file_glasses[200:]
    train_imgs_no_glasses = img_file_no_glasses[200:]
    valid_imgs_glasses = img_file_glasses[:200]
    valid_imgs_no_glasses = img_file_no_glasses[:200]

    # create transform
    transform = transforms.Compose([
        transforms.Resize((config['image_height'], config['image_width'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomApply([AddGaussianNoise(std=0.1)], p=0.3),
        ])

    # create datasets
    train_data = CycleGANDataset(train_imgs_glasses, train_imgs_no_glasses, transform)
    valid_data = CycleGANDataset(valid_imgs_glasses, valid_imgs_no_glasses, transform)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=False)

    return train_loader, valid_loader, train_data, valid_data

class CycleGANDataset(Dataset):
    def __init__(self, files_X, files_Y, transform=None):
        Dataset.__init__(self)

        self.files_X = files_X
        self.files_Y = files_Y
        self.transform = transform
        self.crop_size = 224

        self.len_X = len(self.files_X)
        self.len_Y = len(self.files_Y)
        self.len_dataset = max(self.len_X, self.len_Y)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):

        sample_X = self.files_X[idx % self.len_X]
        sample_Y = self.files_Y[idx % self.len_Y]

        sample_X = Image.open(sample_X)
        sample_Y = Image.open(sample_Y)

        if self.transform:
            sample_X =  self.transform(sample_X)
            sample_Y =  self.transform(sample_Y)

        return sample_X, sample_Y







if __name__ == '__main__':
    from config import config
    train_loader, valid_loader, train_data, valid_data = get_dataloader(config)

    for xx, yy in train_loader:
        print(xx.shape)
        print(yy.shape)
