from source.utils import *
from pandas import read_csv
from tqdm import tqdm
import os

import torch
from torch.utils.data import Dataset
from math import floor, ceil, sqrt, exp
import random
import albumentations 
from albumentations.pytorch import ToTensorV2


class CDDataset(Dataset):
    """
    Change Detection dataset class, used for both training and test data.
        
    Args:
        path (string): Path to the OSCD dataset directory
        train (boolean: get train or test dataset
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    def __init__(self, path, train = True, patch_side = 96, stride = None, transform=None, img_type = 0, normalize = True, FP_MODIFIER = 10):

        
        # basics
        self.transform = transform
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride
        
        if train:
            fname = 'train.txt'
        else:
            fname = 'test.txt'
        
        #print(path + fname)
        self.names = read_csv(os.path.join(path, fname)).columns
        self.n_imgs = self.names.shape[0]
        
        n_pix = 0
        true_pix = 0
        
        
        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            I1, I2, cm = read_sentinel_img_trio(os.path.join(self.path, im_name), img_type, normalize)
            self.imgs_1[im_name] = reshape_for_torch(I1, False)
            self.imgs_2[im_name] = reshape_for_torch(I2, False)
            self.change_maps[im_name] = cm
            
            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()
            
            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i
            
            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_name, 
                                    [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    self.patch_coords.append(current_patch_coords)
                    
        self.weights = [FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]
        
    def get_img(self, im_name):
        return {
            'I1': self.imgs_1[im_name], 
            'I2': self.imgs_2[im_name], 
            'label': self.change_maps[im_name]
        }

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = 1*np.array(label)
        
        sample = {'I1': I1, 'I2': I2, 'label': label}
        
        if self.transform:
           sample = self.augment(sample)

        return sample

    
    def augment(self, sample):
        if self.transform is not None:
            x = sample['I1']
            y = sample['I2']
            gt = sample['label']

            num_channels = x.shape[0]

            image = np.concatenate((x, y), axis = 0)

            print(image)
            print(gt)
            transformed = self.transform(image = image, mask = gt)

            image, gt = transformed['image'], transformed['mask']

            x = image[:num_channels, :, :]
            y = image[num_channels:, :, :]
            
            return {'I1': x, 'I2': y, 'label': gt}
        
        return sample


class RandomCropCDDataset(Dataset):
    """
    Change Detection dataset class, used for both training and test data.
        
    Args:
        path (string): Path to the OSCD dataset directory
        train (boolean: get train or test dataset
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    def __init__(self, path, train = True, patch_side = 96, nums_crop = 3000, transform=None, img_type = 0, normalize = True, FP_MODIFIER = 10):

        # basics
        self.transform = transform
        self.path = path
        self.patch_side = patch_side

        if train:
            fname = 'train.txt'
        else:
            fname = 'test.txt'
        
        #print(path + fname)
        self.names = read_csv(os.path.join(path, fname)).columns
        self.n_imgs = self.names.shape[0]

        nums_crop_per_img = nums_crop // self.n_imgs
        
        n_pix = 0
        true_pix = 0
        
        
        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            I1, I2, cm = read_sentinel_img_trio(os.path.join(self.path, im_name), img_type, normalize)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm
            
            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()
            
            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            self.n_patches_per_image[im_name] = nums_crop_per_img
            self.n_patches += nums_crop_per_img

            n1 = s[1] - self.patch_side
            n2 = s[2] - self.patch_side

            for i in range(nums_crop_per_img):
                n1_random = random.randint(0, n1)
                n2_random = random.randint(0, n2)

                current_patch_coords = (im_name, 
                                    [n1_random, n1_random + self.patch_side, n2_random, n2_random + self.patch_side],
                                    [n1_random + self.patch_side//2, n2_random + self.patch_side//2])
                self.patch_coords.append(current_patch_coords)
                    
        self.weights = [FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]
        
    def get_img(self, im_name):
        return {
            'I1': self.imgs_1[im_name], 
            'I2': self.imgs_2[im_name], 
            'label': self.change_maps[im_name]
        }

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = 1*np.array(label)
        
        sample = {'I1': I1, 'I2': I2, 'label': label}
        
        if self.transform:
           sample = self.augment(sample)

        return sample

    
    def augment(self, sample):
        if self.transform is not None:
            x = sample['I1']
            y = sample['I2']
            gt = sample['label']

            num_channels = x.shape[0]

            image = np.concatenate((x, y), axis = 0)
            transformed = self.transform(image = image, mask = gt)

            image, gt = transformed['image'], transformed['mask']

            x = image[:num_channels, :, :]
            y = image[num_channels:, :, :]
            
            return {'I1': x, 'I2': y, 'label': gt}
        
        return sample