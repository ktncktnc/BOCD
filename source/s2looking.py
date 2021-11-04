import torch, numpy as np, os, imagesize
import albumentations as A
from glob import glob
from typing import Dict
from math import ceil
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torchrs.transforms import Compose, ToTensor


class S2Looking(torch.utils.data.Dataset):
    """ The Satellite Side-Looking (S2Looking) dataset from 'S2Looking: A Satellite Side-Looking
    Dataset for Building Change Detection', Shen at al. (2021)
    https://arxiv.org/abs/2107.09244
    'S2Looking is a building change detection dataset that contains large-scale side-looking
    satellite images captured at varying off-nadir angles. The S2Looking dataset consists of
    5,000 registered bitemporal image pairs (size of 1024*1024, 0.5 ~ 0.8 m/pixel) of rural
    areas throughout the world and more than 65,920 annotated change instances. We provide
    two label maps to separately indicate the newly built and demolished building regions
    for each sample in the dataset.'
    """
    splits = ["train", "val", "test"]


    def __init__(
        self,
        root: str = ".data/s2looking",
        split: str = "train",
        transform: A.Compose = A.Compose([
                A.Resize(256, 256),
                A.RandomRotate90(),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],
            additional_targets = {'image0': 'image'}
        ),
    ):
        #assert split in self.splits
        self.root = root
        self.transform = transform
        self.files, self.image_names = self.load_files(root, split)

    @staticmethod
    def load_files(root: str, split: str):
        files = []
        images = glob(os.path.join(root, split, "Image1", "*.png"))
        images = sorted([os.path.basename(image) for image in images])
        for image in images:
            image1 = os.path.join(root, split, "Image1", image)
            image2 = os.path.join(root, split, "Image2", image)
            mask = os.path.join(root, split, "label", image)
            
            files.append(dict(image1=image1, image2=image2, mask=mask))

        return files, images

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, mask
        x: (2, 13, h, w)
        build_mask: (1, h, w)
        demolish_mask: (1, h, w)
        """
        files = self.files[idx]
        image1 = np.array(Image.open(files["image1"]))
        image2 = np.array(Image.open(files["image2"]))

        mask = np.array(Image.open(files["mask"]))
        mask = np.expand_dims(mask, axis=2)

        transformed = self.transform(image = image1, image0 = image2, mask = mask)
        
        image1 = transformed['image']
        image2 = transformed['image0']
        mask = transformed['mask']

        x = torch.stack([image1, image2], dim=0)
        return dict(x = x, mask = mask)

class CroppedS2Looking(torch.utils.data.Dataset):
    """ The Satellite Side-Looking (S2Looking) dataset from 'S2Looking: A Satellite Side-Looking
    Dataset for Building Change Detection', Shen at al. (2021)
    https://arxiv.org/abs/2107.09244
    'S2Looking is a building change detection dataset that contains large-scale side-looking
    satellite images captured at varying off-nadir angles. The S2Looking dataset consists of
    5,000 registered bitemporal image pairs (size of 1024*1024, 0.5 ~ 0.8 m/pixel) of rural
    areas throughout the world and more than 65,920 annotated change instances. We provide
    two label maps to separately indicate the newly built and demolished building regions
    for each sample in the dataset.'
    """
    splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = ".data/s2looking",
        split: str = "train",
        stride: int = 256,
        patch_size: tuple = (256, 256),
        transform: Compose = Compose([ToTensor()]),
    ):
        #assert split in self.splits
        self.root = root
        self.transform = transform

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, tuple) and len(patch_size) == 2:
            self.patch_size = patch_size
        else:
            raise TypeError("Error type patch_size")

        self.stride = stride

        self.files, self.image_names = self.load_files(root, split)
        self.coords = []
        self.n_patches = 0

        self.init_coords()

    @staticmethod
    def load_files(root: str, split: str):
        files = []
        images = glob(os.path.join(root, split, "Image1", "*.png"))
        images = sorted([os.path.basename(image) for image in images])

        for image in images:
            image1 = os.path.join(root, split, "Image1", image)
            image2 = os.path.join(root, split, "Image2", image)
            mask = os.path.join(root, split, "label", image)
            
            files.append(dict(image1=image1, image2=image2, mask=mask))

        return files, images

    def init_coords(self):
        for idx, item in enumerate(self.files):
            width, height = imagesize.get(item['image1'])

            n1 = ceil((width - self.patch_size[0] + 1) / self.stride)
            n2 = ceil((height - self.patch_size[1] + 1) / self.stride)

            n_patches_i = n1 * n2
            self.n_patches += n_patches_i

            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (idx, 
                                    [self.stride*i, self.stride*i + self.patch_size[0], self.stride*j, self.stride*j + self.patch_size[1]],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    self.coords.append(current_patch_coords)
                    
    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, mask
        x: (2, 13, h, w)
        build_mask: (1, h, w)
        demolish_mask: (1, h, w)
        """
        item = self.coords[idx]
        idx = item[0]
        limits = item[1]
        files = self.files[idx]

        mask = np.array(Image.open(files["mask"]))[limits[0]:limits[1], limits[2]:limits[3]]
        mask = np.expand_dims(mask, axis=2)

        image1 = np.array(Image.open(files["image1"]))[limits[0]:limits[1], limits[2]:limits[3], :]
        image2 = np.array(Image.open(files["image2"]))[limits[0]:limits[1], limits[2]:limits[3], :]

        image1 = self.transform(image1)
        image2 = self.transform(image2)
        mask = self.transform(mask)
        
        x = torch.stack([image1, image2], dim=0)
        return dict(x=x, mask=mask)