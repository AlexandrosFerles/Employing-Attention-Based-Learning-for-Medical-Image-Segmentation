import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image

import nibabel as nib
from torch.utils.data.dataset import Dataset
import glob
from typing import List


class DataSet(data.Dataset):

    def __init__(self, image_folder, mask_folder):
        self.image_path = sorted(list(map(lambda x: os.path.join(image_folder, x), os.listdir(image_folder))))
        self.mask_path = sorted(list(map(lambda x: os.path.join(mask_folder, x), os.listdir(mask_folder))))
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        single_image = self.get_x(index)
        ground_truth = self.get_y(index)
        single_image = self.transform(single_image)
        ground_truth = self.transform(ground_truth)

        return single_image, ground_truth

    def get_x(self, idx):
        img = Image.open(self.image_path[idx]).convert('L')
        return img

    def get_y(self, idx):
        img = Image.open(self.mask_path[idx]).convert('L')
        return img

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_path)


class NiftiDataset(Dataset):
    """
    create a dataset class in PyTorch for reading NIfTI files
    Args:
        source_dir (str): path to source images
        target_dir (str): path to target images
        transform (Callable): transform to apply to both source and target images
        preload (bool): load all data when initializing the dataset
    """

    def __init__(self, source_dir:str, target_dir:str, preload:bool=False):
        self.source_dir, self.target_dir = source_dir, target_dir
        self.source_fns, self.target_fns = self._glob_imgs(source_dir), self._glob_imgs(target_dir)
        self.transform = transforms.Compose([
            transforms.Resize((121,512, 512)),
            transforms.ToTensor(),
        ])
        self.preload = preload
        if len(self.source_fns) != len(self.target_fns) or len(self.source_fns) == 0:
            raise ValueError('Number of source and target images must be equal and non-zero')
        if preload:
            self.imgs = [(nib.load(s).get_data(), nib.load(t).get_data())
                         for s, t in zip(self.source_fns, self.target_fns)]

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, idx:int):
        if not self.preload:
            src_fn, tgt_fn = self.source_fns[idx], self.target_fns[idx]
            sample = (nib.load(src_fn).get_data(), nib.load(tgt_fn).get_data())
        else:
            sample = self.imgs[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _glob_imgs(path: str, ext='*nii.gz*') -> List[str]:
        """ grab all `ext` files in a directory and sort them for consistency """
        fns = sorted(glob.glob(path + ext))
        return fns


class NumpyDataset(data.Dataset):

    def __init__(self, image_folder, mask_folder):
        self.image_path = sorted(list(map(lambda x: os.path.join(image_folder, x), os.listdir(image_folder))))
        self.mask_path = sorted(list(map(lambda x: os.path.join(mask_folder, x), os.listdir(mask_folder))))

    def __getitem__(self, index):
        single_image = self._get_x(index)
        ground_truth = self._get_y(index)

        return single_image, ground_truth

    def _setup_torch_format_data(self, idx):
        img = np.load(self.image_path[idx])['arr_0']
        img = np.expand_dims(img, axis=0)
        # per-dataset normalization to zero mean, 1 std
        timg =  torch.from_numpy(img)  
        timg += 798.8547770393742
        timg /= 889.8269350428398
        return timg

    def _setup_torch_format_masks(self, idx):
        img = np.load(self.mask_path[idx])['arr_0']
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.int16)
        return torch.from_numpy(img)

    def _get_x(self, idx):
        return self._setup_torch_format_data(idx)

    def _get_y(self, idx):
        return self._setup_torch_format_masks(idx)

    def __len__(self):
        return len(self.image_path)