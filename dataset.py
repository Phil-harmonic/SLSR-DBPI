import torch
from torch.utils.data import Dataset
import h5py
import os
from utils import load_image, normalize, image_to_array, downsample, upsample, modcrop
import numpy as np


def checkImage(filename: str):
    return any(filename.endswith(extension) for extension in [".bmp", ".BMP",
                                                              ".jpg", ".JPG",
                                                              ".png", ".PNG",
                                                              ".jpeg", ".JPEG"])


class DatasetFromH5(Dataset):
    def __init__(self, file_path: str):
        super(DatasetFromH5, self).__init__()
        hf = h5py.File(file_path)
        self.hr = hf["hr"]
        self.lr = hf["lr"]

    def __getitem__(self, index):
        return torch.from_numpy(self.lr[index, :, :, :]).float(), torch.from_numpy(self.hr[index, :, :, :]).float()

    def __len__(self):
        return self.hr.shape[0]


class DatasetFromFolder(Dataset):
    def __init__(self, hr_folder: str, lr_folder: str, down_sample: bool = False,
                 up_sample: bool = False, factor: int = 0):
        super(DatasetFromFolder, self).__init__()
        self.hr_images = []
        self.lr_images = []
        if down_sample and up_sample:
            for x in os.listdir(hr_folder):
                if checkImage(x):
                    hr_image = modcrop(load_image(os.path.join(hr_folder, x)), 4)
                    lr_image = upsample(downsample(hr_image, factor), factor)
                    self.hr_images.append(image_to_array(hr_image).astype(np.uint8))
                    self.lr_images.append(image_to_array(lr_image).astype(np.uint8))
        elif down_sample is True and up_sample is False:
            for x in os.listdir(hr_folder):
                if checkImage(x):
                    hr_image = modcrop(load_image(os.path.join(hr_folder, x)), 4)
                    lr_image = downsample(hr_image, factor)
                    self.hr_images.append(image_to_array(hr_image).astype(np.uint8))
                    self.lr_images.append(image_to_array(lr_image).astype(np.uint8))

        elif not down_sample and not up_sample:
            for x in os.listdir(hr_folder):
                if checkImage(x):
                    hr_image = image_to_array(modcrop(load_image(os.path.join(hr_folder, x)), 4)).astype(np.uint8)
                    self.hr_images.append(hr_image)
            for x in os.listdir(lr_folder):
                if checkImage(x):
                    lr_image = image_to_array(modcrop(load_image(os.path.join(lr_folder, x)), 4)).astype(np.uint8)
                    self.lr_images.append(lr_image)

    def __getitem__(self, index):
        return torch.from_numpy(normalize(self.lr_images[index])).float(), \
               torch.from_numpy(normalize(self.hr_images[index])).float()

    def __len__(self):
        return len(self.lr_images)


# class DatasetFromFolder(Dataset):
#     def __init__(self, hrDir: str, lrDir: str, factor: int):
#         super(DatasetFromFolder, self).__init__()
#         self.factor = factor
#         self.hr_paths = [os.path.join(hrDir, x) for x in os.listdir(hrDir) if checkImage(x)]
#         # self.lr_paths = [os.path.join(lrDir, x) for x in os.listdir(lrDir) if checkImage(x)]
#
#     def __getitem__(self, index: int):
#         hr_image = modcrop(load_image(self.hr_paths[index]), factor=self.factor)
#         lr_image = downsample(hr_image, self.factor)
#         hr_image = normalize(image_to_array(hr_image))
#         lr_image = normalize(image_to_array(lr_image))
#
#         return torch.from_numpy(lr_image).float(), torch.from_numpy(hr_image).float()
#
#     def __len__(self):
#         return len(self.hr_paths)
