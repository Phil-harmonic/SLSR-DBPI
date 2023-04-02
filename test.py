import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from networks_DBPI import Upsampler, Downsampler
from utils import checkImage, load_image, modcrop, downsample, image_to_array, normalize, upsample, im2tensor,\
    tensor2im
import matplotlib.pyplot as plt
import cv2
import xlsxwriter
import math
from utils import centerCrop


def test_DBPI(testFolder: str, outroot: str, statePath: str, show_flag: bool = True, save_image: bool = False,
              cal_clarity: bool = False, x4: bool = False):

    if not os.path.exists(outroot):
        os.mkdir(outroot)

    up_sampler = Upsampler(x4=False).cuda()
    up_sampler.load_state_dict(torch.load(statePath))

    img_names = os.listdir(testFolder)
    img_paths = [os.path.join(testFolder, x) for x in img_names if checkImage(x)]
    with torch.no_grad():
        for i, img_path in enumerate(img_paths):
            img_hr = modcrop(load_image(img_path), 2)
            img_lr = upsample(img_hr, scale=2)
            img_hr = normalize(image_to_array(img_hr))
            img_lr = normalize(image_to_array(img_lr))

            img_hr_t = up_sampler(torch.from_numpy(img_hr).float().unsqueeze(0).cuda()).squeeze(0)

            if x4:
                img_hr_t = im2tensor(tensor2im(img_hr_t))
                img_hr_t = up_sampler(img_hr_t).squeeze(0)

            img_hr_t = img_hr_t.cpu().float().numpy()

            if show_flag:
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].imshow(np.transpose(img_lr, (1, 2, 0))), axs[0, 0].set_title("origin")
                axs[1, 0].imshow(np.transpose(img_hr, (1, 2, 0))), axs[0, 1].set_title("Bicubic")
                axs[1, 1].imshow(np.transpose(img_hr_t, (1, 2, 0))), axs[1, 1].set_title("origin2sr")
                plt.show()
            if save_image:
                save_origin_up = np.clip(np.round(np.transpose(img_hr_t, (1, 2, 0)) * 255), 0, 255)
                savert = os.path.join(os.path.join(os.getcwd(), outroot), img_names[i])
                Image.fromarray(np.uint8(save_origin_up)).save(savert)
            print("{}/{}".format(i+1, len(img_names)))


def SMD(img):
    if len(img.shape) > 2:
        img = np.dot(img, [0.299, 0.587, 0.114])
    shape = np.shape(img)
    clarity = 0
    for x in range(1, shape[0] - 1):
        for y in range(0, shape[1]):
            clarity += math.fabs(int(img[x, y]) - int(img[x, y-1]))
            clarity += math.fabs(int(img[x, y]) - int(img[x+1, y]))
    return clarity


def variance(img):
    clarity = 0.
    if len(img.shape) > 2:
        img = np.dot(img, [0.299, 0.587, 0.114])
    mean = np.mean(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            clarity += np.sqrt((img[i][j] - mean) ** 2)
    return clarity


def Laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


if __name__ == '__main__':
    test_DBPI(testFolder="test_imgs",
              statePath="weights\dbpi_up_sampler.pth",
              outroot="output",
              show_flag=False, save_image=True, x4=False)


