import cv2
import h5py
import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


def move2cpu(d):
    return d.detach().cpu().float().numpy()


def tensor2im(imgTensor):
    img_np = np.clip(np.round((np.transpose(move2cpu(imgTensor), (1, 2, 0))) * 255.0), 0, 255)
    # img_np = np.squeeze(img_np, axis=2) if img_np.shape[2] == 1 else img_np
    return img_np.astype(np.uint8)


def im2tensor(im_np):
    im_np = im_np / 255.0 if im_np.dtype == "uint8" else im_np
    # b = len(im_np.shape)
    # im_np = np.expand_dims(im_np, axis=2) if len(im_np.shape) == 2 else im_np
    return torch.FloatTensor(np.transpose(im_np, (2, 0, 1))).unsqueeze(0).cuda()


def load_image(img_path):
    return Image.open(img_path).convert("RGB")


def modcrop(img, factor):
    w = img.width - img.width % factor
    h = img.height - img.height % factor
    return img.crop((0, 0, w, h))


def generatePatch(image, patch_size, stride):
    for i in range(0, image.height - patch_size + 1, stride):
        for j in range(0, image.width - patch_size + 1, stride):
            yield image.crop((j, i, j + patch_size, i + patch_size))


def image_to_array(image):
    return np.array(image).transpose((2, 0, 1))


def normalize(array):
    return array / 255.0


def stdNormalize(array):
    return (array - array.mean()) / array.std()


def downsample(image, scale):
    image = image.resize((image.width // scale, image.height // scale), Image.BICUBIC)
    return image


def upsample(image, scale):
    image = image.resize((image.width * scale, image.height * scale), Image.BICUBIC)
    return image


def centerCrop(image, percent_height: float = 0.35, percent_width: float = 0.4, show_flag: bool = True) -> np.array:
    image = np.array(image)
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]
    center = [height // 2, width // 2]
    # height_crop = int(height * percent_height) // 2
    # width_crop = int(width * percent_width) // 2
    circle = np.zeros(image.shape[0:2], dtype=np.uint8)
    circle = cv2.circle(circle, (center[0], center[1]), 200, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=circle)
    crop = masked[(center[0] - 205):(center[0] + 205), (center[1] - 205):(center[1] + 205), :]
    # crop = image[(center[0] - height_crop):(center[0] + height_crop), (center[1] -
    #                                                                    width_crop):(center[1] + width_crop), :]
    # crop = image[(center[0] - 225):(center[0] + 225), (center[1] - 225):(center[1] + 225), :]
    # plt.imshow(crop)
    # plt.show()
    return crop


def saveCrop(imageFolder: str, saveFolder: str):
    img_names = [name for name in os.listdir(imageFolder) if checkImage(name)]
    img_paths = [os.path.join(imageFolder, x) for x in img_names if checkImage(x)]
    with tqdm(total=len(img_paths), ncols=120) as t:
        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path)
            img_crop = centerCrop(img)
            Image.fromarray(img_crop).save(os.path.join(saveFolder, img_names[i]))
            t.update(i)


def checkImage(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".BMP",
                                                              ".png", ".PNG",
                                                              ".jpg", ".JPG"])


class GenerateDataSet:
    def __init__(self):
        super(GenerateDataSet, self).__init__()

    @staticmethod
    def generateDataSetBySample(imageDir: str, outPath: str, scaleFactor: int, patchSize: int = 96, stride: int = 96):
        hr_patches = []
        lr_patches = []
        num_lr_patches = 0
        pre_num_lr_patches = 0
        num_hr_patches = 0
        pre_num_hr_patches = 0
        h5_file = h5py.File(outPath, 'a')
        lr_dataset = h5_file.create_dataset("lr", [1, 3, patchSize // scaleFactor, patchSize // scaleFactor],
                                            maxshape=[None, 3, patchSize // scaleFactor, patchSize // scaleFactor],
                                            chunks=True, compression="gzip", compression_opts=7)
        hr_dataset = h5_file.create_dataset("hr", [1, 3, patchSize, patchSize],
                                            maxshape=[None, 3, patchSize, patchSize],
                                            chunks=True, compression="gzip", compression_opts=7)
        for imagedir in os.listdir(imageDir):
            for i, img_path in enumerate(sorted(glob.glob("{}/*".format(imagedir)))):
                hr_img = load_image(img_path)
                hr_img = modcrop(hr_img, scaleFactor)
                lr_img = downsample(hr_img, scaleFactor)

                for patch in generatePatch(hr_img, patchSize, stride):
                    patch = image_to_array(patch)
                    # patch = np.expand_dims(normalize(patch.astype(np.float32)), 0)
                    patch = normalize(patch.astype(np.float32))
                    hr_patches.append(patch)

                for patch in generatePatch(lr_img, patchSize // scaleFactor, stride // scaleFactor):
                    patch = image_to_array(patch)
                    # patch = np.expand_dims(normalize(patch.astype(np.float32)), 0)
                    patch = normalize(patch.astype(np.float32))
                    lr_patches.append(patch)
                # if i > 50:
                #     break
                if (i + 1) % 2 == 0:
                    print("----------------creating h5file-------------------")
                    pre_num_lr_patches = num_lr_patches
                    num_lr_patches += len(lr_patches)
                    lr_dataset.resize((num_lr_patches, 3, patchSize // scaleFactor, patchSize // scaleFactor))
                    lr_dataset[pre_num_lr_patches:num_lr_patches] = np.array(lr_patches)
                    lr_patches.clear()

                    pre_num_hr_patches = num_hr_patches
                    num_hr_patches += len(hr_patches)
                    hr_dataset.resize((num_hr_patches, 3, patchSize, patchSize))
                    hr_dataset[pre_num_hr_patches:num_hr_patches] = np.array(hr_patches)
                    hr_patches.clear()

                print("generating patchs... -> image index: {}, patchs: {}, patchSize: {}".format(
                    i + 1, len(hr_patches), patchSize))

        # h5_file.create_dataset(name="lr", data=np.array(lr_patches))
        # h5_file.create_dataset(name="hr", data=np.array(hr_patches))
        h5_file.close()
        print("done")

    @staticmethod
    def generateDatasetFromFolder(self, HrDir: str, LrDir: str, outPath: str, scaleFactor: int,
                                  patchSize: int = 0, stride: int = 0, sample: bool = False):
        hr_patches = []
        lr_patches = []
        num_lr_patches = 0
        pre_num_lr_patches = 0
        num_hr_patches = 0
        pre_num_hr_patches = 0
        h5_file = h5py.File(outPath, 'a')
        lr_dataset = h5_file.create_dataset("lr", [1, 3, patchSize // scaleFactor, patchSize // scaleFactor],
                                            maxshape=[None, 3, patchSize // scaleFactor, patchSize // scaleFactor],
                                            chunks=True, compression="gzip", compression_opts=7)
        hr_dataset = h5_file.create_dataset("hr", [1, 3, patchSize, patchSize],
                                            maxshape=[None, 3, patchSize, patchSize],
                                            chunks=True, compression="gzip", compression_opts=7)
        for i, img_path in enumerate(glob.glob("{}/*".format(HrDir))):
            hr_img = load_image(img_path)
            if patchSize and stride:
                hr_img = modcrop(hr_img, scaleFactor)
                for patch in generatePatch(hr_img, patchSize, stride):
                    patch = image_to_array(patch)
                    patch = normalize(patch.astype(np.float32))
                    hr_patches.append(patch)
            else:
                hr_img = image_to_array(hr_img)
                hr_img = normalize(hr_img.astype(np.float32))
                hr_patches.append(hr_img)
            print("generating patchs... -> image index: {}, patchs: {}".format(i + 1, len(hr_patches)))
            if (i + 1) % 50 == 0:
                print("generating h5file")
                pre_num_hr_patches = num_hr_patches
                num_hr_patches += len(hr_patches)
                hr_dataset.resize((num_hr_patches, 3, patchSize, patchSize))
                hr_dataset[pre_num_hr_patches:num_hr_patches] = np.array(hr_patches)
                hr_patches.clear()

        for i, img_path in enumerate(glob.glob("{}/*".format(LrDir))):
            lr_img = load_image(img_path)
            if sample:
                lr_img = upsample(lr_img, scaleFactor)
                if patchSize and stride:
                    for patch in generatePatch(lr_img, patchSize, stride):
                        patch = image_to_array(patch)
                        patch = normalize(patch.astype(np.float32))
                        lr_patches.append(patch)
                else:
                    lr_img = image_to_array(lr_img)
                    lr_img = normalize(lr_img)
                    lr_patches.append(lr_img)
            else:
                if patchSize and stride:
                    for patch in generatePatch(lr_img, patchSize // scaleFactor, stride // scaleFactor):
                        patch = image_to_array(patch)
                        patch = normalize(patch.astype(np.float32))
                        lr_patches.append(patch)
                else:
                    lr_img = image_to_array(lr_img)
                    lr_img = normalize(lr_img.astype(np.float32))
                    lr_patches.append(lr_img)
            print("generating patchs... -> image index: {}, patchs: {}".format(i + 1, len(lr_patches)))
            if (i + 1) % 50 == 0:
                print("generating h5file")
                pre_num_lr_patches = num_lr_patches
                num_lr_patches += len(lr_patches)
                lr_dataset.resize((num_lr_patches, 3, patchSize // scaleFactor, patchSize // scaleFactor))
                lr_dataset[pre_num_lr_patches:num_lr_patches] = np.array(lr_patches)
                lr_patches.clear()

        h5_file.close()


def make_1ch(im):
    s = im.shape
    assert s[1] == 3
    return im.reshape(s[0] * 3, 1, s[2], s[3])


def make_3ch(im):
    s = im.shape
    assert s[1] == 1
    return im.reshape(s[0] // 3, 3, s[2], s[3])


roots = ["F:\\Personal workshop of CL\\pycharm\\projects\\Datasets\\2021-08 "
         "To Chen L\\hydrate/val/6-6/origin",
         "F:\\Personal workshop of CL\\pycharm\\projects\\Datasets\\2021-08 "
         "To Chen L\\hydrate/val/delete words/6-6/origin",
         "F:/Personal workshop of CL/pycharm/projects/Datasets/2022-04-29 To Chen L/20170504 Berea_Sandstone",
         "F:/Personal workshop of CL/pycharm/projects/Datasets/2022-04-29 To Chen L/20220606 Berea_Sandstone",
         "F:/Personal workshop of CL/pycharm/projects/Datasets/"
         ]


if __name__ == '__main__':
    pass
