import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import measurements, interpolation


class Downsampler(nn.Module):
    def __init__(self, imageChannel: int = 1, numFeatures: int = 64, x4: bool = True):
        super(Downsampler, self).__init__()
        # struct = conf.D_structure
        stride = 0.5
        if x4:
            stride = 0.25
        else:
            stride = 0.5
        self.features = nn.Sequential(OrderedDict([
            ("stride_conv0",
             nn.Conv2d(in_channels=imageChannel, out_channels=numFeatures, kernel_size=(7, 7), padding=7 // 2,
                       stride=(int(1 / stride), int(1 / stride)), bias=False)),
            # ("stride_conv1", nn.Conv2d(in_channels=)),
            ("conv1", nn.Conv2d(in_channels=numFeatures, out_channels=numFeatures, kernel_size=(5, 5),
                                padding=5 // 2, stride=(1, 1), bias=False)),
            ("conv2", nn.Conv2d(in_channels=numFeatures, out_channels=numFeatures, kernel_size=(3, 3),
                                padding=3 // 2, stride=(1, 1), bias=False)),
            ("conv3", nn.Conv2d(in_channels=numFeatures, out_channels=numFeatures, kernel_size=(3, 3),
                                padding=3 // 2, stride=(1, 1), bias=False)),
            ("conv4", nn.Conv2d(in_channels=numFeatures, out_channels=numFeatures, kernel_size=(3, 3),
                                padding=3 // 2, stride=(1, 1), bias=False)),
            ("conv5", nn.Conv2d(in_channels=numFeatures, out_channels=numFeatures, kernel_size=(1, 1),
                                padding=0, stride=(1, 1), bias=False)),
        ]))

        self.features.add_module("outLayer", nn.Conv2d(in_channels=numFeatures, out_channels=imageChannel,
                                                       kernel_size=(1, 1), padding=0, stride=(1, 1), bias=False))
        self.final_act = nn.Tanh()
        self.downsaple = nn.AvgPool2d(kernel_size=4 if x4 else 2, ceil_mode=True)

        self.outputSize = self.forward(torch.FloatTensor(torch.ones([1, 1, 64, 64]))).shape[-1]
        self.forward_shave = int(64 * 4) - self.outputSize

    def forward(self, x):
        out = self.features(x.transpose(1, 0))
        output = out + self.downsaple(x.transpose(1, 0))
        return output.transpose(1, 0)


class Upsampler(nn.Module):
    def __init__(self, numFeatures: int = 64, numLayers: int = 7, imageChannel: int = 3, x4: bool = True):
        super(Upsampler, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(in_channels=imageChannel, out_channels=numFeatures, kernel_size=(3, 3),
                                stride=(1, 1), padding=1, bias=True))
        ]))
        for i in range(1, numLayers):
            self.features.add_module("conv%d" % i, nn.Conv2d(in_channels=numFeatures, out_channels=numFeatures,
                                                             kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True))
        self.features.add_module("outLayer", nn.Conv2d(in_channels=numFeatures, out_channels=imageChannel,
                                                       kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True))
        self.final_act = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=4 if x4 else 2, mode="bicubic")

    def forward(self, x):
        x = self.upsample(x)
        out = self.features(x)
        return x + out


def weights_init_G(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)


def weights_init_U(m):
    """ initialize weights of the upsampler  """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 1.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.)


def weights_init(m):
    """ initialize weights of the upsampler  """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 1.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.)


def zeroize_negligible_val(k, n):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    k_sorted = np.sort(k.flatten())
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[-n - 1]
    # Clip values lower than the minimum value
    filtered_k = np.clip(k - k_n_min, a_min=0, a_max=100)
    # Normalize to sum to 1
    return filtered_k / filtered_k.sum()


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in
             range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in
                                        range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


if __name__ == '__main__':
    down_sampler = Downsampler().cuda()
    up_sampler = Upsampler().cuda()
    down_sampler.load_state_dict(torch.load("DBPI/down_sampler Berea_Sandstone.pth"))
    up_sampler.load_state_dict(torch.load("DBPI/up_sampler Berea_Sandstone.pth"))
    curr_k = torch.FloatTensor(13, 13).cuda()
    delta = torch.Tensor([1., 1., 1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    for ind, w in enumerate(up_sampler.parameters()):
        if len(w.shape) > 1:
            print(w.shape)
            curr_k = F.conv2d(delta, w, padding=12) if ind == 0 else F.conv2d(curr_k, w)

    # curr_k = curr_k.squeeze().flip([0, 1])
    # curr_k = curr_k.detach().cpu().float().numpy()
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(curr_k, cmap="gray")
    # curr_k = np.clip(np.round(curr_k * 255.0), 0, 255)
    # axs[1].imshow(curr_k, cmap="gray")
    # significant_k = zeroize_negligible_val(curr_k, 40)
    # # Force centralization on the kernel
    # centralized_k = kernel_shift(significant_k, sf=2)
    # axs[1].imshow(centralized_k, cmap="gray")
    image = np.transpose(np.array(Image.open("data/test/top view146.bmp").convert("RGB")) / 255., (2, 1, 0))
    image_t = torch.from_numpy(image).float().unsqueeze(0).cuda()

    out = F.conv2d(image_t, curr_k).squeeze(0).squeeze(0)

    out = out.detach().cpu().numpy()
    # out = np.transpose(out, (1, 2, 0))
    plt.imshow(out, cmap="gray")
    # Image.fromarray(np.uint8(curr_k)).save("test.png")

    plt.show()
    print(curr_k)
