import random
import torch
import gdal
import numpy as np
from scipy import special
from torch.nn import functional as F
from degradations import random_add_poisson_noise_pt, random_add_gaussian_noise_pt, random_mixed_kernels


class Transform(object):

    def __init__(self):
        self.scale = 1/50
        self.kernel_list = ('iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso')
        self.kernel_prob = (0.45, 0.25, 0.12, 0.03, 0.12, 0.03)
        self.sinc_prob = 0.1
        self.sinc_prob2 = 0.1
        self.blur_sigma = (0.2, 3)
        self.blur_sigma2= (0.2, 1.5)
        self.betag_range = [0.0005, 0.004]
        self.betap_range = [0.001, 0.002]
        self.gray_noise_prob = 0
        self.gaussian_noise_prob = 0.2
        self.noise_range = [0.001 , 0.002]
        self.noise_range2 = [0.001, 0.0025]
        self.poisson_scale_range = [0.0001, 0.005]
        self.poisson_scale_range2 = [0.0002, 0.008]
        self.second_blur_prob = 0.8
        self.gaussian_noise_prob2 = 0.2
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.final_sinc_prob = 0.8
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    def __call__(self, gt):
        bands = [0, 1, 2, 6]
        gt = gt[:, bands, :, :]
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-np.pi, np.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-np.pi, np.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # BGR to RGB, HWC to CHW, numpy to tensor
        k1 = torch.FloatTensor(kernel)
        k2 = torch.FloatTensor(kernel2)

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(gt, k1)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=self.gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False)
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1]

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, k2)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=self.scale, mode=mode)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False)
        out = F.interpolate(out, scale_factor=1/self.scale, mode='nearest').squeeze()
        return out


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter

    Reference: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter

    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def write_tif(img, path):
    C, H, W = img.shape
    driver = gdal.GetDriverByName('GTiff')   # 选择驱动
    dataset = driver.Create(path, W, H, C, gdal.GDT_Float32)   # 创建文件
    for i in range(C):
        dataset.GetRasterBand(i+1).WriteArray(img[i])   # 写入波段i+1
    dataset.FlushCache()   # 刷新缓存
    dataset = None   # 关闭文件


if __name__ == '__main__':
    from osgeo import gdal_array
    img = gdal_array.LoadFile(r"D:\LeStoreDownload\Common\Henan_t1_min-0000037632-0000012544.tif")#[:, :1600, :1600]
    print(img[0, -10:, -10:])
    # write_tif(np.array(img), r'D:\RSData\test0.tif')
    # img = np.array([img])
    # T = Transform()
    # img = T(img)
    # write_tif(np.array(img[0]), r'D:\RSData\test.tif')


