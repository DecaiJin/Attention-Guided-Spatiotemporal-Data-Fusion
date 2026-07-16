import glob
from osgeo import gdal_array
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from torchvision import transforms
from torch.nn.functional import interpolate


class ImageDataset(Dataset):
    def __init__(self, root, max_value, is_train=True, ):
        # Transforms for low resolution images and high resolution images
        self.f2 = sorted(sorted(glob.glob(os.path.join(root, 't2/hr_data/*.tif'))), key=len)
        self.c2 = sorted(sorted(glob.glob(os.path.join(root, 't2/lr_data/*.tif'))), key=len)
        self.f1 = sorted(sorted(glob.glob(os.path.join(root, 't1/hr_data/*.tif'))), key=len)
        self.c1 = sorted(sorted(glob.glob(os.path.join(root, 't1/lr_data/*.tif'))), key=len)
        self.is_train = is_train
        self.value_max = max_value

    def __getitem__(self, index):
        c1_path = self.c1[index % len(self.c1)]
        c2_path = self.c2[index % len(self.c2)]
        f1_path = self.f1[index % len(self.f1)]
        f2_path = self.f2[index % len(self.f2)]

        name = os.path.basename(c2_path)[:-4]
        c1 = torch.from_numpy(gdal_array.LoadFile(c1_path) / self.value_max)
        c2 = torch.from_numpy(gdal_array.LoadFile(c2_path) / self.value_max)
        f1 = torch.from_numpy(gdal_array.LoadFile(f1_path) / self.value_max)
        f2 = torch.from_numpy(gdal_array.LoadFile(f2_path) / self.value_max)
        # interpolate
        if c1.shape != f1.shape:
            c1 = interpolate(c1.unsqueeze(0), size=f1.size()[-2:], mode='nearest').squeeze()
            c2 = interpolate(c2.unsqueeze(0), size=f1.size()[-2:], mode='nearest').squeeze()
        if self.is_train:
            c1, c2, f1, f2 = [img for img in augment([c1, c2, f1, f2])]
            if np.random.rand() > 0.5:
                return {"c1": c1, "c2": c2, "f1": f1, "f2": f2, 'name': name}
            else:
                return {"c1": c2, "c2": c1, "f1": f2, "f2": f1, 'name': name}
        else:
            return {"c1": c1, "c2": c2, "f1": f1, "f2": f2, 'name': name}

    def __len__(self):
        return len(self.f2)


def augment(img_list):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])
    if np.random.randn() > 0.5:
        return [transform(img) for img in img_list]
    else:
        return img_list




