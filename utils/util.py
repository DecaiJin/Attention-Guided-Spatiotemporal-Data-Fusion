import os
import numpy as np
from osgeo import gdal
import argparse
os.environ['GTIFF_SRS_SOURCE'] = 'EPSG'
try:
    os.environ['PROJ_LIB'] = \
        'D:\ProgramFiles\Anaconda3\envs\pytorch\Library\share\proj'
except:
    pass


class ListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [float(x) for x in values.split(',')])


def write_img(im_data, save_path, auto_mkdir=True, **kwargs):

    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(save_path))
        os.makedirs(dir_name, exist_ok=True)

    projection = kwargs.get("projection", None)
    transform = kwargs.get("transform", None)

    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(save_path, im_width, im_height, im_bands, datatype)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    if projection is not None:
        dataset.SetProjection(projection)
    if transform is not None:
        dataset.SetGeoTransform(transform)

    dataset = None


def save_img(val_data, pred_img, output, iteration, suffix):
    """
    save c1, c2, f1, f2, pred image:
    c1 || c2
    f1 || f2
       || pre
    """
    img_name = val_data['name']
    pred_img = pred_img.detach().cpu().numpy()
    for i in range(len(img_name)):
        num_c, height, width = pred_img[i].shape[-3:]
        out_img = np.zeros((num_c, height*3+8, width*2+4))
        out_img[:, :height, :width] = val_data['c1'][i].detach().cpu().numpy()
        out_img[:, height+4: height*2+4, :width] = val_data['f1'][i].detach().cpu().numpy()
        out_img[:, :height, width+4: width*2+4] = val_data['c2'][i].detach().cpu().numpy()
        out_img[:, height+4: height*2+4, width+4: width*2+4] = val_data['f2'][i].detach().cpu().numpy()
        out_img[:, (height+4)*2: height*3+8, width+4: width*2+4] = pred_img[i]
        out_file = os.path.join(output, f'{iteration}_{img_name[i]}_{suffix}.tif')
        write_img(out_img, out_file)
        del out_img


def make_checkpoint_dir(path):
    model_path = os.path.join(path, 'models')
    visual_path = os.path.join(path, 'visual')

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(visual_path, exist_ok=True)


def imread(path):
    data = gdal.Open(path)
    array = data.ReadAsArray()
    trans = data.GetGeoTransform()
    proj = data.GetProjection()
    return array, trans, proj

