import os
import argparse
import torch
from tqdm import tqdm
from model import build_model
from data import make_dataset
from utils.util import write_img
from osgeo import gdal_array


def parser_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", '-t', type=str, default='./', help="path of inputs")
    parser.add_argument("--state", '-w', type=str, default='./', required=True, help="weight path")
    parser.add_argument("--device", type=str, default='cuda', help="running device")
    # dataloader param
    parser.add_argument("--max_value", type=int, default=1, help="max value of reflectance")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--workers", type=int, default=0, help="workers of dataloader")
    # architecture param
    parser.add_argument("--in_channels", type=int, default=3, help="number of input channels")
    parser.add_argument("--out_channels", type=int, default=3, help="number of output channels")
    parser.add_argument("--num_features", type=int, default=32, help="feature numbers")
    parser.add_argument('--kernel_size', type=int, default=3, help='size of spatiotemporal kernel')
    parser.add_argument('--num_rrdb', type=int, default=8, help='number of RRDB')
    parser.add_argument('--num_msv', type=int, default=3, help='depth of multi scale volume')
    # result path
    parser.add_argument("--result_dir", '-r', type=str, default="./result", help="path of results")

    opt = parser.parse_args()
    opt.is_train = False
    return opt


def test_pipeline():
    opt = parser_option()
    device = torch.device(opt.device)
    os.makedirs(opt.result_dir, exist_ok=True)  # path dir
    model = build_model(opt).to(device)
    # dataloader = make_dataset(opt, is_val=True).to(device)
    dataloader = set_val_dataset(opt.test_path, opt.max_value)
    model.eval()
    with tqdm(total=len(dataloader), bar_format='{l_bar}{bar:20}{r_bar}') as pbar:
        for data in dataloader:
            c1 = data["c1"].float().to(device)
            c2 = data["c2"].float().to(device)
            f1 = data["f1"].float().to(device)
            with torch.no_grad():
                output = model(c1, f1, c2)
                output = output.cpu().detach().numpy()
                pbar.update(1)
                for n in range(output.shape[0]):
                    im = output[n]
                    out_name = data['name'][n]
                    write_img(im, f"{opt.result_dir}/{out_name}.tif")


def set_val_dataset(path_test, max_value):
    n1, n2 = 3, 5   # 1, 3/ 2, 4/ 3, 5
    dataloader = [{
        'c1': torch.from_numpy(gdal_array.LoadFile(os.path.join(path_test, f'M{n1}'))[1:4]/max_value).unsqueeze(0),
        'c2': torch.from_numpy(gdal_array.LoadFile(os.path.join(path_test, f'M{n2}'))[1:4]/max_value).unsqueeze(0),
        'f1': torch.from_numpy(gdal_array.LoadFile(os.path.join(path_test, f'L{n1}'))[1:4]/max_value).unsqueeze(0),
        'f2': torch.from_numpy(gdal_array.LoadFile(os.path.join(path_test, f'L{n2}'))[1:4]/max_value).unsqueeze(0),
        'name': [os.path.basename(path_test)+'_L{}_'.format(n2)+'fine.tif']
    }]
    return dataloader


if __name__ == "__main__":
    test_pipeline()
