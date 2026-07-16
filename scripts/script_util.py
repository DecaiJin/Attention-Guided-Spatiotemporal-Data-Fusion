import os
import sys
import csv
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import write_img, imread


def clip(img, args, file, meta, trans, proj):
    for row in range(0, img.shape[1], args.stride):
        for col in range(0, img.shape[2], args.stride):
            temp = np.array([img[chn, row:row+args.size, col:col+args.size]
                             for chn in range(0, img.shape[0], 1)])
            if temp.shape[1] != args.size or temp.shape[2] != args.size:
                # image padding with zero
                temp = np.pad(temp, pad_width=((0, 0), (0, args.size-temp.shape[1]),
                                               (0, args.size-temp.shape[2])), mode='constant')
            left_x = trans[0] + col*trans[1] + row*trans[2]
            left_y = trans[3] + col*trans[4] + row*trans[5]
            trans = [left_x, trans[1], trans[2], left_y, trans[4], trans[5]]
            if np.all(np.isnan(temp)):
                continue
            if meta is not None:
                for key in meta.keys():
                    if file in meta[key]:
                        pair_num = list(meta[key]).index(file)  # np.where(meta[key] == file)[0]
                        file_name = f"{args.out_prefix}_p{pair_num}_{row}_{col}_{img.shape[1]}_{img.shape[2]}.tif"
                        write_img(np.array(temp), os.path.join(args.output, key, file_name),
                                  transform=trans, projection=proj)
            else:  # to single pair
                file_name = f"{args.out_prefix}_{row}_{col}_{img.shape[1]}_{img.shape[2]}.tif"
                write_img(np.array(temp), os.path.join(args.output, file_name),
                          transform=trans, projection=proj)


def load_csv(path):
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        data = np.array([row for row in csv_reader])
    meta_info = {}
    for i in range(data.shape[1]):
        meta_info[data[0, i]] = data[1:, i]
    return meta_info


if __name__ == "__main__":
    meta_dict = load_csv(r"F:\Works\2_DataFusion\1_Data\Landsat_Modis_CIA\test.csv")
    print(meta_dict)
