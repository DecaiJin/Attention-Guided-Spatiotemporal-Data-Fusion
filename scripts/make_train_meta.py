import os
import random
import copy
import pandas as pd


def make_meta(path, meta_path, fine_prefix="FINE", coarse_prefix='COARSE'):
    """
    Distribute collected coarse and fine image to t1 and t2 training pair
    -folder
        -COARSE_xx_YYYYMMDD.tif
        -FINE_xx_YYYYMMDD.tif
        ...
    -output: the training meta containing multi-sub training images used to clip images
    """
    t1_meta = dict(hr_data=[], lr_data=[])
    train_meta = {}
    files = list(sorted(os.listdir(path)))
    for f in files:
        assert len(f.split('_')) >= 2, "img name format must be 'PREFIX_YYYYMMDD'"
        sensor_name = f.split('_')[0]
        if sensor_name not in t1_meta:
            if sensor_name == fine_prefix:
                t1_meta["hr_data"].append(f)
            elif sensor_name == coarse_prefix:
                t1_meta["lr_data"].append(f)
            else:
                raise Exception("{} must be inferred from input fine or coarse prefix".format(sensor_name))
    t2_meta = copy.deepcopy(t1_meta)
    for key in t2_meta.keys():
        random.seed(123)
        random.shuffle(t2_meta[key])

        train_meta["t1/{}".format(key)] = t1_meta[key]
        train_meta["t2/{}".format(key)] = t2_meta[key]
    df = pd.DataFrame(train_meta)
    df.to_csv(meta_path, index=False)
    return train_meta


if __name__ == "__main__":
    data_path = r'F:\Works\2_DataFusion\1_Data\MDJ\GEE'
    meta_file = r'F:\Works\2_DataFusion\1_Data\MDJ\paired_meta.csv'
    make_meta(data_path, meta_file, fine_prefix='landsat', coarse_prefix='modis')