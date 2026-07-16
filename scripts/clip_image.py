import argparse
from make_train_meta import make_meta
from script_util import *


def main():
    """
    python scripts/clip_image.py --input F:\ --output F:\ -ifp landsat -icp modis --out_prefix train
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./', required=True,
                        help='folder that contains all fine and coarse images')
    parser.add_argument('--output', type=str, default='./dataset/', help='output folder used save trained sub-img')
    parser.add_argument('--stride', type=int, default=240, help='clipping stride')
    parser.add_argument('--size', type=int, default=256, help='size of sub-img')
    parser.add_argument('--input_coarse_prefix', '-icp', type=str, default='COARSE',
                        help='the prefix of the input coarse image')
    parser.add_argument('--input_fine_prefix', '-ifp', type=str, default='FINE',
                        help='the prefix of the input fine image')
    parser.add_argument('--out_prefix', type=str, default='train', help='the prefix of the output image')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    # if input is a folder and contians many paired images(fine and coarse images);
    if os.path.isdir(args.input):
        # build random paired meta information;
        args.meta = 'meta_info'
        output = os.path.join(args.output, f'{args.meta}.csv')
        meta = make_meta(args.input, output, args.input_fine_prefix, args.input_coarse_prefix)
        for key in meta.keys():
            os.makedirs(os.path.join(args.output, key), exist_ok=True)
        # clip all images according to meta information;
        for file in os.listdir(args.input):
            print(f'-processing {file}')
            fp = os.path.join(args.input, file)
            im_data, trans, proj = imread(fp)
            clip(im_data, args, file, meta, trans, proj)
            del im_data
    # if input is a single image
    else:
        meta = None
        os.makedirs(args.output, exist_ok=True)
        print(f'-processing {os.path.basename(args.input)}')
        folder_name = os.path.basename(args.input).split('.')[0]
        im_data, trans, proj = imread(args.input)
        clip(im_data, args, folder_name, meta, trans, proj)
        del im_data


if __name__ == "__main__":
    main()
