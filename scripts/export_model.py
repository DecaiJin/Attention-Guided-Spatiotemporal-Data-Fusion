import sys
import os
import subprocess
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import torch
import argparse
from model import build_model
import openvino.runtime as ov


def model_test(opt, data_dict):
    name = os.path.basename(opt.state)[:-4]
    model_xml = f'{opt.output}/{name}.xml'
    model_bin = f'{opt.output}/{name}.bin'
    core = ov.Core()

    # 加载模型
    net = core.read_model(model=model_xml, weights=model_bin)
    # 创建推理引擎
    compiled_model = core.compile_model(net, 'CPU')
    infer_request = compiled_model.create_infer_request()
    infer_request.infer(inputs=data_dict)

    # here suppose it only have one output.
    output_tensor = infer_request.get_output_tensor()
    print(f"output tensor size:{output_tensor.shape}")
    print("model  test success")


def export(model, data, output, model_name):
    model.eval()
    onnx_filename = os.path.join(output, f'{model_name}.onnx')
    torch.onnx.export(model, data, onnx_filename,
                      input_names=['c1', 'f1', 'c2'], opset_version=12)
    command = [
        "mo", "--input_model", onnx_filename, "--output_dir", output, "--compress_to_fp16", "True"
    ]
    try:
        subprocess.run(command, check=True, shell=True)
        print("export success！")
    except subprocess.CalledProcessError as e:
        print("export failed:", e)


def main():
    # python ./scripts/export_model.py -s agsdf_e200_chn3.pth -o ./scripts --in_channels 3 --out_channels 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", '-s', type=str, required=True, help="path of converted model")
    parser.add_argument("--output", '-o', type=str, required=True, help="path of converted model")
    parser.add_argument("--patch_size", type=int, default=512, help="size of sub-image in predicted process")

    parser.add_argument("--device", type=str, default='cpu', help="running device")
    parser.add_argument("--in_channels", type=int, default=3, help="number of input channels")
    parser.add_argument("--out_channels", type=int, default=3, help="number of output channels")
    parser.add_argument("--num_features", type=int, default=32, help="feature numbers")
    parser.add_argument('--kernel_size', type=int, default=3, help='size of spatiotemporal kernel')
    parser.add_argument('--num_rrdb', type=int, default=8, help='number of RRDB')
    parser.add_argument('--num_msv', type=int, default=3, help='depth of multi scale volume')
    opt = parser.parse_args()

    # load model
    model, _ = build_model(opt)

    # set input data
    c1 = torch.rand(1, opt.in_channels, opt.patch_size, opt.patch_size)
    c2 = torch.rand(1, opt.in_channels, opt.patch_size, opt.patch_size)
    f1 = torch.rand(1, opt.in_channels, opt.patch_size, opt.patch_size)

    input_data = {
        'c1': c1,
        'f1': f1,
        'c2': c2,
    }
    model_name = os.path.basename(opt.state).split('.')[0]
    export(model, input_data, opt.output, model_name)
    model_test(opt, input_data)


if __name__ == "__main__":
    main()
    # create numpy array with size 5*5

