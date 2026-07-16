from .agsdf_arch import AGSDF
from .discriminator_arch import UNetDiscriminatorSN
import torch
import os.path as osp


def build_model(opt):
    is_load = False
    args = {
        "in_channels": opt.in_channels,
        "out_channels": opt.out_channels,
        "kernel_size": opt.kernel_size,
        "num_feat": opt.num_features,
        "num_RRDBs": opt.num_rrdb,
        "num_msv": opt.num_msv
    }
    model = AGSDF(**args)
    # load weight
    if osp.exists(opt.state) and osp.isfile(opt.state):
        model.load_state_dict(
            torch.load(opt.state, map_location=opt.device), strict=True
        )
        is_load = True
    return model, is_load


def build_discriminator(opt):
    is_load = False
    model = UNetDiscriminatorSN(opt.in_channels)
    net_d_weight = osp.join(osp.dirname(opt.state), 'net_d_' + osp.basename(opt.state))
    if osp.exists(net_d_weight) and osp.isfile(net_d_weight):
        model.load_state_dict(
            torch.load(net_d_weight, map_location=opt.device), strict=True
        )
        is_load = True
    return model, is_load

