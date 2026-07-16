import torch
import torch.nn as nn
from torch.nn.functional import interpolate, softmax
from .arch_util import ConvBNReLU2D, Scale, ResidualDenseBlock


class InitLayer(nn.Module):
    def __init__(self, in_channels, num_features, flag=0):
        super(InitLayer, self).__init__()

        self.flag = flag
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.PReLU()
        )
        if flag == 0:
            self.layer2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        else:
            self.layer2 = nn.Conv2d(2 * num_features, num_features, 3, 1, 1)
        self.relu = nn.PReLU()

    def forward(self, inputs, lr):
        out = self.layer1(inputs)
        if self.flag == 0:
            out = self.relu(self.layer2(out))
        else:
            out = self.relu(self.layer2(torch.cat((out, lr), dim=1)))
        return out


class FeatureEx(nn.Module):
    def __init__(self, in_channels, num_feat, expand_ratio=1):
        super(FeatureEx, self).__init__()
        self.lr_head = nn.Sequential(
            ConvBNReLU2D(in_c=in_channels, out_c=num_feat * expand_ratio, ksize=3),
            ConvBNReLU2D(in_c=num_feat * expand_ratio, out_c=num_feat, ksize=1)
        )

        self.ref_head = nn.Sequential(
            ConvBNReLU2D(in_c=in_channels, out_c=num_feat * expand_ratio, ksize=3),
            ConvBNReLU2D(in_c=num_feat * expand_ratio, out_c=num_feat, ksize=1)
        )

        self.ref_du_head = nn.Sequential(
            ConvBNReLU2D(in_c=in_channels, out_c=num_feat * expand_ratio, ksize=3),
            ConvBNReLU2D(in_c=num_feat * expand_ratio, out_c=num_feat, ksize=1)
        )

    def forward(self, lr_img, ref_img, ref_du=None):
        if ref_du is not None:
            return self.lr_head(lr_img), self.ref_head(ref_img), self.ref_du_head(ref_du)
        else:
            return self.lr_head(lr_img), self.ref_head(ref_img)


class OutLayer(nn.Module):
    def __init__(self, num_features, out_channels):
        super(OutLayer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.res_scale = Scale(0)

    def forward(self, inputs, add_feature):
        out = self.layer1(inputs)
        if add_feature is not None:
            add_feature = self.res_scale(interpolate(add_feature, scale_factor=2, mode='nearest'))
            out = self.layer3(add_feature + out)
        return out, self.layer2(out)


class FusionKernel(nn.Module):
    def __init__(self, kernel_size, num_feat, num_RRDBs, scale=2):
        super(FusionKernel, self).__init__()

        self.scale = scale
        self.num = kernel_size**2
        self.aff_scale_const = nn.Parameter(0.5 * self.num * torch.ones(1))

        self.lr_kernel = nn.Sequential(
            ConvBNReLU2D(in_c=num_feat, out_c=num_feat, ksize=1),
            ConvBNReLU2D(in_c=num_feat, out_c=self.num, ksize=1)
        )

        self.ref_kernel = nn.Sequential(
            ConvBNReLU2D(in_c=num_feat, out_c=num_feat, ksize=1),
            ConvBNReLU2D(in_c=num_feat, out_c=self.num, ksize=1)
        )

        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=scale, padding=kernel_size // 2 * scale)
        self.sr_conv = nn.Sequential(*[
            ResidualDenseBlock(nf=num_feat, gc=num_feat) for _ in range(num_RRDBs)])

    def forward(self, weight_map, lr_fea, ref_fea, inputs):
        b, c, h, w = inputs.size()
        h_, w_ = lr_fea.size()[2:]

        lr_kernel = self.lr_kernel(lr_fea)
        ref_kernel = self.ref_kernel(ref_fea)

        lr_kernel = softmax(lr_kernel, dim=1)
        ref_kernel = softmax(ref_kernel, dim=1)

        fuse_kernel = (1 - weight_map) * lr_kernel + weight_map * ref_kernel
        fuse_kernel = torch.tanh(fuse_kernel) / (self.aff_scale_const + 1e-8)
        abs_kernel = torch.abs(fuse_kernel)
        abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
        abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
        fuse_kernel = fuse_kernel / abs_kernel_sum

        inputs_up = interpolate(self.sr_conv(inputs), size=(h_, w_), mode='nearest')
        unfold_inputs = self.unfold(inputs_up).view(b, c, -1, h_, w_)
        output = torch.einsum('bkhw, bckhw->bchw', [fuse_kernel, unfold_inputs])
        return output
