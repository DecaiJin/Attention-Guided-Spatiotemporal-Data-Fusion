# -*- coding: utf-8 -*-
import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, inputs):
        return inputs * self.scale


class InvPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(InvPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).\
            permute(0, 1, 3, 5, 2, 4).contiguous().view(b, -1, y // ratio, x // ratio)


class ConvBNReLU2D(nn.Module):
    def __init__(self, in_c, out_c, ksize=3, stride=1, normal=True, activate=True, bias=True,
                 is_padding=True):
        super(ConvBNReLU2D, self).__init__()
        pad = ksize // 2 if is_padding else 0
        self.conv = nn.Conv2d(in_c, out_c, ksize, stride, pad, bias=bias)
        self.norm = nn.BatchNorm2d(out_c) if normal else None
        self.act = nn.PReLU() if activate else None

    def forward(self, *inputs):
        if len(inputs) == 1:
            out = self.conv(inputs[0])
        else:
            out = self.conv(inputs[0], inputs[1])
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_feat, normal=None, activate=None, bias=None):
        super(ResNet, self).__init__()
        self.conv1 = ConvBNReLU2D(in_c=num_feat, out_c=num_feat,
                                  normal=normal, activate=activate, bias=bias)
        self.conv2 = ConvBNReLU2D(in_c=num_feat, out_c=num_feat,
                                  normal=normal, activate=activate, bias=bias)
        self.relu = nn.PReLU()

    def forward(self, input_feature):
        ipt = self.conv1(input_feature)
        ipt = self.conv2(ipt)
        return self.relu(ipt + input_feature)


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class DownSample(nn.Module):
    def __init__(self, num_feat, using_ips=True):
        super(DownSample, self).__init__()
        scale = 2
        if using_ips:
            self.layers = nn.Sequential(
                ConvBNReLU2D(in_c=num_feat, out_c=num_feat, is_padding=True),
                ConvBNReLU2D(in_c=num_feat, out_c=num_feat, stride=2, is_padding=True),
            )
        else:
            self.layers = nn.Sequential(
                ConvBNReLU2D(in_c=num_feat, out_c=num_feat, ksize=3),
                InvPixelShuffle(ratio=scale),
                ConvBNReLU2D(in_c=num_feat * scale ** 2, out_c=num_feat, ksize=1),
            )

    def forward(self, inputs):
        return self.layers(inputs)


if __name__ == "__main__":
    model = nn.Conv2d(6, 6, 3, 2, 1)
    x = torch.rand(5, 6, 397, 397)
    y = model(x)
    y = model(y)
    print(397 / 4, 397//4)
    print(y.shape)

