# code reference https://github.com/zhwzhong/DAGF
# doi:10.1109/TNNLS.2023.3253472
import torch
from torch import nn
from torch.nn.functional import interpolate
from model.arch_util import DownSample
from model.agsdf_util import InitLayer, FeatureEx, FusionKernel, OutLayer
from guided_filter_pytorch.guided_filter import FastGuidedFilter


class AGSDF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_feat, num_RRDBs, num_msv,
                 using_shuffle=True):
        super(AGSDF, self).__init__()
        self.num_msv = num_msv
        self.guide_filter = FastGuidedFilter(r=16)

        self.FEx = FeatureEx(in_channels, num_feat, expand_ratio=1)
        self.guide_up = nn.ModuleList([FusionKernel(kernel_size, num_feat, num_RRDBs, scale=2)
                                       for _ in range(num_msv)])
        self.c2_down = nn.ModuleList([DownSample(num_feat, using_ips=using_shuffle)
                                      for _ in range(num_msv - 1)])
        self.f1_down = nn.ModuleList([DownSample(num_feat, using_ips=using_shuffle)
                                      for _ in range(num_msv - 1)])
        self.init_conv = nn.ModuleList([InitLayer(in_channels, num_feat, flag=i)
                                        for i in range(num_msv)])
        self.out_conv = nn.ModuleList([OutLayer(num_feat, out_channels)
                                       for _ in range(num_msv)])

    def forward(self, c1, f1, c2):
        # ======================Empowering variation attention====================
        f2_g = self.guide_filter(c1, c2, f1)
        v_att = torch.sigmoid(torch.sum((f2_g - f1).abs(), dim=1, keepdim=True))
        # ============================multi scale volume==========================
        c2_feat, f1_feat = self.FEx(c2, f1)
        c2_msv, f1_msv, att_msv = [c2_feat], [f1_feat], [v_att]
        for num_s in range(self.num_msv - 1):
            c2_feat = self.c2_down[num_s](c2_feat)
            c2_msv.append(c2_feat)
            f1_feat = self.f1_down[num_s](f1_feat)
            f1_msv.append(f1_feat)
            v_att = interpolate(v_att, size=f1_feat.size()[-2:], mode='bilinear', align_corners=False)
            att_msv.append(v_att)
        c2_msv = list(reversed(c2_msv))
        f1_msv = list(reversed(f1_msv))
        att_msv = list(reversed(att_msv))
        # ===========multi scale stf: fusion kernel; guided up-sampling===========
        lr_input = None
        sr_feature = []
        for i in range(self.num_msv):
            h, w = c2_msv[i].size()[-2:]     # (64, 128, 256, 512)
            lr_input = self.init_conv[i](interpolate(c2, size=(h // 2, w // 2), mode='bilinear', align_corners=False),
                                         lr_input)
            lr_input = self.guide_up[i](att_msv[i], c2_msv[i], f1_msv[i], lr_input)
            sr_feature.append(lr_input)
        # ===========================adjust output channels=======================
        out = []
        out1, out2 = None, None
        for i in range(self.num_msv):
            c2 = interpolate(c2, size=sr_feature[i].size()[-2:], mode='bilinear', align_corners=False)
            out1, out2 = self.out_conv[i](sr_feature[i], out1)
            out2 += c2
            out.append(out2)
        return out[-1]


if __name__ == "__main__":
    model = AGSDF(6, 6, 3, 32, 8, 3)
    a = torch.rand(4, 6, 256, 256)
    print(model)