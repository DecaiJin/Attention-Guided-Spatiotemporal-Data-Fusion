import torch
import torch.nn as nn
import torch.nn.functional as F
from model.discriminator_arch import UNetDiscriminatorSN
from model.vgg_arch import VGGFeatureExtractor
from guided_filter_pytorch.guided_filter import FastGuidedFilter


class DeltaLoss(nn.Module):
    def __init__(self, factor):
        super(DeltaLoss, self).__init__()
        self.factor = factor
        self.mse = nn.MSELoss()

    def forward(self, dlc, dlf):
        N, C, H, W = dlf.size()
        assert H % self.factor == 0 and W % self.factor == 0
        dlf = F.unfold(dlf, kernel_size=self.factor, stride=self.factor, padding=0)
        dlf = dlf.view(N, C, int(self.factor**2), int(H*W/self.factor**2))
        dlf = torch.mean(dlf, dim=2).view(N, C, int(H/self.factor), int(W/self.factor))
        loss = self.mse(dlc, dlf)
        return loss


class ArtifactLoss(nn.Module):
    def __init__(self):
        super(ArtifactLoss, self).__init__()
        self.guide_filter = FastGuidedFilter(3)

    def forward(self, img_gt, img_pre, c1, c2, f1):
        sr = self.guide_filter(c1, c2, f1)
        weight_map = get_refined_artifact_map(img_gt, sr, ksize=3)
        art_loss = F.l1_loss(torch.mul(weight_map, img_gt),
                             torch.mul(weight_map, img_pre))
        return art_loss


class PerceptualLoss(nn.Module):
    def __init__(self, num_channels):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(
            channels=num_channels,
            layer_name_list=['conv3_1'],
            range_norm=False)

    def forward(self, img_gt, img_pre):
        gt_fea = self.vgg(img_gt)
        pre_fea = self.vgg(img_pre)
        p_loss = 0
        for k in gt_fea.keys():
            p_loss += F.l1_loss(gt_fea[k].detach(), pre_fea[k])
        return p_loss


class L1Loss(nn.Module):
    def __init__(self, channel_weight=None):
        super(L1Loss, self).__init__()
        self.channel_weight = channel_weight

    def forward(self, img_gt, img_pre):
        if self.channel_weight is not None and len(self.channel_weight) == img_gt.shape[1]:
            loss_pix = 0
            for w in range(len(self.channel_weight)):
                loss_pix += F.l1_loss(img_pre[:, w], img_gt[:, w]) * self.channel_weight[w]
            return loss_pix

        else:
            return F.l1_loss(img_pre, img_gt.detach())


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


def get_local_weights(residual, ksize):

    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')

    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True,
                                   keepdim=True).squeeze(-1).squeeze(-1)
    return pixel_level_weight


def get_refined_artifact_map(img_gt, img_sr, ksize):

    residual_sr = torch.sum(torch.abs(img_gt - img_sr), 1, keepdim=True)
    patch_level_weight = torch.var(residual_sr.clone(), dim=(-1, -2, -3), keepdim=True) ** (1/5)
    pixel_level_weight = get_local_weights(residual_sr.clone(), ksize)  # M
    overall_weight = torch.tanh(patch_level_weight * pixel_level_weight)
    weight_min = overall_weight.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    weight_max = overall_weight.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    weight = (overall_weight - weight_min) / (weight_max - weight_min)
    return weight


if __name__ == "__main__":

    pass
        
