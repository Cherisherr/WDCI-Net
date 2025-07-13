import numpy as np
import torch
import torch.fft
import torch.nn as nn
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F

class LossFre(nn.Module):
    def __init__(self):
        super(LossFre, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, imgs, gts):
        imgs = torch.fft.rfftn(imgs, dim=(2, 3))
        _real = imgs.real
        _imag = imgs.imag
        imgs = torch.cat([_real, _imag], dim=1)
        gts = torch.fft.rfftn(gts, dim=(2, 3))
        _real = gts.real
        _imag = gts.imag
        gts = torch.cat([_real, _imag], dim=1)
        return self.criterion(imgs, gts)


class LossSpa(torch.nn.Module):
    def __init__(self):
        super(LossSpa, self).__init__()

    def tensor2image(self, tensor):
        tensor = tensor.squeeze(0)
        numpy_image = tensor.cpu().detach().numpy().astype(np.float32)
        numpy_image = np.transpose(numpy_image)
        numpy_image = (numpy_image * 255).astype(np.uint8)
        return numpy_image

    def forward(self, imageA, imageB):
        b, _, _, _ = imageA.shape
        ssim = 0
        for i in range(b):
            image1 = imageA[i, :, :, :]
            image2 = imageB[i, :, :, :]
            image1 = self.tensor2image(image1)
            image2 = self.tensor2image(image2)
            a = 1 - compare_ssim(image1, image2, win_size=11, channel_axis=2, data_range=255)
            ssim += a
        return ssim / b


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, full=False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.full = full
        self.channel = 1
        self.window = None

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if self.window is None:
            real_size = min(self.window_size, img1.size(2), img2.size(2))
            self.window = self.create_window(real_size, channel).to(img1.device)
        window = self.window.type_as(img1)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if self.size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if self.full:
            return ret, cs

        return ret