"""
https://github.com/FudanVI/FudanOCR/blob/main/text-gestalt/utils/ssim_psnr.py
"""
import string
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SRMetric"]


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand([channel, 1, window_size, window_size])
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = (
            (2 * mu1_mu2 + C1)
            * (2 * sigma12 + C2)
            / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        )
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean([1, 2, 3])

    def ssim(self, img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.shape
        window = self.create_window(window_size, channel)
        return self._ssim(img1, img2, window, window_size, channel, size_average)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.shape
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            self.window = window
            self.channel = channel
        return self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


class SRMetric:
    def __init__(self, main_indicator="all", **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-05
        self.psnr_result = []
        self.ssim_result = []
        self.calculate_ssim = SSIM()
        self.reset()

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.psnr_result = []
        self.ssim_result = []

    def calculate_psnr(self, img1, img2):
        mse = ((img1 * 255 - img2 * 255) ** 2).mean()
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in string.digits + string.ascii_letters, text)
        )
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        images_sr = pred_label["sr_img"]
        images_hr = pred_label["hr_img"]
        psnr = self.calculate_psnr(images_sr, images_hr)
        ssim = self.calculate_ssim(images_sr, images_hr)
        self.psnr_result.append(psnr)
        self.ssim_result.append(ssim)

    def get_metric(self):
        self.psnr_avg = sum(self.psnr_result) / len(self.psnr_result)
        self.psnr_avg = round(self.psnr_avg.item(), 6)
        self.ssim_avg = sum(self.ssim_result) / len(self.ssim_result)
        self.ssim_avg = round(self.ssim_avg.item(), 6)
        self.all_avg = self.psnr_avg + self.ssim_avg
        self.reset()
        return {
            "psnr_avg": self.psnr_avg,
            "ssim_avg": self.ssim_avg,
            "all": self.all_avg,
        }
