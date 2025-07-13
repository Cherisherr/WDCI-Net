import torch
import torch.nn as nn
from module.wavelet import DWT
from module.IAM import IAM


class DownFRG(nn.Module):
    def __init__(self, dim, n_l_blocks=1):
        super().__init__()
        self.dwt = DWT()
        self.l_conv = nn.Conv2d(dim * 2, dim, 3, 1, 1)
        self.l_blk = nn.Sequential(*[IAM(dim) for _ in range(n_l_blocks)])


    def forward(self, x, x_d):
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)

        x_LL = self.l_conv(torch.cat([x_LL, x_d], dim=1))

        for l_layer in self.l_blk:
            x_LL = l_layer(x_LL)



        return x_LL,x_HL, x_LH, x_HH