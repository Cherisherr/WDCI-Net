import torch
import torch.nn as nn
import torch.nn.functional as F
from module.wavelet import IWT
from module.IAM import IAM


class upFRG3(nn.Module):
    def __init__(self, dim, n_l_blocks=1):
        super().__init__()
        self.iwt = IWT()
        self.l_blk = nn.Sequential(*[IAM(dim) for _ in range(n_l_blocks)])

        self.conv = nn.Conv2d(dim,3,3,1,1)
    def forward(self, x_l, x_h):

        for l_layer in self.l_blk:
            x_l = l_layer(x_l)

        x_l1 = self.conv(x_l)

        n, c, h, w = x_l.shape
        x_LH, x_HL, x_HH = x_h[:n, ...],x_h[n:2*n, ...],x_h[2*n:3*n, ...]

        x_h = torch.cat((x_LH, x_HL, x_HH), dim=1)

        x_l = self.iwt(torch.cat([x_l, x_h], dim=1))

        return x_l,x_l1
