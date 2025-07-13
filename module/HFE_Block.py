import torch
import torch.nn as nn
from module.DTEM import DTEM


class HFE_Block(nn.Module):
    def __init__(self, dim):
        super(HFE_Block, self).__init__()

        self.high_enhance0 = DTEM(in_channels=dim, out_channels=2 * dim)


    def forward(self, xl_HL, xl_LH, xl_HH,xr_HL, xr_LH, xr_HH):

        x_h_l = torch.cat((xl_HL, xl_LH, xl_HH), 0)
        x_h_l = self.high_enhance0(x_h_l)


        x_h_r = torch.cat((xr_HL, xr_LH, xr_HH), 0)
        x_h_r = self.high_enhance0(x_h_r)


        return x_h_l,x_h_r