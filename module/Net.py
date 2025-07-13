import torch
import torch.nn as nn
import torch.nn.functional as F
from module.lowfre import DownFRG
from module.highfre import upFRG
from module.HFCIM import HFCIM
from module.HFE_Block import HFE_Block
from module.highfre3 import upFRG3


class Net(nn.Module):
    def __init__(self, in_chn=3, wf=48, n_l_blocks=[3,2,1]):
        super(Net, self).__init__()
        self.ps_down1 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d((2**2) * in_chn, wf, 1, 1, 0)
        )
        self.ps_down2 = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d((4**2) * in_chn, wf, 1, 1, 0)
        )
        self.ps_down3 = nn.Sequential(
            nn.PixelUnshuffle(8),
            nn.Conv2d((8**2) * in_chn, wf, 1, 1, 0)
        )
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        # encoder of UNet-64
        prev_channels = 0
        self.down_group1 = DownFRG(wf, n_l_blocks=n_l_blocks[0])
        self.down_group2 = DownFRG(wf, n_l_blocks=n_l_blocks[1])
        self.down_group3 = DownFRG(wf, n_l_blocks=n_l_blocks[2])

        # decoder of UNet-64
        self.up_group3 = upFRG3(wf, n_l_blocks=n_l_blocks[2])
        self.up_group2 = upFRG(wf, n_l_blocks=n_l_blocks[1])
        self.up_group1 = upFRG(wf, n_l_blocks=n_l_blocks[0])

        self.last = nn.Conv2d(wf, in_chn, kernel_size=3, stride=1, padding=1, bias=True)
        self.hfe_Block = HFE_Block(wf)
        self.hfcim = HFCIM(wf)
    def forward(self,  x_left, x_right):
        imgL = x_left
        imgR = x_right


        img_downL1,img_downR1 = self.ps_down1(x_left),self.ps_down1(x_right)
        img_downL2,img_downR2 = self.ps_down2(x_left),self.ps_down2(x_right)
        img_downL3,img_downR3 = self.ps_down3(x_left),self.ps_down3(x_right)


        ##### shallow conv #####
        xL1 = self.conv_01(imgL)
        xR1 = self.conv_01(imgR)



        # Down-path (Encoder)
        x_l, xl1_HL, xl1_LH, xl1_HH = self.down_group1(xL1, img_downL1)
        x_R, xr1_HL, xr1_LH, xr1_HH = self.down_group1(xR1, img_downR1)
        xl1_HL, xl1_LH, xl1_HH,xr1_HL, xr1_LH, xr1_HH = self.hfcim(xl1_HL, xl1_LH, xl1_HH,xr1_HL, xr1_LH, xr1_HH)
        x_H1,x_J1,= self.hfe_Block(xl1_HL, xl1_LH, xl1_HH,xr1_HL, xr1_LH, xr1_HH)

        x_l, xl2_HL, xl2_LH, xl2_HH = self.down_group2(x_l, img_downL2)
        x_R, xr2_HL, xr2_LH, xr2_HH = self.down_group2(x_R, img_downR2)
        xl2_HL, xl2_LH, xl2_HH, xr2_HL, xr2_LH, xr2_HH = self.hfcim(xl2_HL, xl2_LH, xl2_HH, xr2_HL, xr2_LH, xr2_HH)
        x_H2, x_J2, = self.hfe_Block(xl2_HL, xl2_LH, xl2_HH, xr2_HL, xr2_LH, xr2_HH)

        x_l, xl3_HL, xl3_LH, xl3_HH = self.down_group3(x_l, img_downL3)
        x_R, xr3_HL, xr3_LH, xr3_HH = self.down_group3(x_R, img_downR3)
        xl3_HL, xl3_LH, xl3_HH, xr3_HL, xr3_LH, xr3_HH = self.hfcim(xl3_HL, xl3_LH, xl3_HH, xr3_HL, xr3_LH, xr3_HH)
        x_H3, x_J3 = self.hfe_Block(xl3_HL, xl3_LH, xl3_HH, xr3_HL, xr3_LH, xr3_HH)


        # Up-path (Decoder)
        x_l,x_l3 = self.up_group3(x_l, x_H3)
        x_l = self.up_group2(x_l, x_H2)
        x_l= self.up_group1(x_l, x_H1)


        x_R,x_R3= self.up_group3(x_R, x_J3)
        x_R = self.up_group2(x_R, x_J2)
        x_R = self.up_group1(x_R, x_J1)


        out_left = self.last(x_l) + imgL
        out_right = self.last(x_R) + imgR

        return out_left,out_right,x_l3,x_R3



if __name__ == "__main__":
    net = Net()
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    #print(net)