import torch
import torch.nn as nn
from module.SKFF import SKFF


class HFCIM(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.h_fusion = SKFF(c, height=3, reduction=8)
    def forward(self, xl_HL, xl_LH, xl_HH,xr_HL, xr_LH, xr_HH):
        x_l= self.h_fusion([xl_HL, xl_LH, xl_HH])
        x_r = self.h_fusion([xr_HL, xr_LH, xr_HH])



        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        xl_HL1 = self.l_proj2(xl_HL).permute(0, 2, 3, 1)  # B, H, W, c
        xl_LH1 = self.l_proj2(xl_LH).permute(0, 2, 3, 1)  # B, H, W, c
        xl_HH1 = self.l_proj2(xl_HH).permute(0, 2, 3, 1)  # B, H, W, c

        xr_HL1 = self.r_proj2(xr_HL).permute(0, 2, 3, 1)  # B, H, W, c
        xr_LH1 = self.r_proj2(xr_LH).permute(0, 2, 3, 1)  # B, H, W, c
        xr_HH1 = self.r_proj2(xr_HH).permute(0, 2, 3, 1)  # B, H, W, c

        #V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        #V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        xr_HL1 = torch.matmul(torch.softmax(attention, dim=-1), xr_HL1)  # B, H, W, c
        xr_LH1 = torch.matmul(torch.softmax(attention, dim=-1), xr_LH1)  # B, H, W, c
        xr_HH1 = torch.matmul(torch.softmax(attention, dim=-1), xr_HH1)  # B, H, W, c

        xl_HL1 = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), xl_HL1)  # B, H, W, c
        xl_LH1 = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), xl_LH1)  # B, H, W, c
        xl_HH1 = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), xl_HH1)  # B, H, W, c

        # scale
        xr_HL1 = xr_HL1.permute(0, 3, 1, 2) * self.beta
        xr_LH1 = xr_LH1.permute(0, 3, 1, 2) * self.beta
        xr_HH1 = xr_HH1.permute(0, 3, 1, 2) * self.beta

        xl_HL1 = xl_HL1.permute(0, 3, 1, 2) * self.gamma
        xl_LH1 = xl_LH1.permute(0, 3, 1, 2) * self.gamma
        xl_HH1 = xl_HH1.permute(0, 3, 1, 2) * self.gamma

        xl_HL = xl_HL + xr_HL1
        xl_LH = xl_LH + xr_LH1
        xl_HH = xl_HH + xr_HH1

        xr_HL = xr_HL + xl_HL1
        xr_LH = xr_LH + xl_LH1
        xr_HH = xr_HH + xl_HH1


        return  xl_HL, xl_LH, xl_HH,xr_HL, xr_LH, xr_HH


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None