import numpy as np
import torch.nn as nn
import torch

def rgb_to_ycbcr(image):
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = 0.5
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)

def ycbcr_to_rgb(image):

    y image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    delta = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    r = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)

upsampler = nn.Upsample(scale_factor=4, mode='bilinear')

class Base7(nn.Module):
    def __init__(self):
        super(Base7, self).__init__()
        self.in_channels = 3
        self.out_channels = 3
        self.m = 4
        self.num_fea = 28
        self.scale = 4
        self.conv1 = nn.Conv2d(self.in_channels, self.num_fea, kernel_size=3, stride=1, padding=1)
        self.convs = nn.Sequential(*[nn.Sequential(nn.Conv2d(self.num_fea, self.num_fea, kernel_size=3, stride=1, padding=1),nn.ReLU(inplace=True)) for _ in range(self.m)])
        self.conv2 = nn.Conv2d(self.num_fea,self.out_channels*(self.scale**2),kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.out_channels*(self.scale**2),self.out_channels*(self.scale**2),kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(self.scale)
    def forward(self, inputs):
        out = inputs
        upsampled_inp = torch.cat([torch.cat([out[:, [i], ...] for _ in range(self.scale**2)], dim=1) for i in range(self.in_channels)], dim=1)
        out = self.conv1(out)
        out = torch.nn.functional.relu_(out)
        out = self.convs(out)     
        out = self.conv2(out)
        out = torch.nn.functional.relu_(out) 
        out = self.conv3(out)
        out = out + upsampled_inp
        out = self.ps(out)
        out = torch.clamp(out,min=0,max=1)
        return out

class Base7yuv(nn.Module): #my original model, only super-resolve Y channel
    def __init__(self):
        super(Base7, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.m = 4
        self.num_fea = 28
        self.scale = 4
        self.conv1 = nn.Conv2d(self.in_channels, self.num_fea, kernel_size=3, stride=1, padding=1)
        self.convs = nn.Sequential(*[nn.Sequential(nn.Conv2d(self.num_fea, self.num_fea, kernel_size=3, stride=1, padding=1),nn.ReLU(inplace=True)) for _ in range(self.m)])
        self.conv2 = nn.Conv2d(self.num_fea,self.out_channels*(self.scale**2),kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.out_channels*(self.scale**2),self.out_channels*(self.scale**2),kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(self.scale)
    def forward(self, inputs):

        ycbcrimage=rgb_to_ycbcr(inputs)
        out = torch.unsqueeze(ycbcrimage[...,0,:,:],1)
        upsampled_inp = torch.cat([out for _ in range(self.scale**2)],dim=1)
        out = self.conv1(out)
        out = torch.nn.functional.relu_(out)
        out = self.convs(out)     
        out = self.conv2(out)
        out = torch.nn.functional.relu_(out) 
        out = self.conv3(out)
        out = out + upsampled_inp
        out = self.ps(out)
        out = torch.cat((out,upsampler(ycbcrimage[...,[1,2],:,:])),1)
        out = torch.clamp(ycbcr_to_rgb(out),min=0,max=1)
        return out

