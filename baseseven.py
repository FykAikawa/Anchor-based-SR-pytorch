import numpy as np
import torch.nn as nn
import torch

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)

def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
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
        out = torch.sigmoid(out)
        out = self.ps(out)
        out = torch.cat((out,upsampler(ycbcrimage[...,[1,2],:,:])),1)
        out = torch.clamp(ycbcr_to_rgb(out),min=0,max=1)
        return out

