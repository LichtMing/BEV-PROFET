import torch
import torch.nn as nn
import torch.nn.functional as F

class UP(nn.Module):
    def __init__(self, in_ch, out_ch, Transpose=False):
        super(UP, self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        else:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.up(x)
        skip = self.conv(x)
        up = F.interpolate(skip, x.shape[-2:])
        x = x + up
        return x