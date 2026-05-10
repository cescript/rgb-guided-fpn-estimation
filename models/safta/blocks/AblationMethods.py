import torch
import torch.nn as nn

# lightweight multiscale fusion module
class MultiScaleFusion(nn.Module):
    def __init__(self, rgb_ch=3, ir_ch=1, bc=24):
        super().__init__()
        self.rgb_s1 = nn.Sequential(nn.Conv2d(rgb_ch, bc, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.rgb_s2 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.rgb_s3 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.ir_s1 = nn.Sequential(nn.Conv2d(ir_ch, bc, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.ir_s2 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.ir_s3 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.fuse = nn.Sequential(nn.Conv2d(bc * 6, bc, kernel_size=1), nn.ReLU())

    def forward(self, x):
        rgb, ir = x[:, :3], x[:, 3:4]
        r1 = self.rgb_s1(rgb)
        r2 = self.rgb_s2(r1)
        r3 = self.rgb_s3(r2)
        i1 = self.ir_s1(ir)
        i2 = self.ir_s2(i1)
        i3 = self.ir_s3(i2)
        return self.fuse(torch.cat([r1, i1, self.up2(r2), self.up2(i2), self.up4(r3), self.up4(i3)], dim=1))

# lightweight attention module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.excitation(self.squeeze(x).view(b, c))
        return x * w.view(b, c, 1, 1)