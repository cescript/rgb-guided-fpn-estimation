import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU()
    
    # relu( [conv->bn->relu->conv->bn] + [conv->bn] )
    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))