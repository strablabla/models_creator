"""U-Net for binary segmentation of 512x512 images."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Classic U-Net: `depth` poolings, channels base, 2*base, ... at each level.

    With depth 4 a 512x512 image goes down to 32x32 at the bottom.
    The output has `out_channels` logits (1 for a black/white mask).
    """

    def __init__(self, in_channels=3, out_channels=1, base=32, depth=4):
        super().__init__()
        self.config = dict(in_channels=in_channels, out_channels=out_channels, base=base, depth=depth)
        chans = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_channels
        for ch in chans:
            self.downs.append(DoubleConv(c, ch))
            c = ch
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for ch in reversed(chans[:-1]):
            self.ups.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.up_convs.append(DoubleConv(ch * 2, ch))
            c = ch
        self.head = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, conv in zip(self.ups, self.up_convs):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([skip, x], dim=1))
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
