import torch.nn as nn
from .parts import *

class ResUnetSegmentation(nn.Module):
    def __init__(self, input_channels = 3, n_classes = 2, encoder_depth = 6):
        super(ResUnetSegmentation, self).__init__()

        self.input_channels = 3
        self.n_classes = 2
        self.encoder_depth = 6

        self.encoder = UnetEncoder(input_channels = input_channels, save_output = True, depth = encoder_depth)
        self.bridge = Bridge(2048, 2048)

        up_blocks = []
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(
            in_channels=128 + 64, 
            out_channels=128, 
            up_conv_in_channels=256, 
            up_conv_out_channels=128)
        )
        up_blocks.append(UpBlockForUNetWithResNet50(
            in_channels=64 + 3, 
            out_channels=64, 
            up_conv_in_channels=128, 
            up_conv_out_channels=64)
        )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, n_classes, kernel_size = 1, stride = 1)
    
    def forward(self, x):
        out, pre_pools = self.encoder(x)
        out = self.bridge(out)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{self.encoder_depth - 1 - i}"
            out = block(out, pre_pools[key])

        out = self.out(out)
        del pre_pools
        return out