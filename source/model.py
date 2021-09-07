import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)

        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class ResSiameseUnet(nn.Module):

    DEPTH = 6

    def __init__(self, n_classes = 2, resnet = None, input_channels = 3, last_up_conv_out_channels = 128):
        super().__init__()

        if resnet == None:
            resnet = models.resnet50(pretrained=True)

        down_blocks = []
        up_blocks = []
        fuse_blocks = []

        #Encode block using resnet50
        input_blocks = list(resnet.children())[:3]
        self.input_block = nn.Sequential(OrderedDict([
          ('conv1', input_blocks[0]),
          ('bn1', input_blocks[1]),
          ('relu', input_blocks[2])
        ]))

        self.input_pool = list(resnet.children())[3]
        self.encoded_out_channels = []

        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
                
                out_channel = getattr(bottleneck[-1], 'conv3', bottleneck[-1].conv2)
                self.encoded_out_channels.append(out_channel.out_channels)

        self.down_blocks = nn.ModuleList(down_blocks)

        for i in range(1, 4):
            fuse_blocks.append(Bridge(self.encoded_out_channels[-(i + 1)], self.encoded_out_channels[-(i + 2)]))
            up_blocks.append(Bridge(self.encoded_out_channels[-i], self.encoded_out_channels[-(i + 1)]))

        fuse_blocks.append(Bridge(self.input_block.conv1.out_channels*2, self.input_block.conv1.out_channels))
        fuse_blocks.append(Bridge(input_channels*2, input_channels))
        self.fuse_blocks = nn.ModuleList(fuse_blocks)

        self.bridge = Bridge(self.encoded_out_channels[-1]*2, self.encoded_out_channels[-1])

        up_blocks.append(UpBlockForUNetWithResNet50(
            in_channels=last_up_conv_out_channels + self.input_block.conv1.out_channels, 
            out_channels= last_up_conv_out_channels, 
            up_conv_in_channels=self.encoded_out_channels[0], 
            up_conv_out_channels=last_up_conv_out_channels)
        )

        up_blocks.append(UpBlockForUNetWithResNet50(
            in_channels=int(last_up_conv_out_channels/2) + input_channels, 
            out_channels=int(last_up_conv_out_channels/2), 
            up_conv_in_channels=last_up_conv_out_channels, 
            up_conv_out_channels=int(last_up_conv_out_channels/2)
        )

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(int(last_up_conv_out_channels/2), n_classes, kernel_size=1, stride=1)

    
    def encode(self, x):
        """
        Encode an input image using resnet50
        Return:
            x: encoded feature
            pre_pools: saved features during forward
        """
        pre_pools = dict()
        pre_pools[f"layer_0"] = x

        x = self.input_block(x)
        pre_pools[f"layer_1"] = x

        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (ResSiameseUnet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x
        
        return x, pre_pools
    
    def fuse_pools(self, pools_x, pools_y):
        """
        Fuse two pools to a pool
        """
        pools = dict()

        for i, block in enumerate(self.fuse_blocks, 1):
            key = f"layer_{ResSiameseUnet.DEPTH - 1 - i}"

            f = torch.cat([pools_x[key], pools_y[key]], 1)
            f = block(f)

            pools[key] = f
        
        return pools

    def decode_siamese(self, f, pools, with_output_feature_map=False):
        """
        Decode a encoded feature with pools
        """
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{ResSiameseUnet.DEPTH - 1 - i}"
            f = block(f, pools[key])
        return f

    def forward(self, x, y, with_output_feature_map=False):
        x, pools_x = self.encode(x)
        y, pools_y = self.encode(y)

        x = torch.cat([x, y], 1)

        x = self.bridge(x)
        
        pools = self.fuse_pools(pools_x, pools_y)

        x = self.decode_siamese(x, pools)
        
        output_feature_map = x
        x = self.out(x)

        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

    def load_resunet_state_dict(self, state_dict):
        input_block_state_dict = OrderedDict()
        down_blocks_state_dicts = []

        for i in range(4):
            down_blocks_state_dicts.append(OrderedDict())

        keys = state_dict.keys()

        for key in keys:
            if not key.startswith("layer"):
                input_block_state_dict[key] = state_dict[key]
            else:
                down_blocks_state_dicts[int(key[5]) - 1][key[7:]] = state_dict[key]

        self.input_block.load_state_dict(input_block_state_dict)

        for i, block in enumerate(self.down_blocks, 0):
            block.load_state_dict(down_blocks_state_dicts[i]) 
