import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from .parts import *

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

        if input_blocks[0].in_channels != input_channels:
            input_blocks[0] = nn.Conv2d(input_channels, 64, padding=3, kernel_size=7, stride=2)
            
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
            fuse_blocks.append(Bridge(self.encoded_out_channels[-(i + 1)]*2, self.encoded_out_channels[-(i + 1)]))
            up_blocks.append(UpBlockForUNetWithResNet50(self.encoded_out_channels[-i], self.encoded_out_channels[-(i + 1)]))

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
            in_channels=int(last_up_conv_out_channels/2 + input_channels), 
            out_channels=int(last_up_conv_out_channels/4), 
            up_conv_in_channels=last_up_conv_out_channels, 
            up_conv_out_channels=int(last_up_conv_out_channels/2)
        ))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(int(last_up_conv_out_channels/4), n_classes, kernel_size=1, stride=1)

    
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

            f = torch.abs(pools_x[key] -  pools_y[key])
            #f = block(f)

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

        x = self.decode_siamese(f = x, pools = pools)
        
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
