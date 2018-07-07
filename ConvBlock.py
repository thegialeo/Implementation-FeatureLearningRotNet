import torch.nn as nn
import BasicBlock as bb


class conv_block(nn.Module):
    '''
    A convolutional block consists of 3 Basic Blocks.
    '''


    def __init__(self, in_channels, out_channels_block1, out_channels_block2, out_channels_block3, kernel_size_block1, \
                 kernel_size_block2, kernel_size_block3):
        super(conv_block, self).__init__()

        self.layers = nn.Sequential()
        self.layers.add_module('ConvB1', bb.basic_block(in_channels, out_channels_block1, kernel_size_block1))
        self.layers.add_module('ConvB2', bb.basic_block(out_channels_block1, out_channels_block2, kernel_size_block2))
        self.layers.add_module('ConvB3', bb.basic_block(out_channels_block2, out_channels_block3, kernel_size_block3))