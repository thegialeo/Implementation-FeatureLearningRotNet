import torch.nn as nn
import BasicBlock as bb



class conv_block(nn.Module):
    """
    A convolutional block consists of 3 Basic Blocks.
    """


    def __init__(self, in_channels, out_channels_block1, out_channels_block2, out_channels_block3, kernel_size_block1, \
                 kernel_size_block2, kernel_size_block3):
        """
        Initialize a convolutional block object.

        :param in_channels: number of channels in the input image
        :param out_channels_block1: number of channels produced by the convolution of the Basic Block 1
        :param out_channels_block2: number of channels produced by the convolution of the Basic Block 2
        :param out_channels_block3: number of channels produced by the convolution of the Basic Block 3
        :param kernel_size_block1: size of the convolving kernel for the Basic Block 1
        :param kernel_size_block2: size of the convolving kernel for the Basic Block 2
        :param kernel_size_block3: size of the convolving kernel for the Basic Block 3
        """

        super(conv_block, self).__init__()

        self.layers = nn.Sequential()
        self.layers.add_module('ConvB1', bb.basic_block(in_channels, out_channels_block1, kernel_size_block1))
        self.layers.add_module('ConvB2', bb.basic_block(out_channels_block1, out_channels_block2, kernel_size_block2))
        self.layers.add_module('ConvB3', bb.basic_block(out_channels_block2, out_channels_block3, kernel_size_block3))


    def forward(self, x):
        """
        Forward an image 'x' through the convolutional block and return the output.

        :param x: input image that should be forwarded through the convolutional block
        :return: result of forwarding the image through the convolutional block
        """

        return self.layers(x)
