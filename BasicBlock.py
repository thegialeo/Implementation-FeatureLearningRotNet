import torch.nn as nn



class basic_block(nn.Module):
    """
    A Basic Block consists of a convolutional layer + batch norm + ReLU.
    """


    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Initialize a Basic Block object.

        :param in_channels: number of channels in the input image
        :param out_channels: number of channels produced by the convolution layer
        :param kernel_size: size of the convolving kernel
        """

        super(basic_block, self).__init__()

        padding = (kernel_size - 1) / 2

        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_channels))
        self.layers.add_module('ReLU', nn.ReLU(inplace=True))


    def forward(self, x):
        """
        Forward an image 'x' through the Basic Block and return the output.

        :param x: input image that should be forwarded through the Basic Block
        :return: result of forwarding the image through the Basic Block
        """

        return self.layers(x)