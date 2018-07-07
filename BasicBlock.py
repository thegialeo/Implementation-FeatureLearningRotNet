import torch.nn as nn



class basic_block(nn.Module):
    '''
    A Basic Block consists of a convolutional layer + batch norm + ReLU.
    '''


    def __init__(self, in_channels, out_channels, kernel_size):
        super(basic_block, self).__init__()

        padding = (kernel_size - 1) / 2

        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_channels))
        self.layers.add_module('ReLU', nn.ReLU(inplace=True))


    def forward(self, x):
        return self.layers(x)
