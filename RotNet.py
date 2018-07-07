import torch.nn as nn
import ConvBlock as cb
import GlobalAveragePooling as pool



class RotNet(nn.Module):
    '''
    The RotNet model consists of 3, 4 or 5 convolutional blocks. Furthermore, max pooling is applied between
    convolutional block 1 and 2, average pooling is applied between convolutional block 2 and 3 (optional:
    additionally applied between convolutional block 3 and 4). The convolutional blocks are followed by global average
    pooling and a linear classifier for the rotation task.
    '''


    def __init__(self, num_classes, num_conv_block=3, in_channels=3, avg_pool3=True):
        '''
        Initialize a RotNet object.

        :param num_classes: number of classes in the classification task
        :param num_conv_block: number of convolutional blocks (has to be at least 3 or more). Default: 3
        :param in_channels: number of channels in the input image. Default: 3
        :param avg_pool3: apply additional average pooling between convolutional block 3 and 4. Default: True
        '''

        super(RotNet, self).__init__()

        blocks = [nn.Sequential() for i in range(num_conv_block)]

        # convolutional block 1
        blocks[0].add_module('Block1_Conv', cb.conv_block(in_channels, 192, 160, 96, 5, 1, 1))
        blocks[0].add_module('Block1_MaxPool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # convolutional block 2
        blocks[1].add_module('Block2_Conv', cb.conv_block(96, 196, 192, 192, 5, 1, 1))
        blocks[1].add_module('Block2_AvgPool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        # convolutional block 3
        blocks[2].add_module('Block3_Conv', cb.conv_block(192, 192, 192, 192, 3, 1, 1))


