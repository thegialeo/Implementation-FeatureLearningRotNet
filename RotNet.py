import torch.nn as nn
import BasicBlock as bb
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


