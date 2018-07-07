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


