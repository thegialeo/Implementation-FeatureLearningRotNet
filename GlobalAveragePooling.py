import torch.nn as nn
import torch.nn.functional as F



class GlobalAveragePooling(nn.Module):
    """
    Custom layer for global average pooling (GAP).
    """


    def __init__(self):

        super(GlobalAveragePooling, self).__init__()


    def forward(self, feat):

        num_channels = feat.size(1)

        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)
