import torch.nn as nn



class Flatten(nn.Module):
    """
    A custom module that will flatten the image. This layer is useful when the image needs to be forwarded to a
    fully-connected layer.
    """


    def __init__(self):
        """
        Initialize a Flatten layer object.
        """

        super(Flatten, self).__init__()


    def forward(self, feat):
        """
        Forward the feature map output of the RotNet 'feat' through the Flatten layer and return the output.

        :param feat: feature map that should be forwarded through the Flatten layer
        :return: result of forwarding the feature map through the Flatten layer
        """

        return feat.view(feat.size(0), -1)