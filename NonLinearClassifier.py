import torch.nn as nn
import math
import numpy as np
import Flatten as fl



class NonLinearClassifier(nn.Module):
    """
    A classifier that consists of 3 fully-connected layers. The 2 hidden layers have 200 feature channels each and are
    followed by batch-norm and ReLU units.
    """


    def __init__(self, num_classes, in_channels):
        """
        Initialize a classifier object.

        :param num_classes: number of classes in the classification task
        :param in_channels: number of channels in the input feature map
        """

        super(NonLinearClassifier, self).__init__()

        self.classifier = nn.Sequential()
        self.classifier.add_module('Flatten', fl.Flatten())

        # 1st fully-connected layer
        self.classifier.add_module('Linear_1', nn.Linear(in_channels, 200, bias=False))
        self.classifier.add_module('BatchNorm_1', nn.BatchNorm1d(200))
        self.classifier.add_module('ReLU_1', nn.ReLU(inplace=True))

        # 2nd fully-connected layer
        self.classifier.add_module('Linear_2', nn.Linear(200, 200, bias=False))
        self.classifier.add_module('BatchNorm_2', nn.BatchNorm1d(200))
        self.classifier.add_module('ReLU_2', nn.ReLU(inplace=True))

        # 3rd fully.connected layer
        self.classifier.add_module('Linear_3', nn.Linear(200, num_classes))

        self.weight_init()


    def forward(self, feat):
        """
        Forward the feature map output of the RotNet 'feat' through the classifier and return the output.

        :param feat: feature map that should be forwarded through the classifier
        :return: result of forwarding the feature map through the classifier
        """
        
        return self.classifier(feat)


    def weight_init(self):
        """
        Initialize the weights for all layers of the classifier.
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.fill_(1)
                module.bias.data.zero()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                feat_out = module.out_features
                std = np.sqrt(2.0 / feat_out)
                module.weight.data.normal_(0.0, std)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
