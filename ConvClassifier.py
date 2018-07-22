import torch.nn as nn
import ConvBlock as cb
import GlobalAveragePooling as pool



class ConvClassifier(nn.Module):
    """
    A classifier consisting of one convolutional block.
    """


    def __init__(self, num_classes, in_channels):
        """
        Initialize a classifier object.

        :param num_classes: number of classes in the classification task
        :param in_channels: number of channels in the input feature map
        """

        super(ConvClassifier, self).__init__()

        self.classifier = nn.Sequential()
        self.classifier.add_module('Block3_Conv', cb.conv_block(in_channels, 192, 192, 192, 3, 1, 1))
        self.classifier.add_module('GlobalAveragePooling', pool.GlobalAveragePooling())
        self.classifier.add_module('Linear', nn.Linear(192, num_classes))

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
                if module.weight.requires_grad:
                    module.weight.data.fill_(1)
                if module.bias.requires_grad:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                feat_out = module.out_features
                std = np.sqrt(2.0 / feat_out)
                module.weight.data.normal_(0.0, std)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)