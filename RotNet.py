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


    def __init__(self, num_classes, num_conv_block=3, in_channels=3, add_avg_pool=True):
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

        # optional average pooling
        if num_conv_block > 3 and add_avg_pool:
            blocks[2].add_module('Block3_AvgPool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        # add generic blocks, if more than 3 convolutional blocks (no pooling)
        for block in range(3, num_conv_block):
            blocks[block].add_module('Block{}_Conv'.format(block+1), cb.conv_block(192, 192, 192, 192, 3, 1, 1))

        # add global average pooling + linear classifier layer
        blocks.append(nn.Sequential())
        blocks[-1].add_module('GlobalAveragePooling', pool.GlobalAveragePooling())
        blocks[-1].add_module('Classifier', nn.Linear(192,num_classes))

        # create name structures for the network
        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = ['conv{}'.format(block + 1) for block in range(num_conv_blocks)] + ['classifier', ]


    def find_highest_feature(self, out_feat_keys):
        '''
        Finds the highest output feature name in out_feat_keys. Default: return the name of the feature output of the
        last layer. (Here, highest output feature means: The "deepest" feature map "farthest" away from the input)

        :param out_feat_keys: list of feature names. Possible feature names are: 'conv1', 'conv2', ..., 'convX',
        'classifier' with X = number of convolutional blocks in the network
        :return: out_feat_keys and max_out_feat (the name of the highest output feature in out_feat_keys)
        '''

        out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat


    def forward(self, x, out_feat_keys=None):
        '''
        Forward an image 'x' through the RotNet and return the asked output features.

        :param x: input image that should be forwarded through the RotNet
        :param out_feat_keys: list/tuple with feature names of features that should be returned. Default: return the
        highest feature
        :return: If multiple output feature were asked then a list of output features is returned in the same order as
        in 'out_feat_keys'. If a single output feature was asked then that one output feature is returned (not as a
        list).
        '''

        out_feat_keys, max_out_feat = self.find_highest_feature(out_feat_keys)

        out_feats = [None] * len(out_feat_keys)

        feat = x
        for i in range(max_out_feat + 1):
            feat = self._feature_blocks[i](feat)
            key = self.all_feat_names[i]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats

        return out_feats


    def weight_init(self):
        '''
        Initialize the weights for all layers of the RotNet.
        '''

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.weight.requires_grad:
                    n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                    module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad:
                    module.weight.data.fill_(1)
                if module.bias.requires_grad:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                if module.bias.requires_grad:
                    module.bias.data.zero()