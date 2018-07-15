import torch


def get_accuracy(loader, net, rot=None, printing=True, classifier=None, conv_block_num=None):
    '''
    Compute the accuracy of a neural network on a test or evaluation set wrapped by a loader. Optional: If rot is
    provided, the rotation prediction task is tested instead of the classification task (neural network is used for
    testing). Optional: If a classifier and conv_block_num is provided, the classifier is attached to the feature map of
    the x-th convolutional block of the neural network (where x = conv_block_num + 1) and tested on dataset wrapped by
    the loader.

    :param loader: loader that wraps the test or evaluation dataset
    :param net: the neural network that should be tested
    :param rot: list of classes for the rotation task. Possible classes are: '90', '180', '270'. Optional argument, if
    provided the neural network will be tested for the rotation task instead of the classification task.
    :param printing: if True, the accuracy will be additionally printed to the console
    :param classifier: optional argument, if provided, the classifier will be attached to the feature map of the x-th
    convolutional block of the neural network (where x = conv_block_num + 1) and tested on dataset wrapped by the
    loader.
    :param conv_block_num: convolutional block of the RotNet minus 1 to which the classifier should be attached to
    :return: accuracy
    '''
    