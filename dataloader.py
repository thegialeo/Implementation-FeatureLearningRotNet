import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_cifar(save_path):
    '''
    Check if the CIFAR10 dataset already exists in the directory "save path". If not, the CIFAR10 dataset is downloaded.
    Returns trainset and testset of CIFAR10. Applied transformations: ToTensor() and normalization.

    :param save_path: subdirectory, where the CIFAR10 dataset should be load from or downloaded to
    :return:
    '''
    pass