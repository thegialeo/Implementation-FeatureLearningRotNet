import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_cifar(save_path):
    '''
    Check if the CIFAR10 dataset already exists in the directory "save path". If not, the CIFAR10 dataset is downloaded.
    Returns trainset, testset and classes of CIFAR10. Applied transformations: ToTensor() and normalization.

    :param save_path: subdirectory, where the CIFAR10 dataset should be load from or downloaded to
    :return: trainset, testset, classes
    '''

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49139968, 0.48215841, 0.44653091), \
                                                                                (0.24703223, 0.24348513, 0.26158784))])

    trainset = datasets.CIFAR10(root=save_path, train=True, transform=transform, download=True)
    testset = datasets.CIFAR10(root=save_path, train=False, transform=transform, download=True)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return trainset, testset, classes

