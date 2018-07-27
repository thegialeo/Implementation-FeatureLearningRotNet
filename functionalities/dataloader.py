import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_cifar(save_path):
    """
    Check if the CIFAR10 dataset already exists in the directory "save path". If not, the CIFAR10 dataset is downloaded.
    Returns trainset, testset and classes of CIFAR10. Applied transformations: ToTensor() and normalization.

    :param save_path: subdirectory, where the CIFAR10 dataset should be load from or downloaded to
    :return: trainset, testset, classes of CIFAR10
    """

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x/255.0 for x in [125.3, 123.0, 113.9]], [x/255.0 for x in [63.0, 62.1, 66.7]])]) #(0.49139968, 0.48215841, 0.44653091),
                                                                                    #(0.24703223, 0.24348513, 0.26158784)

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([x/255.0 for x in [125.3, 123.0, 113.9]], [x/255.0 for x in [63.0, 62.1, 66.7]])]) #(0.49139968, 0.48215841, 0.44653091),
                                                               #(0.24703223, 0.24348513, 0.26158784)

    trainset = datasets.CIFAR10(root=save_path, train=True, transform=train_transform, download=True)
    testset = datasets.CIFAR10(root=save_path, train=False, transform=transform, download=True)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return trainset, testset, classes


def make_dataloaders(trainset, testset, valid_size, batch_size, subset=None):
    """
    Create loaders for the train-, validation- and testset.

    :param trainset: trainset for which a train loader and valication loader will be created
    :param testset: testset for which we want to create a test loader
    :param valid_size: size of the dataset wrapped by the validation loader
    :param batchsize: size of the batch the loader will load during training
    :param subset: number of images per category (maximum: 5000 -> corresponds to whole training set)
    :return: trainloader, validloader, testloader
    """

    if subset is None:
        indices = torch.randperm(len(trainset))
        train_idx = indices[:len(indices) - valid_size]
        valid_idx = indices[len(indices) - valid_size:]

        trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx))

        validloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_idx))

        testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=batch_size)

        return trainloader, validloader, testloader

    else:
        subset_idx = []
        counter = list(0.0 for i in range(10))

        for index, label in enumerate(trainset.train_labels):
            if counter[label] < subset:
                subset_idx.append(index)
            counter[label] += 1

        trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_idx))

        testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=batch_size)

        return  trainloader, testloader

