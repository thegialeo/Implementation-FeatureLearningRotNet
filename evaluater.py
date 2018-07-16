import torch
import rotation as rtt


# This metric function is copy pasted from the paper "Unsupervised Representation Learning by Predicting Image
# Rotations" by Spyros Gidaris, Praveer Singh, Nikos Komodakis.
# The code can be found at: "https://github.com/gidariss/FeatureLearningRotNet"
# The function was renamed from "accuracy()" to "accuracy_from_paper()"
def accuracy_from_paper(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_accuracy(loader, net, rot=None, printing=True, classifier=None, conv_block_num=None, use_paper_metric=False):
    """
    Compute the accuracy of a neural network on a test or evaluation set wrapped by a loader.

    Optional: If rot is provided, the rotation prediction task is tested instead of the classification task (neural
    network is used for testing).

    Optional: If a classifier and conv_block_num is provided, the classifier is attached to the feature map of
    the x-th convolutional block of the neural network (where x = conv_block_num) and tested on dataset wrapped by
    the loader.

    :param loader: loader that wraps the test or evaluation dataset
    :param net: the neural network that should be tested
    :param rot: list of classes for the rotation task. Possible classes are: '90', '180', '270'. Optional argument, if
    provided the neural network will be tested for the rotation task instead of the classification task.
    :param printing: if True, the accuracy will be additionally printed to the console
    :param classifier: optional argument, if provided, the classifier will be attached to the feature map of the x-th
    convolutional block of the neural network (where x = conv_block_num) and tested on dataset wrapped by the
    loader.
    :param conv_block_num: number of the RotNet convolutional block to which the classifier will be attached
    :param use_paper_metric: use the metric from the paper "Unsupervised Representation Learning by Predicting Image
    Rotations" by Spyros Gidaris, Praveer Singh, Nikos Komodakis. Default: False
    :return: accuracy
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conv_block_num -= 1

    correct = 0.0
    total = 0.0

    if use_paper_metric:
        accuracy_lst = []

    with torch.no_grad():
        for data in loader:
            images, labels = data
            net.to(device)
            if rot == None:
                images, labels = images.to(device), labels.to(device)
                if classifier == None:
                    outputs = net(images)
                    if use_paper_metric:
                        tmp_accuracy = accuracy_from_paper(outputs, labels)[0].item()
                        accuracy_lst.append(tmp_accuracy)
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                else:
                    feats = net(images, out_feat_keys=[net.all_feat_names[conv_block_num]])
                    outputs = classifier(feats)
                    if use_paper_metric:
                        tmp_accuracy = accuracy_from_paper(outputs, labels)[0].item()
                        accuracy_lst.append(tmp_accuracy)
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
            else:
                rot_images, class_labels, rot_labels = rtt.create_rot_batch(images, labels, rot=rot)
                rot_images, rot_labels = rot_images.to(device), rot_labels.to(device)
                outputs = net(rot_images)
                if use_paper_metric:
                    tmp_accuracy = accuracy_from_paper(outputs, rot_labels)[0].item()
                    accuracy_lst.append(tmp_accuracy)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    total += rot_labels.size(0)
                    correct += (predicted == rot_labels.long()).sum().item()

    if use_paper_metric:
        accuracy = sum(accuracy_lst) / float(len(accuracy_lst))
    else:
        accuracy = 100.0 * correct / total

    if printing:
        print('Accuracy: {: .3f} %'.format(accuracy))

    return accuracy


def get_class_accuracy(num_class, loader, net, class_names, rot=None, printing=True, classifier=None,
                       conv_block_num=None):
    """
    Computes the accuracy of a neural network for every class on a test or evaluation set wrapped by a loader.

    Optional: If rot is provided, the rotation prediction task is tested instead of the classification task (neural
    network is used for testing).

    Optional: If a classifier and conv_block_num is provided, the classifier is attached to the feature map of
    the x-th convolutional block of the neural network (where x = conv_block_num) and tested on dataset wrapped by
    the loader.

    :param num_class: number of total classes in the classification task
    :param loader: loader that wraps the test or evaluation dataset
    :param net: the neural network that should be tested
    :param class_names: list of class names corresponding the labels
    :param rot: list of classes for the rotation task. Possible classes are: '90', '180', '270'. Optional argument, if
    provided the neural network will be tested for the rotation task instead of the classification task.
    :param printing: if True, the accuracies will be additionally printed to the console
    :param classifier: optional argument, if provided, the classifier will be attached to the feature map of the x-th
    convolutional block of the neural network (where x = conv_block_num) and tested on dataset wrapped by the
    loader.
    :param conv_block_num: number of the RotNet convolutional block to which the classifier will be attached
    :return: list of accuracy of every class
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conv_block_num -= 1

    class_correct = list(0.0 for i in range(num_class))
    class_total = list(0.0 for i in range(num_class))

    accuracy = []

    with torch.no_grad():
        for data in loader:
            images, labels = data
            net.to(device)
            if rot == None:
                images, labels = images.to(device), labels.to(device)
                if classifier == None:
                    outputs = net(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(len(labels)):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                else:
                    feats = net(images, out_feat_keys=[net.all_feat_names[conv_block_num]])
                    outputs = classifier(feats)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(len(labels)):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
            else:
                rot_images, class_labels, rot_labels = rtt.create_rot_batch(images, labels, rot=rot)
                rot_images, rot_labels = rot_images.to(device), rot_labels(device)
                outputs = net(rot_images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == rot_labels.long()).squeeze()
                for i in range(len(rot_labels)):
                    label = rot_labels[i].int()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    for i in range(num_class):
        class_accuracy = 100 * class_correct[i] / class_total[i]
        accuracy.append(class_accuracy)
        if printing:
            print('Accuracy of {} : {: .3f} %'.format(class_names[i], class_accuracy))

    return accuracy
